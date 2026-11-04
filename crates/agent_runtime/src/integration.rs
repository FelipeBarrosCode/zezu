use std::{
    fs::{File, OpenOptions},
    io::{Read as _, Seek as _, SeekFrom},
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use crate::{RuntimeStore, orchestration::TaskIdentity};
use anyhow::{Context as _, Result, bail, ensure};
use rusqlite::{OptionalExtension as _, params};
use serde::{Deserialize, Serialize};

const MAXIMUM_OUTPUT: u64 = 64 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationCheck {
    pub program: String,
    pub arguments: Vec<String>,
    pub timeout_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationRequest {
    pub id: String,
    pub agent_id: String,
    pub generation: i64,
    #[serde(default)]
    pub source_owner: Option<TaskIdentity>,
    pub target_branch: String,
    pub source_commit: String,
    pub checks: Vec<IntegrationCheck>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationRecord {
    pub request: IntegrationRequest,
    pub state: String,
    pub before_commit: String,
    pub integrated_commit: Option<String>,
    pub reverted_commit: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct GitIntegration {
    store: RuntimeStore,
    repository: PathBuf,
    directory: PathBuf,
    busy: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
}

struct OwnedChild(std::process::Child);
impl std::ops::Deref for OwnedChild {
    type Target = std::process::Child;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for OwnedChild {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if let Err(error) = terminate_check(&mut self.0) {
            log::error!("owned integration process cleanup failed: {error:#}");
        }
    }
}

struct Admission(Arc<AtomicBool>);
impl Drop for Admission {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

struct IntegrationOwner(File);
impl Drop for IntegrationOwner {
    fn drop(&mut self) {
        // A forked process can retain the file description after this handle closes.
        if let Err(error) = self.0.unlock() {
            log::error!("integration owner unlock failed: {error:#}");
        }
    }
}

impl GitIntegration {
    pub async fn open(store: RuntimeStore, repository: PathBuf) -> Result<Self> {
        let (repository, directory) = worker(move || {
            let repository = repository.canonicalize()?;
            let common = checked_git(
                &repository,
                ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            )?;
            let directory = PathBuf::from(common).join("zezu-integration");
            std::fs::create_dir_all(directory.join("empty-hooks"))?;
            Ok((repository, directory))
        })
        .await?;
        store
            .write(|connection| {
                connection.execute_batch(
                    "CREATE TABLE IF NOT EXISTS runtime_integrations (
                id TEXT PRIMARY KEY, repository TEXT NOT NULL, agent_id TEXT NOT NULL REFERENCES agents(id),
                request TEXT NOT NULL, state TEXT NOT NULL, before_commit TEXT NOT NULL,
                integrated_commit TEXT, reverted_commit TEXT, error TEXT
            );
            CREATE TABLE IF NOT EXISTS runtime_integration_worktrees (
                path TEXT PRIMARY KEY, repository TEXT NOT NULL, operation_id TEXT NOT NULL
            );",
                )?;
                Ok(())
            })
            .await?;
        Ok(Self {
            store,
            repository,
            directory,
            busy: Arc::new(AtomicBool::new(false)),
            stopping: Arc::new(AtomicBool::new(false)),
        })
    }

    pub async fn record(&self, id: String) -> Result<Option<IntegrationRecord>> {
        let repository = self.repository_key()?;
        self.store.read(move |connection| {
            connection.query_row("SELECT request,state,before_commit,integrated_commit,reverted_commit,error FROM runtime_integrations WHERE id=? AND repository=?", params![id,repository], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, Option<String>>(3)?, row.get::<_, Option<String>>(4)?, row.get::<_, Option<String>>(5)?))
            }).optional()?.map(|row| Ok(IntegrationRecord { request: serde_json::from_str(&row.0)?, state: row.1, before_commit: row.2, integrated_commit: row.3, reverted_commit: row.4, error: row.5 })).transpose()
        }).await
    }

    pub async fn integrate(&self, request: IntegrationRequest) -> Result<IntegrationRecord> {
        validate_request(&request)?;
        let admission = self.admit()?;
        let this = self.clone();
        worker(move || {
            let _admission = admission;
            let _owner = this.owner()?;
            futures::executor::block_on(async {
                if let Some(record) = this.record(request.id.clone()).await? {
                    ensure!(
                        record.request == request,
                        "integration ID was reused for different work"
                    );
                    ensure!(
                        record.state == "integrated",
                        "integration requires recovery: {}",
                        record.state
                    );
                    return Ok(record);
                }
                this.check_generation(&request).await?;
                this.checked_git(
                    &this.repository,
                    ["check-ref-format", "--branch", &request.target_branch],
                )?;
                let before = this.branch_head(&request.target_branch)?;
                let source = this.checked_git(
                    &this.repository,
                    [
                        "rev-parse",
                        "--verify",
                        "--end-of-options",
                        &format!("{}^{{commit}}", request.source_commit),
                    ],
                )?;
                ensure!(
                    source == request.source_commit,
                    "integration requires an immutable full commit ID"
                );
                let record = IntegrationRecord {
                    request,
                    state: "checking".into(),
                    before_commit: before,
                    integrated_commit: None,
                    reverted_commit: None,
                    error: None,
                };
                this.save(record.clone(), true).await?;
                this.execute(record, false).await
            })
        })
        .await
    }

    pub async fn revert(
        &self,
        id: String,
        checks: Vec<IntegrationCheck>,
    ) -> Result<IntegrationRecord> {
        validate_checks(&checks)?;
        let admission = self.admit()?;
        let this = self.clone();
        worker(move || {
            let _admission = admission;
            let _owner = this.owner()?;
            futures::executor::block_on(async {
                let mut record = this
                    .record(id)
                    .await?
                    .context("integration does not exist")?;
                if record.state == "reverted" {
                    return Ok(record);
                }
                ensure!(
                    record.state == "integrated",
                    "integration is not ready for revert"
                );
                record.request.checks = checks;
                record.state = "reverting".into();
                this.save(record.clone(), false).await?;
                this.execute(record, true).await
            })
        })
        .await
    }

    pub async fn recover(&self) -> Result<Vec<IntegrationRecord>> {
        let admission = self.admit()?;
        let this = self.clone();
        worker(move || {
            let _admission = admission;
            let _owner = this.owner()?;
            futures::executor::block_on(async {
                let repository = this.repository_key()?;
                let ids = this.store.read(move |connection| {
                    let mut statement = connection.prepare("SELECT id FROM runtime_integrations WHERE repository=? AND state IN ('checking','prepared','publishing','reverting','revert-prepared','revert-publishing') ORDER BY id")?;
                    Ok(statement.query_map([repository], |row| row.get::<_, String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?)
                }).await?;
                let mut recovered = Vec::new();
                for id in ids {
                    let mut record = this.record(id).await?.context("integration disappeared")?;
                    let reverting = record.state.starts_with("revert");
                    let candidate = if reverting { record.reverted_commit.as_deref() } else { record.integrated_commit.as_deref() };
                    let head = this.branch_head(&record.request.target_branch)?;
                    let published = match candidate { Some(candidate) => this.contains(candidate, &head)?, None => false };
                    if published {
                        record.state = if reverting { "reverted" } else { "integrated" }.into();
                        record.error = None;
                    } else {
                        record.state = "uncertain".into();
                        record.error = Some("interrupted integration was not proven published; inspect its retained staging worktree and Git history before retrying".into());
                    }
                    this.save(record.clone(), false).await?;
                    recovered.push(record);
                }
                this.recover_target_worktrees().await?;
                Ok(recovered)
            })
        }).await
    }

    pub fn request_shutdown(&self) {
        self.stopping.store(true, Ordering::Release);
    }

    fn git<const COUNT: usize>(
        &self,
        directory: &Path,
        arguments: [&str; COUNT],
    ) -> Result<Output> {
        git_with_cancel(directory, arguments, Some(&self.stopping))
    }

    fn checked_git<const COUNT: usize>(
        &self,
        directory: &Path,
        arguments: [&str; COUNT],
    ) -> Result<String> {
        let output = self.git(directory, arguments)?;
        ensure!(
            output.status.success(),
            "Git integration failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(String::from_utf8(output.stdout)?.trim_end().to_owned())
    }

    fn repository_key(&self) -> Result<String> {
        Ok(self
            .directory
            .to_str()
            .context("integration repository path must be UTF-8")?
            .to_owned())
    }

    fn admit(&self) -> Result<Admission> {
        ensure!(
            !self.stopping.load(Ordering::Acquire),
            "integration owner is shutting down"
        );
        ensure!(
            self.busy
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            "another integration is active; retry after it finishes"
        );
        Ok(Admission(self.busy.clone()))
    }

    fn owner(&self) -> Result<IntegrationOwner> {
        let owner = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.directory.join("owner.lock"))?;
        owner
            .try_lock()
            .context("another process is integrating this repository")?;
        Ok(IntegrationOwner(owner))
    }

    fn branch_head(&self, branch: &str) -> Result<String> {
        self.checked_git(
            &self.repository,
            [
                "rev-parse",
                "--verify",
                "--end-of-options",
                &format!("refs/heads/{branch}^{{commit}}"),
            ],
        )
    }

    fn contains(&self, candidate: &str, head: &str) -> Result<bool> {
        let output = self.git(
            &self.repository,
            ["merge-base", "--is-ancestor", candidate, head],
        )?;
        match output.status.code() {
            Some(0) => Ok(true),
            Some(1) => Ok(false),
            _ => bail!(
                "failed to inspect integration ancestry: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        }
    }

    async fn check_generation(&self, request: &IntegrationRequest) -> Result<()> {
        let current = self.store.snapshot(request.agent_id.clone()).await?;
        ensure!(
            current.generation == request.generation,
            "integration belongs to a stale agent generation"
        );
        ensure!(
            !matches!(current.state.as_str(), "cancelled" | "paused" | "failed"),
            "agent is not allowed to integrate in state {}",
            current.state
        );
        if let Some(source) = &request.source_owner {
            let current = self.store.snapshot(source.agent_id.clone()).await?;
            ensure!(
                current.generation == source.generation,
                "integration source belongs to a stale generation"
            );
            ensure!(
                !matches!(current.state.as_str(), "cancelled" | "paused" | "failed"),
                "integration source is no longer authorized"
            );
        }
        Ok(())
    }

    async fn save(&self, record: IntegrationRecord, insert: bool) -> Result<()> {
        let repository = self.repository_key()?;
        self.store.write(move |connection| {
            let request = serde_json::to_string(&record.request)?;
            if insert {
                connection.execute("INSERT INTO runtime_integrations(id,repository,agent_id,request,state,before_commit,integrated_commit,reverted_commit,error) VALUES (?,?,?,?,?,?,?,?,?)", params![record.request.id,repository,record.request.agent_id,request,record.state,record.before_commit,record.integrated_commit,record.reverted_commit,record.error])?;
            } else {
                ensure!(connection.execute("UPDATE runtime_integrations SET request=?,state=?,before_commit=?,integrated_commit=?,reverted_commit=?,error=? WHERE id=? AND repository=?", params![request,record.state,record.before_commit,record.integrated_commit,record.reverted_commit,record.error,record.request.id,repository])? == 1, "integration record disappeared");
            }
            Ok(())
        }).await
    }

    async fn execute(
        &self,
        mut record: IntegrationRecord,
        reverting: bool,
    ) -> Result<IntegrationRecord> {
        let staging = self.directory.join(uuid::Uuid::new_v4().to_string());
        let result = self
            .prepare_and_publish(&mut record, reverting, &staging)
            .await;
        if staging.exists() {
            if let Err(error) = self.checked_git(
                &self.repository,
                [
                    "worktree",
                    "remove",
                    "--force",
                    staging.to_str().context("non-UTF8 staging path")?,
                ],
            ) {
                log::error!("integration staging cleanup failed: {error:#}");
            }
        }
        match result {
            Ok(()) => Ok(record),
            Err(error) => {
                let published = if reverting {
                    record.reverted_commit.as_deref()
                } else {
                    record.integrated_commit.as_deref()
                };
                let published = match published {
                    Some(commit) => {
                        self.contains(commit, &self.branch_head(&record.request.target_branch)?)?
                    }
                    None => false,
                };
                record.state = if published {
                    if reverting { "reverted" } else { "integrated" }
                } else if record.state.ends_with("publishing") {
                    "uncertain"
                } else if reverting {
                    "integrated"
                } else {
                    "failed"
                }
                .into();
                record.error = Some(format!("{error:#}"));
                self.save(record, false).await?;
                Err(error)
            }
        }
    }

    async fn prepare_and_publish(
        &self,
        record: &mut IntegrationRecord,
        reverting: bool,
        staging: &Path,
    ) -> Result<()> {
        let target = self.branch_head(&record.request.target_branch)?;
        if !reverting {
            ensure!(
                target == record.before_commit,
                "target moved before integration"
            );
            self.check_generation(&record.request).await?;
        }
        self.checked_git(
            &self.repository,
            [
                "worktree",
                "add",
                "--detach",
                staging.to_str().context("non-UTF8 staging path")?,
                &target,
            ],
        )?;
        if reverting {
            let commit = record
                .integrated_commit
                .as_deref()
                .context("integration has no commit")?;
            ensure!(
                self.contains(commit, &target)?,
                "integration is no longer in the target branch"
            );
            self.checked_git(staging, ["revert", "--no-commit", commit])?;
        } else {
            self.checked_git(
                staging,
                [
                    "merge",
                    "--squash",
                    "--no-commit",
                    "--",
                    &record.request.source_commit,
                ],
            )?;
        }
        let difference = self.git(staging, ["diff", "--cached", "--quiet"])?;
        ensure!(
            difference.status.code() == Some(1),
            "artifact produces no new changes"
        );
        let checked_tree = self.checked_git(staging, ["write-tree"])?;
        for (index, check) in record.request.checks.iter().enumerate() {
            self.run_check(staging, check, &record.request, reverting, index)
                .await?;
        }
        ensure!(
            self.checked_git(staging, ["write-tree"])? == checked_tree,
            "configured checks staged changes; review them before integration"
        );
        ensure!(
            self.checked_git(staging, ["diff", "--name-only"])?
                .is_empty(),
            "configured checks modified tracked files; changes must be reviewed before integration"
        );
        let message = format!(
            "{} agent changes\n\nZezu-Integration: {}",
            if reverting { "Revert" } else { "Integrate" },
            record.request.id
        );
        self.checked_git(staging, ["commit", "--no-gpg-sign", "-m", &message])?;
        let candidate = self.checked_git(staging, ["rev-parse", "HEAD"])?;
        self.checked_git(
            &self.repository,
            [
                "update-ref",
                &format!("refs/zezu/integrations/{candidate}"),
                &candidate,
            ],
        )?;
        if reverting {
            record.reverted_commit = Some(candidate.clone());
        } else {
            record.integrated_commit = Some(candidate.clone());
        }
        record.state = if reverting {
            "revert-prepared"
        } else {
            "prepared"
        }
        .into();
        self.save(record.clone(), false).await?;
        ensure!(
            self.branch_head(&record.request.target_branch)? == target,
            "target moved while checks were running"
        );
        if !reverting {
            self.check_generation(&record.request).await?;
        }
        record.state = if reverting {
            "revert-publishing"
        } else {
            "publishing"
        }
        .into();
        self.save(record.clone(), false).await?;
        self.publish(
            &record.request.id,
            &record.request.target_branch,
            &target,
            &candidate,
        )
        .await?;
        record.state = if reverting { "reverted" } else { "integrated" }.into();
        record.error = None;
        self.save(record.clone(), false).await
    }

    async fn publish(
        &self,
        operation_id: &str,
        branch: &str,
        expected: &str,
        candidate: &str,
    ) -> Result<()> {
        let listing = self.checked_git(&self.repository, ["worktree", "list", "--porcelain"])?;
        let target_ref = format!("branch refs/heads/{branch}");
        for block in listing.split("\n\n") {
            if block.lines().any(|line| line == target_ref) {
                let path = block
                    .lines()
                    .find_map(|line| line.strip_prefix("worktree "))
                    .context("target worktree has no path")?;
                let path = Path::new(path);
                ensure!(
                    self.checked_git(path, ["status", "--porcelain"])?
                        .is_empty(),
                    "target worktree has uncommitted changes"
                );
                ensure!(
                    self.checked_git(path, ["rev-parse", "HEAD"])? == expected,
                    "target moved before publication"
                );
                self.checked_git(
                    path,
                    [
                        "merge",
                        "--ff-only",
                        "--no-edit",
                        "--no-overwrite-ignore",
                        candidate,
                    ],
                )?;
                return Ok(());
            }
        }
        let path = self
            .directory
            .join(format!("target-{}", uuid::Uuid::new_v4()));
        let key = path.to_str().context("target worktree path")?.to_owned();
        let operation_id = operation_id.to_owned();
        let repository = self.repository_key()?;
        self.store
            .write({
                let key = key.clone();
                move |connection| {
                    connection.execute(
                        "INSERT INTO runtime_integration_worktrees(path,repository,operation_id) VALUES (?,?,?)",
                        params![key, repository, operation_id],
                    )?;
                    Ok(())
                }
            })
            .await?;
        let result = (|| {
            self.checked_git(&self.repository, ["worktree", "add", &key, branch])?;
            ensure!(
                self.checked_git(&path, ["rev-parse", "HEAD"])? == expected,
                "target moved before publication"
            );
            self.checked_git(
                &path,
                [
                    "merge",
                    "--ff-only",
                    "--no-edit",
                    "--no-overwrite-ignore",
                    candidate,
                ],
            )?;
            Ok(())
        })();
        self.cleanup_target_worktree(key).await?;
        result
    }

    async fn cleanup_target_worktree(&self, path: String) -> Result<()> {
        let location = Path::new(&path);
        ensure!(
            location.parent() == Some(self.directory.as_path())
                && location
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("target-")),
            "invalid persisted integration worktree location"
        );
        if location.exists() {
            ensure!(
                !std::fs::symlink_metadata(location)?.is_symlink(),
                "integration worktree cannot be a symlink"
            );
            ensure!(
                self.checked_git(location, ["status", "--porcelain"])?
                    .is_empty(),
                "recovered integration worktree contains edits; inspect it before cleanup"
            );
            self.checked_git(&self.repository, ["worktree", "remove", &path])?;
        }
        self.store
            .write(move |connection| {
                connection.execute(
                    "DELETE FROM runtime_integration_worktrees WHERE path=?",
                    [&path],
                )?;
                Ok(())
            })
            .await
    }

    async fn recover_target_worktrees(&self) -> Result<()> {
        let repository = self.repository_key()?;
        let paths = self
            .store
            .read(move |connection| {
                let mut statement = connection
                    .prepare("SELECT path FROM runtime_integration_worktrees WHERE repository=? ORDER BY path")?;
                Ok(statement
                    .query_map([repository], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?)
            })
            .await?;
        for path in paths {
            self.cleanup_target_worktree(path).await?;
        }
        Ok(())
    }

    #[allow(
        clippy::disallowed_methods,
        reason = "Configured checks run on the owned integration worker and stdio is never converted to smol."
    )]
    async fn run_check(
        &self,
        staging: &Path,
        check: &IntegrationCheck,
        request: &IntegrationRequest,
        reverting: bool,
        index: usize,
    ) -> Result<()> {
        let output_path = self
            .directory
            .join(format!("check-{}-{index}.log", uuid::Uuid::new_v4()));
        let output = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&output_path)?;
        let mut command = Command::new(&check.program);
        command
            .args(&check.arguments)
            .current_dir(staging)
            .stdin(Stdio::null())
            .stdout(output.try_clone()?)
            .stderr(output.try_clone()?);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt as _;
            command.process_group(0);
        }
        let mut child = OwnedChild(
            command
                .spawn()
                .with_context(|| format!("start configured check {}", check.program))?,
        );
        let deadline = Instant::now() + Duration::from_secs(check.timeout_seconds);
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            let cancellation = if reverting {
                self.store
                    .snapshot(request.agent_id.clone())
                    .await
                    .map(|_| ())
            } else {
                self.check_generation(request).await
            };
            if Instant::now() >= deadline
                || self.stopping.load(Ordering::Acquire)
                || cancellation.is_err()
                || output.metadata()?.len() > MAXIMUM_OUTPUT * 16
            {
                terminate_check(&mut child)?;
                std::fs::remove_file(&output_path)?;
                cancellation?;
                bail!(
                    "configured check exceeded its time/output limit: {}",
                    check.program
                );
            }
            std::thread::sleep(Duration::from_millis(25));
        };
        let mut output = output;
        let length = output.metadata()?.len();
        output.seek(SeekFrom::Start(length.saturating_sub(MAXIMUM_OUTPUT)))?;
        let mut diagnostic = Vec::new();
        output.take(MAXIMUM_OUTPUT).read_to_end(&mut diagnostic)?;
        let diagnostic = String::from_utf8_lossy(&diagnostic);
        std::fs::remove_file(output_path)?;
        ensure!(
            status.success(),
            "configured check failed ({}): {diagnostic}",
            check.program
        );
        Ok(())
    }
}

fn validate_request(request: &IntegrationRequest) -> Result<()> {
    ensure!(
        !request.id.is_empty()
            && request.id.len() <= 256
            && !request.id.contains(['\n', '\r', '\0']),
        "invalid integration ID"
    );
    ensure!(
        request.source_commit.len() == 40
            && request
                .source_commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "source must be a full immutable Git commit ID"
    );
    ensure!(
        !request.target_branch.is_empty()
            && request.target_branch.len() <= 256
            && !request.target_branch.contains(['\n', '\r', '\0']),
        "invalid target branch"
    );
    validate_checks(&request.checks)
}

fn validate_checks(checks: &[IntegrationCheck]) -> Result<()> {
    ensure!(checks.len() <= 32, "too many configured checks");
    for check in checks {
        ensure!(
            !check.program.is_empty()
                && check.program.len() <= 4096
                && !check.program.contains('\0'),
            "invalid check executable"
        );
        ensure!(
            (1..=3600).contains(&check.timeout_seconds),
            "check timeout must be 1–3600 seconds"
        );
        ensure!(
            check.arguments.len() <= 256
                && check
                    .arguments
                    .iter()
                    .all(|argument| argument.len() <= 65536 && !argument.contains('\0')),
            "check arguments exceed limits"
        );
    }
    Ok(())
}

fn git<const COUNT: usize>(directory: &Path, arguments: [&str; COUNT]) -> Result<Output> {
    git_with_cancel(directory, arguments, None)
}

#[allow(
    clippy::disallowed_methods,
    reason = "Git runs on an owned OS worker thread and these stdio settings are never converted to smol."
)]
fn git_with_cancel<const COUNT: usize>(
    directory: &Path,
    arguments: [&str; COUNT],
    stopping: Option<&AtomicBool>,
) -> Result<Output> {
    ensure!(
        !stopping.is_some_and(|stopping| stopping.load(Ordering::Acquire)),
        "integration owner is shutting down"
    );
    let scratch = std::env::temp_dir().join(format!("zezu-integration-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&scratch)?;
    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            if let Err(error) = std::fs::remove_dir_all(&self.0) {
                log::error!("Git output cleanup failed: {error}");
            }
        }
    }
    let _scratch = Scratch(scratch.clone());
    let mut stdout = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(scratch.join("stdout"))?;
    let mut stderr = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(scratch.join("stderr"))?;
    let mut command = Command::new("git");
    command
        .arg("-c")
        .arg("core.hooksPath=/dev/null")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .env_remove("GIT_COMMON_DIR")
        .env_remove("GIT_OBJECT_DIRECTORY")
        .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES")
        .env("GIT_TERMINAL_PROMPT", "0")
        .args(arguments)
        .current_dir(directory)
        .stdin(Stdio::null())
        .stdout(stdout.try_clone()?)
        .stderr(stderr.try_clone()?);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }
    let mut child = OwnedChild(command.spawn().context("start Git integration command")?);
    let deadline = Instant::now() + Duration::from_secs(300);
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if stopping.is_some_and(|stopping| stopping.load(Ordering::Acquire))
            || Instant::now() >= deadline
            || stdout.metadata()?.len() > MAXIMUM_OUTPUT * 16
            || stderr.metadata()?.len() > MAXIMUM_OUTPUT * 16
        {
            terminate_check(&mut child)?;
            bail!("Git integration command was cancelled or exceeded its time/output limit");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    ensure!(
        stdout.metadata()?.len() <= MAXIMUM_OUTPUT * 16
            && stderr.metadata()?.len() <= MAXIMUM_OUTPUT * 16,
        "Git output exceeds the runtime byte limit"
    );
    stdout.seek(SeekFrom::Start(0))?;
    stderr.seek(SeekFrom::Start(0))?;
    let mut stdout_bytes = Vec::new();
    let mut stderr_bytes = Vec::new();
    stdout.read_to_end(&mut stdout_bytes)?;
    stderr.read_to_end(&mut stderr_bytes)?;
    Ok(Output {
        status,
        stdout: stdout_bytes,
        stderr: stderr_bytes,
    })
}

fn checked_git<const COUNT: usize>(directory: &Path, arguments: [&str; COUNT]) -> Result<String> {
    let output = git(directory, arguments)?;
    ensure!(
        output.status.success(),
        "Git integration failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim_end().to_owned())
}

#[allow(
    clippy::disallowed_methods,
    reason = "Git runs on an owned OS worker thread and these stdio settings are never converted to smol."
)]
fn terminate_check(child: &mut std::process::Child) -> Result<()> {
    #[cfg(unix)]
    {
        let result = Command::new("/bin/kill")
            .args(["-KILL", "--", &format!("-{}", child.id())])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;
        if !result.success() && child.try_wait()?.is_none() {
            child.kill()?;
        }
    }
    #[cfg(windows)]
    {
        let result = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &child.id().to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;
        if !result.success() && child.try_wait()?.is_none() {
            child.kill()?;
        }
    }
    #[cfg(not(any(unix, windows)))]
    if child.try_wait()?.is_none() {
        child.kill()?;
    }
    child.wait()?;
    Ok(())
}

async fn worker<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let (sender, receiver) = futures::channel::oneshot::channel();
    std::thread::Builder::new()
        .name("agent-integration".into())
        .spawn(move || {
            if let Err(Err(error)) = sender.send(operation()) {
                log::error!("detached integration failed: {error:#}");
            }
        })?;
    receiver.await.context("integration worker stopped")?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeLimits;

    struct Fixture {
        directory: tempfile::TempDir,
        store: RuntimeStore,
        integration: GitIntegration,
        source: String,
    }

    impl Fixture {
        async fn new() -> Result<Self> {
            let directory = tempfile::tempdir()?;
            let repository = directory.path().join("repository");
            std::fs::create_dir(&repository)?;
            checked_git(&repository, ["init", "--initial-branch=main"])?;
            checked_git(&repository, ["config", "user.name", "Runtime Test"])?;
            checked_git(
                &repository,
                ["config", "user.email", "test@example.invalid"],
            )?;
            std::fs::write(repository.join("agent.txt"), "before\n")?;
            checked_git(&repository, ["add", "agent.txt"])?;
            checked_git(&repository, ["commit", "-m", "initial"])?;
            checked_git(&repository, ["switch", "-c", "agent"])?;
            std::fs::write(repository.join("agent.txt"), "after\n")?;
            checked_git(&repository, ["commit", "-am", "agent change"])?;
            let source = checked_git(&repository, ["rev-parse", "HEAD"])?;
            checked_git(&repository, ["switch", "main"])?;
            let store = RuntimeStore::open(
                directory.path().join("runtime.sqlite"),
                RuntimeLimits::default(),
            )?;
            store
                .create_root("agent".into(), serde_json::json!({}))
                .await?;
            let integration = GitIntegration::open(store.clone(), repository).await?;
            Ok(Self {
                directory,
                store,
                integration,
                source,
            })
        }

        fn request(&self) -> IntegrationRequest {
            IntegrationRequest {
                id: "operation".into(),
                agent_id: "agent".into(),
                generation: 0,
                source_owner: None,
                target_branch: "main".into(),
                source_commit: self.source.clone(),
                checks: Vec::new(),
            }
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn integration_owner_unlocks_while_an_inherited_descriptor_remains_open() -> Result<()> {
        let fixture = Fixture::new().await?;
        let owner = fixture.integration.owner()?;
        let inherited = owner.0.try_clone()?;
        assert!(fixture.integration.owner().is_err());
        drop(owner);
        let next_owner = fixture.integration.owner()?;
        assert!(fixture.integration.owner().is_err());
        drop(inherited);
        assert!(fixture.integration.owner().is_err());
        drop(next_owner);
        drop(fixture.integration.owner()?);
        fixture.store.shutdown().await
    }

    #[tokio::test]
    async fn integration_is_idempotent_and_selective_revert_preserves_later_changes() -> Result<()>
    {
        let fixture = Fixture::new().await?;
        let request = fixture.request();
        let record = fixture.integration.integrate(request.clone()).await?;
        assert_eq!(record.state, "integrated");
        let repeated = fixture.integration.integrate(request).await?;
        assert_eq!(record, repeated);
        let repository = &fixture.integration.repository;
        assert_eq!(
            std::fs::read_to_string(repository.join("agent.txt"))?,
            "after\n"
        );
        std::fs::write(repository.join("unrelated.txt"), "user change\n")?;
        checked_git(repository, ["add", "unrelated.txt"])?;
        checked_git(repository, ["commit", "-m", "later user change"])?;
        let reverted = fixture
            .integration
            .revert(record.request.id, Vec::new())
            .await?;
        assert_eq!(reverted.state, "reverted");
        assert_eq!(
            std::fs::read_to_string(repository.join("agent.txt"))?,
            "before\n"
        );
        assert_eq!(
            std::fs::read_to_string(repository.join("unrelated.txt"))?,
            "user change\n"
        );
        fixture.store.shutdown().await
    }

    #[tokio::test]
    async fn conflicts_and_dirty_destinations_never_overwrite_user_changes() -> Result<()> {
        let fixture = Fixture::new().await?;
        let repository = &fixture.integration.repository;
        std::fs::write(repository.join("agent.txt"), "unsaved user work\n")?;
        let before = fixture.integration.branch_head("main")?;
        assert!(
            fixture
                .integration
                .integrate(fixture.request())
                .await
                .is_err()
        );
        assert_eq!(fixture.integration.branch_head("main")?, before);
        assert_eq!(
            std::fs::read_to_string(repository.join("agent.txt"))?,
            "unsaved user work\n"
        );
        checked_git(repository, ["commit", "-am", "conflicting user change"])?;
        let mut request = fixture.request();
        request.id = "conflicting".into();
        let before = fixture.integration.branch_head("main")?;
        assert!(fixture.integration.integrate(request).await.is_err());
        assert_eq!(fixture.integration.branch_head("main")?, before);
        assert_eq!(
            std::fs::read_to_string(repository.join("agent.txt"))?,
            "unsaved user work\n"
        );
        assert!(checked_git(repository, ["status", "--porcelain"])?.is_empty());
        fixture.store.shutdown().await
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn failed_checks_leave_the_destination_unchanged_and_report_diagnostics() -> Result<()> {
        let fixture = Fixture::new().await?;
        let mut request = fixture.request();
        request.checks.push(IntegrationCheck {
            program: "/bin/sh".into(),
            arguments: vec!["-c".into(), "printf 'check failed'; exit 7".into()],
            timeout_seconds: 5,
        });
        let before = fixture.integration.branch_head("main")?;
        let error = fixture
            .integration
            .integrate(request.clone())
            .await
            .err()
            .context("failed check")?;
        assert!(error.to_string().contains("check failed"));
        assert_eq!(fixture.integration.branch_head("main")?, before);
        assert_eq!(
            fixture
                .integration
                .record(request.id)
                .await?
                .context("record")?
                .state,
            "failed"
        );
        fixture.store.shutdown().await
    }

    #[tokio::test]
    async fn interrupted_publication_is_reconciled_without_reapplying_changes() -> Result<()> {
        let fixture = Fixture::new().await?;
        let mut record = fixture.integration.integrate(fixture.request()).await?;
        record.state = "publishing".into();
        fixture.integration.save(record, false).await?;
        let before = fixture.integration.branch_head("main")?;
        let recovered = fixture.integration.recover().await?;
        assert_eq!(
            recovered.first().context("recovered record")?.state,
            "integrated"
        );
        assert_eq!(fixture.integration.branch_head("main")?, before);
        assert!(fixture.directory.path().exists());
        fixture.store.shutdown().await
    }

    #[tokio::test]
    async fn a_failed_revert_preserves_the_integration_and_can_be_retried() -> Result<()> {
        let fixture = Fixture::new().await?;
        let record = fixture.integration.integrate(fixture.request()).await?;
        let repository = &fixture.integration.repository;
        std::fs::write(
            repository.join("agent.txt"),
            "later same-line user change\n",
        )?;
        checked_git(repository, ["commit", "-am", "later conflicting change"])?;
        let later = checked_git(repository, ["rev-parse", "HEAD"])?;
        assert!(
            fixture
                .integration
                .revert(record.request.id.clone(), Vec::new())
                .await
                .is_err()
        );
        assert_eq!(
            fixture
                .integration
                .record(record.request.id.clone())
                .await?
                .context("integration")?
                .state,
            "integrated"
        );
        assert_eq!(
            std::fs::read_to_string(repository.join("agent.txt"))?,
            "later same-line user change\n"
        );
        checked_git(repository, ["revert", "--no-edit", &later])?;
        assert_eq!(
            fixture
                .integration
                .revert(record.request.id, Vec::new())
                .await?
                .state,
            "reverted"
        );
        fixture.store.shutdown().await
    }

    #[tokio::test]
    async fn publication_does_not_overwrite_ignored_user_files() -> Result<()> {
        let fixture = Fixture::new().await?;
        let repository = &fixture.integration.repository;
        checked_git(repository, ["switch", "agent"])?;
        std::fs::write(repository.join("ignored.log"), "agent artifact\n")?;
        checked_git(repository, ["add", "ignored.log"])?;
        checked_git(repository, ["commit", "-m", "add generated artifact"])?;
        let source = checked_git(repository, ["rev-parse", "HEAD"])?;
        checked_git(repository, ["switch", "main"])?;
        std::fs::write(repository.join(".gitignore"), "ignored.log\n")?;
        checked_git(repository, ["add", ".gitignore"])?;
        checked_git(repository, ["commit", "-m", "ignore local log"])?;
        std::fs::write(repository.join("ignored.log"), "local user log\n")?;
        let mut request = fixture.request();
        request.source_commit = source;
        assert!(fixture.integration.integrate(request).await.is_err());
        assert_eq!(
            std::fs::read_to_string(repository.join("ignored.log"))?,
            "local user log\n"
        );
        fixture.store.shutdown().await
    }

    #[tokio::test]
    async fn integration_of_an_unchecked_out_branch_uses_an_owned_target_worktree() -> Result<()> {
        let fixture = Fixture::new().await?;
        let repository = &fixture.integration.repository;
        checked_git(repository, ["switch", "-c", "user-branch"])?;
        let user_head = checked_git(repository, ["rev-parse", "HEAD"])?;
        let record = fixture.integration.integrate(fixture.request()).await?;
        assert_eq!(checked_git(repository, ["rev-parse", "HEAD"])?, user_head);
        assert_eq!(
            std::fs::read_to_string(repository.join("agent.txt"))?,
            "before\n"
        );
        assert_eq!(
            fixture.integration.branch_head("main")?,
            record
                .integrated_commit
                .clone()
                .context("published commit")?
        );
        assert!(fixture.integration.recover().await?.is_empty());
        let published = record.integrated_commit.context("published commit")?;
        checked_git(
            repository,
            [
                "show-ref",
                "--verify",
                &format!("refs/zezu/integrations/{published}"),
            ],
        )?;
        let worktrees = checked_git(repository, ["worktree", "list", "--porcelain"])?;
        assert!(!worktrees.contains("target-"));
        fixture.store.shutdown().await
    }

    #[tokio::test]
    async fn integrations_are_scoped_to_their_source_repository() -> Result<()> {
        let fixture = Fixture::new().await?;
        let record = fixture.integration.integrate(fixture.request()).await?;
        let other = fixture.directory.path().join("other");
        checked_git(
            fixture.directory.path(),
            [
                "clone",
                "--no-hardlinks",
                fixture
                    .integration
                    .repository
                    .to_str()
                    .context("repository path")?,
                other.to_str().context("other repository path")?,
            ],
        )?;
        let other = GitIntegration::open(fixture.store.clone(), other).await?;
        assert!(other.record(record.request.id.clone()).await?.is_none());
        assert!(other.revert(record.request.id, Vec::new()).await.is_err());
        assert!(other.recover().await?.is_empty());
        fixture.store.shutdown().await
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancellation_of_the_source_owner_stops_checks_before_publication() -> Result<()> {
        let fixture = Fixture::new().await?;
        let child = fixture
            .store
            .spawn_child("agent".into(), "source".into(), 0, true)
            .await?;
        let marker = fixture.directory.path().join("check-started");
        let mut request = fixture.request();
        request.source_owner = Some(TaskIdentity {
            agent_id: child.clone(),
            generation: 0,
        });
        request.checks = vec![IntegrationCheck {
            program: "/bin/sh".into(),
            arguments: vec![
                "-c".into(),
                "touch \"$1\"; sleep 10".into(),
                "runtime-check".into(),
                marker.to_str().context("marker path")?.into(),
            ],
            timeout_seconds: 30,
        }];
        let before = fixture.integration.branch_head("main")?;
        let cancel = async {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !marker.exists() {
                ensure!(Instant::now() < deadline, "check did not start");
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            fixture.store.cancel(child).await
        };
        let (integration, cancellation) =
            futures::join!(fixture.integration.integrate(request), cancel);
        cancellation?;
        assert!(integration.is_err());
        assert_eq!(fixture.integration.branch_head("main")?, before);
        assert_eq!(
            fixture.store.snapshot("agent".into()).await?.state,
            "waiting"
        );
        fixture.store.shutdown().await
    }
}
