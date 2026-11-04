use std::{
    collections::HashMap,
    io::Read as _,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, anyhow, ensure};
use rusqlite::{Connection, OptionalExtension as _, params};
use serde::{Deserialize, Serialize};

use crate::{RuntimeStore, orchestration::TaskIdentity};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceLease {
    pub slot_id: i64,
    pub agent_id: String,
    pub generation: i64,
    pub token: String,
    pub path: PathBuf,
    pub base_commit: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceArtifact {
    pub sequence: i64,
    pub id: String,
    pub agent_id: String,
    pub generation: i64,
    pub commit: String,
    pub base_commit: String,
}

type PermitMap = Mutex<HashMap<String, crate::resource::ResourcePermit>>;
static SHARED_PERMITS: OnceLock<Mutex<HashMap<PathBuf, Weak<PermitMap>>>> = OnceLock::new();

fn shared_permits(directory: &Path) -> Result<Arc<PermitMap>> {
    let mut pools = SHARED_PERMITS
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|_| anyhow!("workspace permit registry poisoned"))?;
    pools.retain(|_, pool| pool.strong_count() > 0);
    if let Some(pool) = pools.get(directory).and_then(Weak::upgrade) {
        return Ok(pool);
    }
    let permits = Arc::new(Mutex::new(HashMap::new()));
    pools.insert(directory.to_path_buf(), Arc::downgrade(&permits));
    Ok(permits)
}

#[derive(Clone)]
pub struct WorkspacePool {
    store: RuntimeStore,
    repository: PathBuf,
    directory: PathBuf,
    capacity: usize,
    stopping: Arc<AtomicBool>,
    resources: Option<crate::resource::RuntimeResources>,
    permits: Arc<PermitMap>,
}

impl WorkspacePool {
    pub async fn open(
        store: RuntimeStore,
        repository: PathBuf,
        directory: PathBuf,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&capacity),
            "workspace capacity must be between 1 and 4096"
        );
        let (repository, directory) = blocking(move |cancelled| {
            let repository = repository.canonicalize()?;
            let actual = git(&repository, &["rev-parse", "--show-toplevel"], &cancelled)?;
            ensure!(
                Path::new(actual.trim()).canonicalize()? == repository,
                "workspace repository must be its top-level directory"
            );
            std::fs::create_dir_all(&directory)?;
            let directory = directory.canonicalize()?;
            ensure!(
                !directory.starts_with(&repository),
                "workspace pool must be outside the source worktree"
            );
            Ok((repository, directory))
        })
        .await?;
        let permits = shared_permits(&directory)?;
        let pool = Self {
            store,
            repository,
            directory,
            capacity,
            stopping: Arc::new(AtomicBool::new(false)),
            resources: None,
            permits,
        };
        let repository_key = pool.repository_key()?;
        let directory_key = pool
            .directory
            .to_str()
            .context("workspace directory must be UTF-8")?
            .to_owned();
        pool.store.write(move |connection| {
            Self::initialize_connection(connection)?;
            let transaction = connection.transaction()?;
            let existing: Option<(String,usize)> = transaction.query_row("SELECT directory,capacity FROM runtime_workspace_pools WHERE repository=?", [&repository_key], |row| Ok((row.get(0)?,row.get(1)?))).optional()?;
            if let Some((existing_directory,_)) = existing {
                ensure!(existing_directory == directory_key, "workspace directory differs from the persisted owner");
                transaction.execute("UPDATE runtime_workspace_pools SET capacity=? WHERE repository=?",params![capacity,repository_key])?;
            } else {
                transaction.execute("INSERT INTO runtime_workspace_pools(repository,directory,capacity) VALUES (?,?,?)", params![repository_key,directory_key,capacity])?;
            }
            for slot in 0..capacity {
                let path = Path::new(&directory_key).join(format!("slot-{slot:04}")).join(Path::new(&repository_key).file_name().context("repository root name")?);
                transaction.execute("INSERT OR IGNORE INTO runtime_workspaces(repository,slot,path,state) VALUES (?,?,?,'available')", params![repository_key,slot,path.to_str().context("workspace path")?])?;
            }
            transaction.commit()?;
            Ok(())
        }).await?;
        Ok(pool)
    }

    pub fn request_shutdown(&self) {
        self.stopping.store(true, Ordering::Release);
    }

    async fn run_operation<T: Send + 'static>(
        &self,
        operation: impl FnOnce(Cancellation) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        ensure!(
            !self.stopping.load(Ordering::Acquire),
            "workspace pool is shutting down"
        );
        blocking_with_stop(operation, self.stopping.clone()).await
    }

    pub fn with_resources(mut self, resources: crate::resource::RuntimeResources) -> Self {
        self.resources = Some(resources);
        self
    }

    pub(crate) fn initialize_connection(connection: &Connection) -> Result<()> {
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS runtime_workspace_pools(repository TEXT PRIMARY KEY,directory TEXT NOT NULL,capacity INTEGER NOT NULL CHECK(capacity>0));
             CREATE TABLE IF NOT EXISTS runtime_workspaces(
                repository TEXT NOT NULL REFERENCES runtime_workspace_pools(repository),slot INTEGER NOT NULL,
                path TEXT NOT NULL UNIQUE,state TEXT NOT NULL CHECK(state IN ('available','creating','leased','busy','uncertain')),
                agent_id TEXT REFERENCES agents(id),generation INTEGER,token TEXT,base_commit TEXT,
                PRIMARY KEY(repository,slot));
             CREATE INDEX IF NOT EXISTS runtime_workspace_available ON runtime_workspaces(repository,state,slot);
             CREATE TABLE IF NOT EXISTS runtime_workspace_artifacts(
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT NOT NULL UNIQUE,agent_id TEXT NOT NULL REFERENCES agents(id),
                generation INTEGER NOT NULL,commit_id TEXT NOT NULL,base_commit TEXT NOT NULL,repository TEXT NOT NULL,
                slot INTEGER NOT NULL,token TEXT NOT NULL);
             CREATE INDEX IF NOT EXISTS runtime_artifacts_agent ON runtime_workspace_artifacts(agent_id,sequence);
             CREATE TABLE IF NOT EXISTS runtime_resource_waits(job_id TEXT PRIMARY KEY REFERENCES jobs(id),kind TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS runtime_resource_epochs(kind TEXT PRIMARY KEY,epoch INTEGER NOT NULL);
             INSERT OR IGNORE INTO runtime_resource_epochs(kind,epoch) VALUES ('workspace',0);"
        )?;
        Ok(())
    }

    pub async fn reclaim_finished(&self) -> Result<usize> {
        let repository_key = self.repository_key()?;
        let candidates:Vec<WorkspaceLease>=self.store.read(move |connection| {
            let mut statement=connection.prepare("SELECT w.slot,w.agent_id,w.generation,w.token,w.path,w.base_commit FROM runtime_workspaces w JOIN agents a ON a.id=w.agent_id WHERE w.repository=? AND w.state IN ('leased','uncertain') AND a.state IN ('completed','failed','cancelled') ORDER BY w.slot")?;
            Ok(statement.query_map([repository_key],lease_row)?.collect::<rusqlite::Result<_>>()?)
        }).await?;
        let mut released = 0;
        for lease in candidates {
            self.validate_path(&lease)?;
            if !lease.path.exists() {
                continue;
            }
            let path = lease.path.clone();
            let head = self
                .run_operation(move |cancelled| {
                    if !git(
                        &path,
                        &["status", "--porcelain=v1", "--untracked-files=all"],
                        &cancelled,
                    )?
                    .is_empty()
                    {
                        return Ok(None);
                    }
                    Ok(Some(
                        git(&path, &["rev-parse", "HEAD"], &cancelled)?
                            .trim()
                            .to_owned(),
                    ))
                })
                .await?;
            let Some(head) = head else {
                continue;
            };
            let repository_key = self.repository_key()?;
            let token = lease.token.clone();
            let changed=self.store.write(move |connection| {
                let transaction=connection.transaction()?;
                let preserved=head==lease.base_commit || transaction.query_row("SELECT EXISTS(SELECT 1 FROM runtime_workspace_artifacts WHERE repository=? AND slot=? AND token=? AND commit_id=?)",params![repository_key,lease.slot_id,lease.token,head],|row|row.get::<_,bool>(0))?;
                if !preserved {return Ok(false);}
                let changed=transaction.execute("UPDATE runtime_workspaces SET state='available',agent_id=NULL,generation=NULL,token=NULL,base_commit=NULL WHERE repository=? AND slot=? AND token=? AND state IN ('leased','uncertain') AND EXISTS(SELECT 1 FROM agents a WHERE a.id=runtime_workspaces.agent_id AND a.state IN ('completed','failed','cancelled'))",params![repository_key,lease.slot_id,lease.token])?==1;
                transaction.commit()?;
                Ok(changed)
            }).await?;
            if changed {
                self.permits
                    .lock()
                    .map_err(|_| anyhow!("workspace permits poisoned"))?
                    .remove(&token);
                self.store.notify_workspace_available().await?;
                released += 1;
            }
        }
        Ok(released)
    }

    pub async fn acquire(
        &self,
        task: TaskIdentity,
        base_ref: String,
    ) -> Result<Option<WorkspaceLease>> {
        self.recover().await?;
        let recoverable = {
            let repository_key = self.repository_key()?;
            let owner = task.clone();
            self.store.read(move |connection| {
                let resumed:bool=connection.query_row("SELECT EXISTS(SELECT 1 FROM agents WHERE id=? AND generation=? AND pause_requested=0 AND state IN ('queued','running','waiting'))",params![owner.agent_id,owner.generation],|row|row.get(0))?;
                ensure!(resumed,"resume the task before acquiring its editing workspace");
                connection.query_row("SELECT slot,agent_id,generation,token,path,base_commit FROM runtime_workspaces WHERE repository=? AND agent_id=? AND generation<? AND state IN ('uncertain','leased','busy') LIMIT 1",params![repository_key,owner.agent_id,owner.generation],lease_row).optional().map_err(Into::into)
            }).await?
        };
        if let Some(old) = recoverable {
            let old = old.clone();
            let repository_key = self.repository_key()?;
            let slot = old.slot_id;
            let token = old.token.clone();
            self.store.write(move |connection| {connection.execute("UPDATE runtime_workspaces SET state='uncertain' WHERE repository=? AND slot=? AND token=?",params![repository_key,slot,token])?;Ok(())}).await?;
            return Ok(Some(self.resume_lease(old, task).await?));
        }
        if let Some(lease) = self.store.workspace_lease_for_task(task.clone()).await? {
            self.validate_path(&lease)?;
            return Ok(Some(lease));
        }
        self.reclaim_finished().await?;
        let permit = match &self.resources {
            Some(resources) => {
                match resources.try_reserve(crate::resource::ResourceKind::Workspace, 0)? {
                    Some(permit) => Some(permit),
                    None => return Ok(None),
                }
            }
            None => None,
        };
        let repository = self.repository.clone();
        let base_commit = self
            .run_operation(move |cancelled| {
                git(
                    &repository,
                    &[
                        "rev-parse",
                        "--verify",
                        "--end-of-options",
                        &format!("{base_ref}^{{commit}}"),
                    ],
                    &cancelled,
                )
                .map(|value| value.trim().to_owned())
            })
            .await?;
        let repository_key = self.repository_key()?;
        let capacity = self.capacity;
        let lease = self.store.write(move |connection| {
            let transaction = connection.transaction()?;
            validate_task(&transaction,&task)?;
            let available: Option<(i64,String)> = transaction.query_row("SELECT slot,path FROM runtime_workspaces WHERE repository=? AND slot<? AND state='available' ORDER BY slot LIMIT 1", params![repository_key,capacity], |row| Ok((row.get(0)?,row.get(1)?))).optional()?;
            let Some((slot_id,path)) = available else { return Ok(None); };
            let token = uuid::Uuid::new_v4().to_string();
            transaction.execute("UPDATE runtime_workspaces SET state='creating',agent_id=?,generation=?,token=?,base_commit=? WHERE repository=? AND slot=? AND state='available'", params![task.agent_id,task.generation,token,base_commit,repository_key,slot_id])?;
            transaction.commit()?;
            Ok(Some(WorkspaceLease {slot_id,agent_id:task.agent_id,generation:task.generation,token,path:PathBuf::from(path),base_commit}))
        }).await?;
        let Some(lease) = lease else {
            return Ok(None);
        };
        self.validate_path(&lease)?;
        let repository = self.repository.clone();
        let work = lease.clone();
        let result = self
            .run_operation(move |cancelled| {
                if work.path.exists() {
                    ensure!(
                        git(
                            &work.path,
                            &["status", "--porcelain=v1", "--untracked-files=all"],
                            &cancelled
                        )?
                        .is_empty(),
                        "workspace has uncheckpointed edits; recovery is required"
                    );
                    ensure!(
                        git(&work.path, &["rev-parse", "--show-toplevel"], &cancelled)?.trim()
                            == work.path.to_str().context("worktree path")?,
                        "workspace path is not its own worktree"
                    );
                    git(
                        &work.path,
                        &["reset", "--hard", &work.base_commit],
                        &cancelled,
                    )?;
                } else {
                    std::fs::create_dir_all(work.path.parent().context("workspace parent")?)?;
                    git(
                        &repository,
                        &[
                            "worktree",
                            "add",
                            "--detach",
                            work.path.to_str().context("worktree path")?,
                            &work.base_commit,
                        ],
                        &cancelled,
                    )?;
                }
                Ok(())
            })
            .await;
        let repository_key = self.repository_key()?;
        let current = lease.clone();
        self.store.write(move |connection| {
            let transaction = connection.transaction()?;
            let valid = validate_task(&transaction,&TaskIdentity {agent_id:current.agent_id.clone(),generation:current.generation}).is_ok();
            let state = if result.is_ok() && valid {"leased"} else {"uncertain"};
            transaction.execute("UPDATE runtime_workspaces SET state=? WHERE repository=? AND slot=? AND token=?",params![state,repository_key,current.slot_id,current.token])?;
            transaction.commit()?;
            result?;
            ensure!(valid,"workspace owner changed during acquisition");
            Ok(())
        }).await?;
        if let Some(permit) = permit {
            self.permits
                .lock()
                .map_err(|_| anyhow!("workspace permits poisoned"))?
                .insert(lease.token.clone(), permit);
        }
        Ok(Some(lease))
    }

    pub async fn checkpoint(
        &self,
        lease: WorkspaceLease,
        message: String,
    ) -> Result<WorkspaceArtifact> {
        ensure!(
            !message.is_empty() && message.len() <= 8192,
            "checkpoint message must contain 1 to 8192 bytes"
        );
        self.reserve_operation(&lease).await?;
        let path = lease.path.clone();
        let hooks_directory = self.directory.join("disabled-hooks");
        let artifact_id = uuid::Uuid::new_v4().to_string();
        let reference = format!("refs/zezu/artifacts/{artifact_id}");
        let commit = self
            .run_operation(move |cancelled| {
                std::fs::create_dir_all(&hooks_directory)?;
                ensure!(
                    !std::fs::symlink_metadata(&hooks_directory)?
                        .file_type()
                        .is_symlink(),
                    "hooks directory cannot be a symlink"
                );
                ensure!(
                    std::fs::read_dir(&hooks_directory)?.next().is_none(),
                    "runtime checkpoint hooks directory must be empty"
                );
                git(&path, &["add", "--all"], &cancelled)?;
                git(
                    &path,
                    &[
                        "-c",
                        &format!(
                            "core.hooksPath={}",
                            hooks_directory.to_str().context("hooks directory")?
                        ),
                        "-c",
                        "commit.gpgsign=false",
                        "commit",
                        "--allow-empty",
                        "-m",
                        &message,
                    ],
                    &cancelled,
                )?;
                let commit = git(&path, &["rev-parse", "HEAD"], &cancelled)?
                    .trim()
                    .to_owned();
                // An artifact must remain reachable even after its pooled worktree is reused.
                git(&path, &["update-ref", &reference, &commit], &cancelled)?;
                Ok(commit)
            })
            .await;
        let commit = match commit {
            Ok(commit) => commit,
            Err(error) => {
                self.end_operation(&lease, "uncertain").await?;
                return Err(error);
            }
        };
        let repository_key = self.repository_key()?;
        let mut artifact = WorkspaceArtifact {
            sequence: 0,
            id: artifact_id,
            agent_id: lease.agent_id.clone(),
            generation: lease.generation,
            commit,
            base_commit: lease.base_commit.clone(),
        };
        let saved = artifact.clone();
        let sequence = self.store.write(move |connection| {
            let transaction = connection.transaction()?;
            check_operation(&transaction,&repository_key,&lease)?;
            transaction.execute("INSERT INTO runtime_workspace_artifacts(id,agent_id,generation,commit_id,base_commit,repository,slot,token) VALUES (?,?,?,?,?,?,?,?)",params![saved.id,saved.agent_id,saved.generation,saved.commit,saved.base_commit,repository_key,lease.slot_id,lease.token])?;
            transaction.execute("UPDATE runtime_workspaces SET state='leased' WHERE repository=? AND slot=? AND token=?", params![repository_key,lease.slot_id,lease.token])?;
            let sequence = transaction.last_insert_rowid();
            transaction.commit()?;
            Ok(sequence)
        }).await?;
        artifact.sequence = sequence;
        Ok(artifact)
    }

    pub async fn release(&self, lease: WorkspaceLease) -> Result<()> {
        self.reserve_operation(&lease).await?;
        let path = lease.path.clone();
        let result = self
            .run_operation(move |cancelled| {
                ensure!(
                    git(
                        &path,
                        &["status", "--porcelain=v1", "--untracked-files=all"],
                        &cancelled
                    )?
                    .is_empty(),
                    "checkpoint edits before releasing a workspace"
                );
                Ok(())
            })
            .await;
        if let Err(error) = result {
            self.end_operation(&lease, "leased").await?;
            return Err(error);
        }
        let repository_key = self.repository_key()?;
        let token = lease.token.clone();
        self.store.write(move |connection| {
            let transaction = connection.transaction()?;
            check_operation(&transaction,&repository_key,&lease)?;
            transaction.execute("UPDATE runtime_workspaces SET state='available',agent_id=NULL,generation=NULL,token=NULL,base_commit=NULL WHERE repository=? AND slot=? AND token=?",params![repository_key,lease.slot_id,lease.token])?;
            transaction.commit()?;
            Ok(())
        }).await?;
        self.permits
            .lock()
            .map_err(|_| anyhow!("workspace permits poisoned"))?
            .remove(&token);
        self.store.notify_workspace_available().await?;
        Ok(())
    }

    pub async fn recover(&self) -> Result<Vec<WorkspaceLease>> {
        let repository_key = self.repository_key()?;
        let leases: Vec<WorkspaceLease> = self.store.write(move |connection| {
            connection.execute("UPDATE runtime_workspaces SET state='uncertain' WHERE repository=? AND state IN ('creating','leased','busy') AND (NOT EXISTS(SELECT 1 FROM agents a WHERE a.id=runtime_workspaces.agent_id AND a.generation=runtime_workspaces.generation AND a.state NOT IN ('cancelled','completed','failed')))",[&repository_key])?;
            let mut statement = connection.prepare("SELECT slot,agent_id,generation,token,path,base_commit FROM runtime_workspaces WHERE repository=? AND state='uncertain' ORDER BY slot")?;
            Ok(statement.query_map([repository_key],lease_row)?.collect::<rusqlite::Result<_>>()?)
        }).await?;
        {
            let mut permits = self
                .permits
                .lock()
                .map_err(|_| anyhow!("workspace permits poisoned"))?;
            for lease in &leases {
                permits.remove(&lease.token);
            }
        }
        if !leases.is_empty() {
            self.store.notify_workspace_available().await?;
        }
        Ok(leases)
    }

    pub async fn resume_lease(
        &self,
        old: WorkspaceLease,
        owner: TaskIdentity,
    ) -> Result<WorkspaceLease> {
        ensure!(
            old.agent_id == owner.agent_id,
            "a recovered workspace cannot be transferred to another task"
        );
        self.validate_path(&old)?;
        let path = old.path.clone();
        self.run_operation(move |cancelled| {
            ensure!(
                path.exists(),
                "worktree creation was interrupted; inspect and reconcile it before resuming"
            );
            git(&path, &["rev-parse", "--is-inside-work-tree"], &cancelled)?;
            Ok(())
        })
        .await?;
        let permit = self
            .resources
            .as_ref()
            .map(|resources| resources.try_acquire(crate::resource::ResourceKind::Workspace, 0))
            .transpose()?;
        let repository_key = self.repository_key()?;
        let resumed = self.store.write(move |connection| {
            let transaction = connection.transaction()?;
            validate_task(&transaction,&owner)?;
            let resumed:bool=transaction.query_row("SELECT pause_requested=0 AND state!='paused' FROM agents WHERE id=?",[&owner.agent_id],|row|row.get(0))?;
            ensure!(resumed,"resume the task before restoring its editing lease");
            let token = uuid::Uuid::new_v4().to_string();
            let changed = transaction.execute("UPDATE runtime_workspaces SET generation=?,token=?,state='leased' WHERE repository=? AND slot=? AND token=? AND state='uncertain'",params![owner.generation,token,repository_key,old.slot_id,old.token])?;
            ensure!(changed == 1,"recovered workspace lease changed");
            transaction.commit()?;
            Ok(WorkspaceLease {generation:owner.generation,token,..old})
        }).await?;
        if let Some(permit) = permit {
            self.permits
                .lock()
                .map_err(|_| anyhow!("workspace permits poisoned"))?
                .insert(resumed.token.clone(), permit);
        }
        Ok(resumed)
    }

    pub async fn artifacts(
        &self,
        agent_id: String,
        after: i64,
        limit: usize,
    ) -> Result<Vec<WorkspaceArtifact>> {
        ensure!(
            after >= 0 && (1..=256).contains(&limit),
            "invalid artifact page"
        );
        let repository_key = self.repository_key()?;
        self.store.read(move |connection| {
            let mut statement = connection.prepare("SELECT sequence,id,agent_id,generation,commit_id,base_commit FROM runtime_workspace_artifacts WHERE repository=? AND agent_id=? AND sequence>? ORDER BY sequence LIMIT ?")?;
            Ok(statement.query_map(params![repository_key,agent_id,after,limit],|row| Ok(WorkspaceArtifact {sequence:row.get(0)?,id:row.get(1)?,agent_id:row.get(2)?,generation:row.get(3)?,commit:row.get(4)?,base_commit:row.get(5)?}))?.collect::<rusqlite::Result<_>>()?)
        }).await
    }

    pub async fn artifact(&self, id: String) -> Result<Option<WorkspaceArtifact>> {
        let repository_key = self.repository_key()?;
        self.store.read(move |connection| connection.query_row("SELECT sequence,id,agent_id,generation,commit_id,base_commit FROM runtime_workspace_artifacts WHERE repository=? AND id=?",params![repository_key,id],|row| Ok(WorkspaceArtifact {sequence:row.get(0)?,id:row.get(1)?,agent_id:row.get(2)?,generation:row.get(3)?,commit:row.get(4)?,base_commit:row.get(5)?})).optional().map_err(Into::into)).await
    }

    fn repository_key(&self) -> Result<String> {
        Ok(self
            .repository
            .to_str()
            .context("repository path must be UTF-8")?
            .to_owned())
    }

    fn validate_path(&self, lease: &WorkspaceLease) -> Result<()> {
        ensure!(
            lease.slot_id >= 0 && usize::try_from(lease.slot_id)? < 4096,
            "invalid workspace slot"
        );
        let expected = self
            .directory
            .join(format!("slot-{:04}", lease.slot_id))
            .join(
                self.repository
                    .file_name()
                    .context("repository root name")?,
            );
        let parent = expected.parent().context("workspace parent")?;
        if let Ok(metadata) = std::fs::symlink_metadata(parent) {
            ensure!(
                !metadata.file_type().is_symlink() && parent.canonicalize()? == parent,
                "workspace parent escapes its pool"
            );
        }
        ensure!(
            lease.path == expected,
            "workspace lease path differs from its pool slot"
        );
        if let Ok(metadata) = std::fs::symlink_metadata(&lease.path) {
            ensure!(
                !metadata.file_type().is_symlink(),
                "workspace slot cannot be a symlink"
            );
            ensure!(
                lease.path.canonicalize()? == expected,
                "workspace path escapes its pool"
            );
        }
        Ok(())
    }

    async fn reserve_operation(&self, lease: &WorkspaceLease) -> Result<()> {
        self.validate_path(lease)?;
        let lease = lease.clone();
        let repository_key = self.repository_key()?;
        self.store.write(move |connection| {
            let transaction = connection.transaction()?;
            check_lease(&transaction, &repository_key, &lease)?;
            transaction.execute("UPDATE runtime_workspaces SET state='busy' WHERE repository=? AND slot=? AND token=?", params![repository_key,lease.slot_id,lease.token])?;
            transaction.commit()?;
            Ok(())
        }).await
    }

    async fn end_operation(&self, lease: &WorkspaceLease, state: &str) -> Result<()> {
        let lease = lease.clone();
        let repository_key = self.repository_key()?;
        let state = state.to_owned();
        self.store.write(move |connection| {
            let changed = connection.execute("UPDATE runtime_workspaces SET state=? WHERE repository=? AND slot=? AND token=? AND state='busy'", params![state,repository_key,lease.slot_id,lease.token])?;
            ensure!(changed == 1, "workspace operation owner changed");
            Ok(())
        }).await
    }
}

impl RuntimeStore {
    pub async fn workspace_has_uncheckpointed_edits(&self, task: TaskIdentity) -> Result<bool> {
        let Some(lease) = self.workspace_lease_for_task(task).await? else {
            return Ok(false);
        };
        let path = lease.path.clone();
        let head = blocking(move |cancelled| {
            if !git(
                &path,
                &["status", "--porcelain=v1", "--untracked-files=all"],
                &cancelled,
            )?
            .is_empty()
            {
                return Ok(None);
            }
            Ok(Some(
                git(&path, &["rev-parse", "HEAD"], &cancelled)?
                    .trim()
                    .to_owned(),
            ))
        })
        .await?;
        let Some(head) = head else {
            return Ok(true);
        };
        if head == lease.base_commit {
            return Ok(false);
        }
        self.read(move |connection| {
            let preserved:bool=connection.query_row("SELECT EXISTS(SELECT 1 FROM runtime_workspace_artifacts WHERE agent_id=? AND generation=? AND token=? AND commit_id=?)",params![lease.agent_id,lease.generation,lease.token,head],|row|row.get(0))?;
            Ok(!preserved)
        }).await
    }

    pub async fn workspace_resource_epoch(&self) -> Result<i64> {
        self.read(|connection| {
            connection
                .query_row(
                    "SELECT epoch FROM runtime_resource_epochs WHERE kind='workspace'",
                    [],
                    |row| row.get(0),
                )
                .map_err(Into::into)
        })
        .await
    }

    pub async fn park_workspace_job(&self, job: crate::Job, observed_epoch: i64) -> Result<()> {
        self.write(move |connection| {
            let transaction=connection.transaction()?;
            let current=transaction.query_row("SELECT epoch FROM runtime_resource_epochs WHERE kind='workspace'",[],|row|row.get::<_,i64>(0))?;
            let changed=transaction.execute("UPDATE jobs SET state='queued' WHERE id=? AND agent_id=? AND generation=? AND state='dispatched' AND EXISTS(SELECT 1 FROM agents a WHERE a.id=jobs.agent_id AND a.generation=jobs.generation AND a.state NOT IN ('cancelled','failed','completed'))",params![job.id,job.agent_id,job.generation])?;
            ensure!(changed==1,"workspace wait job is stale");
            if current==observed_epoch {transaction.execute("INSERT OR REPLACE INTO runtime_resource_waits(job_id,kind) VALUES (?,'workspace')",[&job.id])?;}
            transaction.commit()?;
            Ok(())
        }).await
    }

    pub async fn notify_workspace_available(&self) -> Result<()> {
        self.write(|connection| {
            let transaction=connection.transaction()?;
            transaction.execute("UPDATE runtime_resource_epochs SET epoch=epoch+1 WHERE kind='workspace'",[])?;
            transaction.execute("DELETE FROM runtime_resource_waits WHERE job_id IN (SELECT w.job_id FROM runtime_resource_waits w JOIN jobs j ON j.id=w.job_id JOIN agents a ON a.id=j.agent_id AND a.generation=j.generation WHERE w.kind='workspace' AND j.state='queued' AND a.pause_requested=0 AND a.state IN ('queued','running','waiting') ORDER BY j.rowid LIMIT 1)",[])?;
            transaction.commit()?;
            Ok(())
        }).await
    }

    pub(crate) fn recover_workspace_waits(connection: &Connection) -> Result<()> {
        connection.execute(
            "DELETE FROM runtime_resource_waits WHERE kind='workspace'",
            [],
        )?;
        connection.execute(
            "UPDATE runtime_resource_epochs SET epoch=epoch+1 WHERE kind='workspace'",
            [],
        )?;
        Ok(())
    }

    pub async fn workspace_lease_for_task(
        &self,
        task: TaskIdentity,
    ) -> Result<Option<WorkspaceLease>> {
        self.read(move |connection| {
            let exists:bool=connection.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_workspaces')",[],|row|row.get(0))?;
            if !exists {return Ok(None);}
            let mut statement = connection.prepare("SELECT w.slot,w.agent_id,w.generation,w.token,w.path,w.base_commit FROM runtime_workspaces w JOIN agents a ON a.id=w.agent_id AND a.generation=w.generation AND a.state NOT IN ('cancelled','failed','completed') WHERE w.agent_id=? AND w.generation=? AND w.state IN ('leased','busy') ORDER BY w.repository LIMIT 2")?;
            let mut leases = statement.query_map(params![task.agent_id,task.generation],lease_row)?.collect::<rusqlite::Result<Vec<_>>>()?;
            ensure!(leases.len() <= 1,"task has leases in multiple workspace pools");
            Ok(leases.pop())
        }).await
    }
}

fn validate_task(connection: &Connection, task: &TaskIdentity) -> Result<()> {
    let current: bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM agents WHERE id=? AND generation=? AND state NOT IN ('cancelled','completed','failed'))",params![task.agent_id,task.generation],|row|row.get(0))?;
    ensure!(current, "stale workspace owner generation");
    Ok(())
}

fn check_lease(connection: &Connection, repository: &str, lease: &WorkspaceLease) -> Result<()> {
    validate_task(
        connection,
        &TaskIdentity {
            agent_id: lease.agent_id.clone(),
            generation: lease.generation,
        },
    )?;
    let valid: bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM runtime_workspaces WHERE repository=? AND slot=? AND token=? AND agent_id=? AND generation=? AND state='leased')",params![repository,lease.slot_id,lease.token,lease.agent_id,lease.generation],|row|row.get(0))?;
    ensure!(valid, "stale workspace lease");
    Ok(())
}

fn check_operation(
    connection: &Connection,
    repository: &str,
    lease: &WorkspaceLease,
) -> Result<()> {
    validate_task(
        connection,
        &TaskIdentity {
            agent_id: lease.agent_id.clone(),
            generation: lease.generation,
        },
    )?;
    let valid: bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM runtime_workspaces WHERE repository=? AND slot=? AND token=? AND agent_id=? AND generation=? AND state='busy')",params![repository,lease.slot_id,lease.token,lease.agent_id,lease.generation],|row|row.get(0))?;
    ensure!(valid, "stale workspace operation");
    Ok(())
}

fn lease_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<WorkspaceLease> {
    Ok(WorkspaceLease {
        slot_id: row.get(0)?,
        agent_id: row.get(1)?,
        generation: row.get(2)?,
        token: row.get(3)?,
        path: PathBuf::from(row.get::<_, String>(4)?),
        base_commit: row.get(5)?,
    })
}

struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

struct Cancellation {
    local: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
}
trait CancellationFlag {
    fn cancelled(&self) -> bool;
}
impl CancellationFlag for AtomicBool {
    fn cancelled(&self) -> bool {
        self.load(Ordering::Acquire)
    }
}
impl CancellationFlag for Cancellation {
    fn cancelled(&self) -> bool {
        self.local.load(Ordering::Acquire) || self.stopping.load(Ordering::Acquire)
    }
}

async fn blocking<T: Send + 'static>(
    operation: impl FnOnce(Cancellation) -> Result<T> + Send + 'static,
) -> Result<T> {
    blocking_with_stop(operation, Arc::new(AtomicBool::new(false))).await
}

async fn blocking_with_stop<T: Send + 'static>(
    operation: impl FnOnce(Cancellation) -> Result<T> + Send + 'static,
    stopping: Arc<AtomicBool>,
) -> Result<T> {
    let cancelled = Arc::new(AtomicBool::new(false));
    let guard = CancelOnDrop(cancelled.clone());
    let (sender, receiver) = futures::channel::oneshot::channel();
    std::thread::Builder::new()
        .name("agent-workspace".into())
        .spawn(move || {
            let result = operation(Cancellation {
                local: cancelled,
                stopping,
            });
            if let Err(Err(error)) = sender.send(result) {
                log::error!("detached workspace operation failed: {error:#}");
            }
        })?;
    let result = receiver.await.context("workspace worker stopped")?;
    drop(guard);
    result
}

struct GitProcess(std::process::Child);

impl Drop for GitProcess {
    fn drop(&mut self) {
        match self.0.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => {
                if let Err(error) = terminate_git(&mut self.0) {
                    log::debug!("git exited during cleanup: {error}");
                }
                if let Err(error) = self.0.wait() {
                    log::error!("failed to reap git process: {error}");
                }
            }
        }
    }
}

#[allow(
    clippy::disallowed_methods,
    reason = "Git runs on an owned OS worker thread and these stdio settings are never converted to smol."
)]
fn terminate_git(child: &mut std::process::Child) -> Result<()> {
    #[cfg(unix)]
    {
        let status = Command::new("/bin/kill")
            .args(["-KILL", "--", &format!("-{}", child.id())])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;
        if !status.success() && child.try_wait()?.is_none() {
            child.kill()?;
        }
    }
    #[cfg(windows)]
    {
        let status = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;
        if !status.success() && child.try_wait()?.is_none() {
            child.kill()?;
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        child.kill()?;
    }
    Ok(())
}

#[allow(
    clippy::disallowed_methods,
    reason = "Git runs on an owned OS worker thread and these stdio settings are never converted to smol."
)]
fn git(repository: &Path, arguments: &[&str], cancelled: &impl CancellationFlag) -> Result<String> {
    ensure!(!cancelled.cancelled(), "workspace operation was cancelled");
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(repository)
        .args(["-c", "core.hooksPath=/dev/null"])
        .args(arguments)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .env_remove("GIT_COMMON_DIR")
        .env_remove("GIT_OBJECT_DIRECTORY")
        .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES")
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        command.creation_flags(0x00000200);
    }
    let mut child = GitProcess(command.spawn()?);
    let stdout = child.0.stdout.take().context("git stdout")?;
    let stderr = child.0.stderr.take().context("git stderr")?;
    let output = std::thread::Builder::new()
        .name("workspace-git-output".into())
        .spawn(move || {
            let mut bytes = Vec::new();
            stdout
                .take(1024 * 1024)
                .read_to_end(&mut bytes)
                .map(|_| bytes)
        })?;
    let errors = std::thread::Builder::new()
        .name("workspace-git-errors".into())
        .spawn(move || {
            let mut bytes = Vec::new();
            stderr
                .take(1024 * 1024)
                .read_to_end(&mut bytes)
                .map(|_| bytes)
        })?;
    let deadline = Instant::now() + Duration::from_secs(120);
    let interrupted = loop {
        if child.0.try_wait()?.is_some() {
            break false;
        }
        if cancelled.cancelled() || Instant::now() >= deadline {
            if let Err(error) = terminate_git(&mut child.0) {
                log::debug!("git exited while cancellation was requested: {error}");
            }
            break true;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let status = child.0.wait()?;
    let stdout = output
        .join()
        .map_err(|_| anyhow!("git output reader panicked"))??;
    let stderr = errors
        .join()
        .map_err(|_| anyhow!("git error reader panicked"))??;
    ensure!(
        !interrupted,
        "workspace git operation cancelled or timed out"
    );
    ensure!(
        status.success(),
        "git {} failed: {}",
        arguments.first().copied().unwrap_or("operation"),
        String::from_utf8_lossy(&stderr)
    );
    Ok(String::from_utf8(stdout)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeLimits;
    use serde_json::json;

    async fn fixture(capacity: usize) -> Result<(tempfile::TempDir, RuntimeStore, WorkspacePool)> {
        let directory = tempfile::tempdir()?;
        let repository = directory.path().join("repository");
        std::fs::create_dir(&repository)?;
        let cancelled = AtomicBool::new(false);
        git(&repository, &["init"], &cancelled)?;
        git(
            &repository,
            &["config", "user.name", "Runtime test"],
            &cancelled,
        )?;
        git(
            &repository,
            &["config", "user.email", "runtime@example.invalid"],
            &cancelled,
        )?;
        std::fs::write(repository.join("file.txt"), "base")?;
        git(&repository, &["add", "--all"], &cancelled)?;
        git(&repository, &["commit", "-m", "initial"], &cancelled)?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        store.create_root("root".into(), json!({})).await?;
        let pool = WorkspacePool::open(
            store.clone(),
            repository,
            directory.path().join("pool"),
            capacity,
        )
        .await?;
        Ok((directory, store, pool))
    }

    fn owner() -> TaskIdentity {
        TaskIdentity {
            agent_id: "root".into(),
            generation: 0,
        }
    }

    #[tokio::test]
    async fn workspace_capacity_and_checkpoint_survive_reuse() -> Result<()> {
        let (_directory, store, pool) = fixture(1).await?;
        let lease = pool
            .acquire(owner(), "HEAD".into())
            .await?
            .context("lease")?;
        let duplicate = pool
            .acquire(owner(), "HEAD".into())
            .await?
            .context("idempotent acquisition")?;
        assert_eq!(duplicate.token, lease.token);
        assert_eq!(duplicate.path, lease.path);
        store.create_root("another".into(), json!({})).await?;
        assert!(
            pool.acquire(
                TaskIdentity {
                    agent_id: "another".into(),
                    generation: 0
                },
                "HEAD".into()
            )
            .await?
            .is_none()
        );
        std::fs::write(lease.path.join("file.txt"), "agent change")?;
        assert!(pool.release(lease.clone()).await.is_err());
        let artifact = pool.checkpoint(lease.clone(), "checkpoint".into()).await?;
        pool.release(lease.clone()).await?;
        let next = pool
            .acquire(owner(), "HEAD".into())
            .await?
            .context("reused")?;
        assert_eq!(next.slot_id, lease.slot_id);
        assert_ne!(next.token, lease.token);
        assert_eq!(std::fs::read_to_string(next.path.join("file.txt"))?, "base");
        assert!(pool.checkpoint(lease, "stale".into()).await.is_err());
        let saved = git(
            &pool.repository,
            &["show", &format!("{}:file.txt", artifact.commit)],
            &AtomicBool::new(false),
        )?;
        assert_eq!(saved, "agent change");
        pool.release(next).await?;
        store.shutdown().await
    }

    #[tokio::test]
    async fn resuming_a_paused_workspace_waiter_rechecks_capacity() -> Result<()> {
        let (_directory, store, pool) = fixture(1).await?;
        let occupied = pool
            .acquire(owner(), "HEAD".into())
            .await?
            .context("occupied workspace")?;
        store.create_root("waiter".into(), json!({})).await?;
        store
            .enqueue_text_input("waiter".into(), "request".into(), "edit".into())
            .await?;
        let slice = store.admit().await?.context("waiting task slice")?;
        assert_eq!(slice.agent_id, "waiter");
        store
            .record_job(
                slice.clone(),
                "workspace-request".into(),
                json!({"tool":"runtime_workspace_acquire","arguments":{},"tool_call_id":"acquire"}),
            )
            .await?;
        store
            .finish(slice, json!({}), crate::SliceOutcome::Waiting)
            .await?;
        let dispatched = store.dispatch_job("workspace-request".into()).await?;
        let epoch = store.workspace_resource_epoch().await?;
        store.park_workspace_job(dispatched, epoch).await?;
        store.pause("waiter".into()).await?;

        pool.release(occupied).await?;
        assert!(store.queued_jobs(8).await?.is_empty());
        let still_blocked: bool = store.read(|connection| connection.query_row("SELECT EXISTS(SELECT 1 FROM runtime_resource_waits WHERE job_id='workspace-request')", [], |row|row.get(0)).map_err(Into::into)).await?;
        assert!(
            still_blocked,
            "release should preserve a paused task's resource wait"
        );

        store.resume("waiter".into()).await?;
        let ready = store.queued_jobs(8).await?;
        assert_eq!(ready.len(), 1);
        assert_eq!(
            ready.first().context("resumed workspace request")?.id,
            "workspace-request"
        );
        let workspace = pool
            .acquire(
                TaskIdentity {
                    agent_id: "waiter".into(),
                    generation: 0,
                },
                "HEAD".into(),
            )
            .await?
            .context("capacity after resume")?;
        pool.release(workspace).await?;
        store.shutdown().await
    }

    #[tokio::test]
    async fn resizing_preserves_existing_leases_and_artifacts() -> Result<()> {
        let (_directory, store, pool) = fixture(1).await?;
        let first = pool
            .acquire(owner(), "HEAD".into())
            .await?
            .context("first")?;
        store.create_root("second".into(), json!({})).await?;
        let grown = WorkspacePool::open(
            store.clone(),
            pool.repository.clone(),
            pool.directory.clone(),
            2,
        )
        .await?;
        let second = grown
            .acquire(
                TaskIdentity {
                    agent_id: "second".into(),
                    generation: 0,
                },
                "HEAD".into(),
            )
            .await?
            .context("grown slot")?;
        assert_eq!(second.slot_id, 1);
        let smaller = WorkspacePool::open(
            store.clone(),
            pool.repository.clone(),
            pool.directory.clone(),
            1,
        )
        .await?;
        smaller
            .checkpoint(
                second.clone(),
                "preserve checkpoint in a retired slot".into(),
            )
            .await?;
        smaller.release(second).await?;
        store.create_root("third".into(), json!({})).await?;
        assert!(
            smaller
                .acquire(
                    TaskIdentity {
                        agent_id: "third".into(),
                        generation: 0
                    },
                    "HEAD".into()
                )
                .await?
                .is_none()
        );
        smaller.release(first).await?;
        store.shutdown().await
    }

    #[tokio::test]
    async fn interrupted_owner_preserves_dirty_edits_for_explicit_resume() -> Result<()> {
        let (directory, store, pool) = fixture(1).await?;
        let lease = pool
            .acquire(owner(), "HEAD".into())
            .await?
            .context("lease")?;
        std::fs::write(lease.path.join("file.txt"), "unfinished")?;
        store.shutdown().await?;
        let restored = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        let pool =
            WorkspacePool::open(restored.clone(), pool.repository, pool.directory, 1).await?;
        let recovered = pool.recover().await?;
        assert_eq!(recovered.len(), 1);
        assert_eq!(
            std::fs::read_to_string(lease.path.join("file.txt"))?,
            "unfinished"
        );
        let generation = restored.snapshot("root".into()).await?.generation;
        restored.resume("root".into()).await?;
        let resumed = pool
            .acquire(
                TaskIdentity {
                    agent_id: "root".into(),
                    generation,
                },
                "HEAD".into(),
            )
            .await?
            .context("restored dirty lease")?;
        assert_ne!(resumed.token, lease.token);
        assert_eq!(
            std::fs::read_to_string(resumed.path.join("file.txt"))?,
            "unfinished"
        );
        pool.checkpoint(resumed.clone(), "recover edits".into())
            .await?;
        pool.release(resumed).await?;
        restored.shutdown().await
    }
}
