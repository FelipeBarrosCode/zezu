use std::{ops::Range, time::Duration};

use agent::NativeRuntime;
use agent_runtime::{AgentSnapshot, FleetSummary, RuntimeStore, TranscriptPage, UncertainJob};
use anyhow::{Context as _, Result};
use gpui::{App, Context, Entity, Render, Task, Window, uniform_list};
use ui::{Button, Label, prelude::*};
use util::ResultExt as _;

const PAGE_SIZE: usize = 100;
const TRANSCRIPT_PAGE_SIZE: usize = 50;

pub struct RuntimeFleet {
    host: Entity<NativeRuntime>,
    rows: Vec<AgentSnapshot>,
    summary: FleetSummary,
    root: Option<String>,
    after: Option<String>,
    next: Option<String>,
    selected: Option<String>,
    transcript: Option<TranscriptPage>,
    transcript_before: Option<i64>,
    error: Option<String>,
    uncertain_jobs: Vec<UncertainJob>,
    _refresh: Task<()>,
    _transcript: Option<Task<()>>,
    _command: Option<Task<()>>,
}

impl RuntimeFleet {
    pub fn new(host: Entity<NativeRuntime>, cx: &mut Context<Self>) -> Self {
        let refresh = cx.spawn(async move |this, cx| {
            loop {
                let parameters = this.read_with(cx, |this, cx| {
                    this.host.read(cx).active().map(|runtime| {
                        (
                            runtime.store,
                            this.root.clone(),
                            this.after.clone(),
                            this.selected.clone(),
                        )
                    })
                });
                let Ok(parameters) = parameters else { return };
                if let Some((store, root, after, selected)) = parameters {
                    let page = store
                        .fleet_page(root.clone(), after.clone(), PAGE_SIZE)
                        .await;
                    let summary = store.fleet_summary(root.clone()).await;
                    let uncertain = match &selected {
                        Some(id) => store.uncertain_jobs(id.clone(), 20).await,
                        None => Ok(Vec::new()),
                    };
                    if this
                        .update(cx, |this, cx| {
                            if this.after != after || this.root != root {
                                return;
                            }
                            match (page, summary) {
                                (Ok(page), Ok(summary)) => {
                                    this.rows = page.agents;
                                    this.next = page.next_cursor;
                                    this.summary = summary;
                                }
                                (Err(error), _) | (_, Err(error)) => {
                                    this.error = Some(format!("{error:#}"))
                                }
                            }
                            if this.selected == selected {
                                match uncertain {
                                    Ok(jobs) => this.uncertain_jobs = jobs,
                                    Err(error) => this.error = Some(format!("{error:#}")),
                                }
                            }
                            cx.notify();
                        })
                        .is_err()
                    {
                        return;
                    }
                }
                cx.background_executor().timer(Duration::from_secs(1)).await;
            }
        });
        Self {
            host,
            rows: Vec::new(),
            summary: FleetSummary::default(),
            root: None,
            after: None,
            next: None,
            selected: None,
            transcript: None,
            transcript_before: None,
            error: None,
            uncertain_jobs: Vec::new(),
            _refresh: refresh,
            _transcript: None,
            _command: None,
        }
    }

    fn store(&self, cx: &App) -> Result<RuntimeStore> {
        self.host
            .read(cx)
            .active()
            .map(|runtime| runtime.store)
            .context("The native runtime starts with your first native conversation")
    }

    fn select(&mut self, id: String, before: Option<i64>, cx: &mut Context<Self>) {
        let store = match self.store(cx) {
            Ok(store) => store,
            Err(error) => {
                self.error = Some(error.to_string());
                cx.notify();
                return;
            }
        };
        self.selected = Some(id.clone());
        self.uncertain_jobs.clear();
        self.transcript_before = before;
        self._transcript = Some(cx.spawn(async move |this, cx| {
            let page = store
                .transcript_page(id.clone(), before, TRANSCRIPT_PAGE_SIZE)
                .await;
            let error = store.agent_error(id.clone()).await;
            let uncertain = store.uncertain_jobs(id.clone(), 20).await;
            this.update(cx, |this, cx| {
                if this.selected.as_ref() != Some(&id) || this.transcript_before != before {
                    return;
                }
                match (page, error, uncertain) {
                    (Ok(page), Ok(error), Ok(jobs)) => {
                        this.transcript = Some(page);
                        this.error = error;
                        this.uncertain_jobs = jobs;
                    }
                    (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => {
                        this.error = Some(format!("{error:#}"))
                    }
                }
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
    }

    fn acknowledge(&mut self, job: UncertainJob, cx: &mut Context<Self>) {
        let store = match self.store(cx) {
            Ok(store) => store,
            Err(error) => {
                self.error = Some(error.to_string());
                cx.notify();
                return;
            }
        };
        self._command = Some(cx.spawn(async move |this, cx| {
            let result = store
                .acknowledge_uncertain_job(
                    job.id,
                    job.agent_id.clone(),
                    job.generation,
                    job.agent_state,
                )
                .await;
            let succeeded = result.is_ok();
            this.update(cx, |this, cx| {
                apply_command_result(&mut this.error, result);
                if succeeded && this.selected.as_ref() == Some(&job.agent_id) {
                    this.select(job.agent_id, this.transcript_before, cx);
                }
                cx.notify();
            })
            .log_err();
        }));
    }

    fn command(&mut self, operation: &'static str, cx: &mut Context<Self>) {
        let Some(id) = self.selected.clone() else {
            return;
        };
        let store = match self.store(cx) {
            Ok(store) => store,
            Err(error) => {
                self.error = Some(error.to_string());
                cx.notify();
                return;
            }
        };
        self._command = Some(cx.spawn(async move |this, cx| {
            let result = match operation {
                "pause" => store.pause(id).await,
                "resume" => store.resume(id).await,
                "stop" => store.cancel(id).await,
                _ => Err(anyhow::anyhow!("unknown runtime command")),
            };
            this.update(cx, |this, cx| {
                self::apply_command_result(&mut this.error, result);
                cx.notify();
            })
            .log_err();
        }));
    }

    fn render_rows(
        &mut self,
        range: Range<usize>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<gpui::AnyElement> {
        range
            .filter_map(|index| self.rows.get(index).cloned())
            .map(|agent| {
                let selected = self.selected.as_ref() == Some(&agent.id);
                let id = agent.id.clone();
                Button::new(
                    index_key(&agent.id),
                    format!("{} · {}", agent.state, agent.id),
                )
                .toggle_state(selected)
                .on_click(cx.listener(move |this, _, _, cx| this.select(id.clone(), None, cx)))
                .into_any_element()
            })
            .collect()
    }
}

fn apply_command_result(error: &mut Option<String>, result: Result<()>) {
    *error = result.err().map(|error| format!("{error:#}"));
}

fn index_key(id: &str) -> gpui::SharedString {
    id.to_owned().into()
}

impl Render for RuntimeFleet {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let transcript = self
            .transcript
            .as_ref()
            .map(|page| {
                page.entries
                    .iter()
                    .map(|entry| {
                        serde_json::to_string_pretty(&entry.payload)
                            .unwrap_or_else(|error| error.to_string())
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n")
            })
            .unwrap_or_default();
        let older = self.transcript.as_ref().and_then(|page| page.next_cursor);
        let inherited = self
            .transcript
            .as_ref()
            .and_then(|page| page.inherited.clone());
        v_flex()
            .size_full()
            .gap_2()
            .p_2()
            .child(Label::new(format!(
                "{} agents · {} running · {} queued · {} paused",
                self.summary.total, self.summary.running, self.summary.queued, self.summary.paused
            )))
            .child(
                h_flex()
                    .gap_1()
                    .child(
                        Button::new("runtime-first", "First page").on_click(cx.listener(
                            |this, _, _, cx| {
                                this.after = None;
                                this.rows.clear();
                                cx.notify();
                            },
                        )),
                    )
                    .child(
                        Button::new("runtime-next", "Next page")
                            .disabled(self.next.is_none())
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.after = this.next.clone();
                                this.rows.clear();
                                cx.notify();
                            })),
                    ),
            )
            .child(
                h_flex()
                    .gap_1()
                    .child(
                        Button::new("runtime-all-runs", "All runs")
                            .disabled(self.root.is_none())
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.root = None;
                                this.after = None;
                                this.rows.clear();
                                cx.notify();
                            })),
                    )
                    .when_some(self.selected.clone(), |parent, id| {
                        parent.child(
                            Button::new("runtime-selected-run", "Only this run").on_click(
                                cx.listener(move |this, _, _, cx| {
                                    if let Some(agent) =
                                        this.rows.iter().find(|agent| agent.id == id)
                                    {
                                        this.root = Some(agent.root_id.clone());
                                        this.after = None;
                                        this.rows.clear();
                                        cx.notify();
                                    }
                                }),
                            ),
                        )
                    }),
            )
            .child(
                uniform_list(
                    "runtime-agents",
                    self.rows.len(),
                    cx.processor(Self::render_rows),
                )
                .h(px(200.))
                .w_full(),
            )
            .when_some(self.selected.clone(), |parent, id| {
                parent.child(
                    h_flex()
                        .gap_1()
                        .child(
                            Button::new("runtime-pause", "Pause")
                                .on_click(cx.listener(|this, _, _, cx| this.command("pause", cx))),
                        )
                        .child(
                            Button::new("runtime-resume", "Resume")
                                .on_click(cx.listener(|this, _, _, cx| this.command("resume", cx))),
                        )
                        .child(
                            Button::new("runtime-stop", "Stop subtree")
                                .on_click(cx.listener(|this, _, _, cx| this.command("stop", cx))),
                        )
                        .child(Button::new("runtime-latest", "Latest messages").on_click(
                            cx.listener(move |this, _, _, cx| this.select(id.clone(), None, cx)),
                        )),
                )
            })
            .when_some(older, |parent, before| {
                parent.child(
                    Button::new("runtime-older", "Older messages").on_click(cx.listener(
                        move |this, _, _, cx| {
                            if let Some(id) = this.selected.clone() {
                                this.select(id, Some(before), cx);
                            }
                        },
                    )),
                )
            })
            .when(!self.uncertain_jobs.is_empty(), |parent| {
                parent.child(
                    v_flex().id("runtime-recovery").max_h(px(240.)).overflow_y_scroll().gap_1()
                        .child(Label::new("Interrupted tools have unknown outcomes. They may have changed files or external state. Inspect changes before continuing. Acknowledging records an error and never retries the operation. Resume or send a message after resolving all interrupted tools.").color(Color::Warning))
                        .children(self.uncertain_jobs.iter().cloned().map(|job| {
                            h_flex().gap_1()
                                .child(Label::new(format!("{} · {}", job.tool_name.as_deref().unwrap_or("Tool"), job.id)))
                                .child(Button::new(index_key(&format!("runtime-recover-{}", job.id)), "Continue without retrying")
                                    .disabled(job.agent_state == "running")
                                    .on_click(cx.listener(move |this, _, _, cx| this.acknowledge(job.clone(), cx))))
                        }))
                )
            })
            .when_some(inherited, |parent, reference| {
                parent.child(
                    Button::new("runtime-inherited", "Inherited context").on_click(cx.listener(
                        move |this, _, _, cx| {
                            this.select(
                                reference.agent_id.clone(),
                                Some(reference.cutoff.saturating_add(1)),
                                cx,
                            )
                        },
                    )),
                )
            })
            .when_some(self.error.clone(), |parent, error| {
                parent.child(Label::new(error).color(Color::Error))
            })
            .child(
                div()
                    .id("runtime-transcript")
                    .flex_1()
                    .overflow_y_scroll()
                    .child(Label::new(transcript)),
            )
    }
}
