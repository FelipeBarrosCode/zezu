use std::{cell::RefCell, rc::Rc};

use agent_client_protocol::v2::schema as acp;
use agent_settings::AgentProfileId;
use anyhow::{Result, anyhow};
use gpui::{App, Task};

pub type RuntimeModeSetter = Rc<dyn Fn(AgentProfileId, &mut App) -> Task<Result<()>>>;
pub type RuntimeModeGetter = Rc<dyn Fn() -> AgentProfileId>;

pub struct RuntimeSessionModes {
    current: Rc<RefCell<AgentProfileId>>,
    modes: Vec<(AgentProfileId, String)>,
    setter: RuntimeModeSetter,
    gate: Rc<futures::lock::Mutex<()>>,
    getter: Option<RuntimeModeGetter>,
}

impl RuntimeSessionModes {
    pub fn new(
        current: AgentProfileId,
        modes: Vec<(AgentProfileId, String)>,
        setter: RuntimeModeSetter,
    ) -> Self {
        Self {
            current: Rc::new(RefCell::new(current)),
            modes,
            setter,
            gate: Rc::default(),
            getter: None,
        }
    }

    pub fn with_current_profile(mut self, getter: RuntimeModeGetter) -> Self {
        self.getter = Some(getter);
        self
    }
}

impl acp_thread::AgentSessionModes for RuntimeSessionModes {
    fn current_mode(&self) -> acp::SessionConfigValueId {
        let profile = self
            .getter
            .as_ref()
            .map(|getter| getter())
            .unwrap_or_else(|| self.current.borrow().clone());
        acp::SessionConfigValueId::new(profile.as_str().to_owned())
    }
    fn all_modes(&self) -> Vec<acp::SessionConfigSelectOption> {
        self.modes
            .iter()
            .map(|(id, name)| {
                acp::SessionConfigSelectOption::new(
                    acp::SessionConfigValueId::new(id.as_str().to_owned()),
                    name.clone(),
                )
            })
            .collect()
    }
    fn set_mode(&self, mode: acp::SessionConfigValueId, cx: &mut App) -> Task<Result<()>> {
        let Some((profile, _)) = self
            .modes
            .iter()
            .find(|(id, _)| id.as_str() == mode.0.as_ref())
        else {
            return Task::ready(Err(anyhow!("agent profile is unavailable: {}", mode.0)));
        };
        let profile = profile.clone();
        let current = self.current.clone();
        let setter = self.setter.clone();
        let gate = self.gate.clone();
        cx.spawn(async move |cx| {
            let _guard = gate.lock().await;
            cx.update(|cx| setter(profile.clone(), cx)).await?;
            *current.borrow_mut() = profile;
            Ok(())
        })
    }
}

pub type RuntimeConfigSetter = Rc<
    dyn Fn(
        acp::SessionConfigId,
        acp::SessionConfigOptionValue,
        &mut App,
    ) -> Task<Result<Vec<acp::SessionConfigOption>>>,
>;

pub struct RuntimeSessionConfigOptions {
    options: Rc<RefCell<Vec<acp::SessionConfigOption>>>,
    setter: RuntimeConfigSetter,
    gate: Rc<futures::lock::Mutex<()>>,
    changed: Rc<RefCell<watch::Sender<()>>>,
    receiver: watch::Receiver<()>,
}

impl RuntimeSessionConfigOptions {
    pub fn new(options: Vec<acp::SessionConfigOption>, setter: RuntimeConfigSetter) -> Self {
        let (changed, receiver) = watch::channel(());
        Self {
            options: Rc::new(RefCell::new(options)),
            setter,
            gate: Rc::default(),
            changed: Rc::new(RefCell::new(changed)),
            receiver,
        }
    }

    pub fn replace_options(&self, options: Vec<acp::SessionConfigOption>) {
        *self.options.borrow_mut() = options;
        if self.changed.borrow_mut().send(()).is_err() {
            log::debug!("runtime configuration watcher detached");
        }
    }
}

impl acp_thread::AgentSessionConfigOptions for RuntimeSessionConfigOptions {
    fn config_options(&self) -> Vec<acp::SessionConfigOption> {
        self.options.borrow().clone()
    }
    fn set_config_option(
        &self,
        id: acp::SessionConfigId,
        value: acp::SessionConfigOptionValue,
        cx: &mut App,
    ) -> Task<Result<Vec<acp::SessionConfigOption>>> {
        let options = self.options.clone();
        let setter = self.setter.clone();
        let gate = self.gate.clone();
        let changed = self.changed.clone();
        cx.spawn(async move |cx| {
            let _guard = gate.lock().await;
            let supported = options
                .borrow()
                .iter()
                .find(|option| option.config_id == id)
                .is_some_and(|option| supports_value(option, &value));
            if !supported {
                return Err(anyhow!(
                    "unsupported agent configuration value for {}",
                    id.0
                ));
            }
            let updated = cx.update(|cx| setter(id, value, cx)).await?;
            *options.borrow_mut() = updated.clone();
            if changed.borrow_mut().send(()).is_err() {
                log::debug!("runtime configuration watcher detached");
            }
            Ok(updated)
        })
    }
    fn watch(&self, _: &mut App) -> Option<watch::Receiver<()>> {
        Some(self.receiver.clone())
    }
}

fn supports_value(
    option: &acp::SessionConfigOption,
    value: &acp::SessionConfigOptionValue,
) -> bool {
    match (&option.kind, value) {
        (acp::SessionConfigKind::Boolean(_), acp::SessionConfigOptionValue::Boolean { .. }) => true,
        (acp::SessionConfigKind::Select(select), acp::SessionConfigOptionValue::Id { value }) => {
            match &select.options {
                acp::SessionConfigSelectOptions::Ungrouped(options) => {
                    options.iter().any(|option| &option.value == value)
                }
                acp::SessionConfigSelectOptions::Grouped(groups) => groups
                    .iter()
                    .any(|group| group.options.iter().any(|option| &option.value == value)),
                _ => false,
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acp_thread::{AgentSessionConfigOptions as _, AgentSessionModes as _};

    #[gpui::test]
    async fn profile_changes_wait_for_success_and_unknown_profiles_never_call_setter(
        cx: &mut gpui::TestAppContext,
    ) {
        let calls = Rc::new(std::cell::Cell::new(0));
        let setter: RuntimeModeSetter = {
            let calls = calls.clone();
            Rc::new(move |_, _| {
                calls.set(calls.get() + 1);
                Task::ready(Err(anyhow!("storage unavailable")))
            })
        };
        let modes = RuntimeSessionModes::new(
            AgentProfileId("ask".into()),
            vec![
                (AgentProfileId("ask".into()), "Ask".into()),
                (AgentProfileId("write".into()), "Write".into()),
            ],
            setter,
        );
        assert!(
            cx.update(|cx| modes.set_mode("unknown".into(), cx))
                .await
                .is_err()
        );
        assert_eq!(calls.get(), 0);
        assert!(
            cx.update(|cx| modes.set_mode("write".into(), cx))
                .await
                .is_err()
        );
        assert_eq!(calls.get(), 1);
        assert_eq!(modes.current_mode().0.as_ref(), "ask");
    }

    #[gpui::test]
    async fn config_options_reject_unknown_values_and_publish_committed_state(
        cx: &mut gpui::TestAppContext,
    ) {
        let setter: RuntimeConfigSetter = Rc::new(|_, _, _| {
            Task::ready(Ok(vec![acp::SessionConfigOption::boolean(
                "thinking", "Thinking", true,
            )]))
        });
        let options = RuntimeSessionConfigOptions::new(
            vec![acp::SessionConfigOption::boolean(
                "thinking", "Thinking", false,
            )],
            setter,
        );
        assert!(
            cx.update(|cx| options.set_config_option(
                "thinking".into(),
                acp::SessionConfigOptionValue::id("invalid"),
                cx
            ))
            .await
            .is_err()
        );
        let mut receiver = cx.update(|cx| options.watch(cx)).expect("watcher");
        cx.update(|cx| {
            options.set_config_option(
                "thinking".into(),
                acp::SessionConfigOptionValue::boolean(true),
                cx,
            )
        })
        .await
        .expect("update");
        receiver
            .changed()
            .await
            .expect("committed state notification");
        let option = options
            .config_options()
            .into_iter()
            .next()
            .expect("thinking");
        assert!(
            matches!(option.kind,acp::SessionConfigKind::Boolean(boolean) if boolean.current_value)
        );
    }
}
