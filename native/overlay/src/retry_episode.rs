use std::collections::VecDeque;
use std::time::Duration;

use tokio::time::Instant;

use crate::state::{NativeQuietTailEpisode, NativeQuietTailPhase};

pub(crate) const RETRY_AUDIT_CAPACITY: usize = 128;

#[derive(Debug, Clone, Copy)]
pub(crate) struct FreshRetryPolicy {
    pub(crate) cadence: Duration,
    pub(crate) deadline: Duration,
    pub(crate) max_completed: u32,
}

impl FreshRetryPolicy {
    pub(crate) fn new(cadence: Duration, deadline: Duration, max_completed: u32) -> Self {
        Self {
            cadence,
            deadline,
            max_completed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FreshRetryChannel {
    SelfChannel,
    Peer,
}

impl FreshRetryChannel {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::SelfChannel => "self",
            Self::Peer => "peer",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FreshSchedule {
    pub(crate) channel: FreshRetryChannel,
    pub(crate) trigger_generation: u64,
    pub(crate) required_scene_generation: u64,
    pub(crate) target_identity: String,
    pub(crate) completed: u32,
    pub(crate) max_completed: u32,
    pub(crate) phase: NativeQuietTailPhase,
    pub(crate) episode_generation: u64,
    pub(crate) deadline: Instant,
    pub(crate) next_due: Instant,
}

impl FreshSchedule {
    pub(crate) fn same_intent(&self, other: &Self) -> bool {
        self.channel == other.channel
            && self.trigger_generation == other.trigger_generation
            && self.required_scene_generation == other.required_scene_generation
            && self.target_identity == other.target_identity
            && self.episode_generation == other.episode_generation
            && self.phase == other.phase
    }

    pub(crate) fn expired_at(&self, now: Instant) -> bool {
        now > self.deadline
    }

    pub(crate) fn accepts_transferred_due_from(&self, other: &Self) -> bool {
        self.channel == other.channel
            && self.trigger_generation > other.trigger_generation
            && self.required_scene_generation > other.required_scene_generation
            && self.target_identity == other.target_identity
            && self.completed == other.completed
            && self.max_completed == other.max_completed
            && self.phase == other.phase
            && self.episode_generation == other.episode_generation
            && self.deadline == other.deadline
            && self.next_due == other.next_due
    }
}

#[derive(Debug, Clone)]
pub(crate) struct EpisodeAccounting {
    pub(crate) target_identity: String,
    pub(crate) phase: NativeQuietTailPhase,
    pub(crate) episode_generation: u64,
    pub(crate) completed: u32,
    pub(crate) max_completed: u32,
    pub(crate) deadline: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FreshAuditFact {
    pub(crate) channel: FreshRetryChannel,
    pub(crate) trigger_generation: u64,
    pub(crate) outcome: &'static str,
    pub(crate) completed: u32,
    pub(crate) at: Duration,
}

#[derive(Debug, Clone)]
pub(crate) struct RetryIntent {
    pub(crate) generation: Option<u64>,
    pub(crate) episode: Option<NativeQuietTailEpisode>,
    pub(crate) target_identity: Option<String>,
    pub(crate) required_scene_generation: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct RetryTransition {
    pub(crate) schedule: FreshSchedule,
    pub(crate) outcome: &'static str,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct RetryReconcileResult {
    pub(crate) transitions: Vec<RetryTransition>,
    pub(crate) cause_transfer: Option<(FreshSchedule, FreshSchedule)>,
}

#[derive(Debug)]
pub(crate) struct RetryEpisodes {
    observed_self_generation: Option<u64>,
    observed_peer_generation: Option<u64>,
    self_schedule: Option<FreshSchedule>,
    peer_schedule: Option<FreshSchedule>,
    self_accounting: Option<EpisodeAccounting>,
    peer_accounting: Option<EpisodeAccounting>,
    self_ended_episode: Option<(String, NativeQuietTailPhase, u64)>,
    peer_ended_episode: Option<(String, NativeQuietTailPhase, u64)>,
    audit_started_at: Instant,
    audit: VecDeque<FreshAuditFact>,
    audit_dropped: u64,
}

impl RetryEpisodes {
    pub(crate) fn new() -> Self {
        Self {
            observed_self_generation: None,
            observed_peer_generation: None,
            self_schedule: None,
            peer_schedule: None,
            self_accounting: None,
            peer_accounting: None,
            self_ended_episode: None,
            peer_ended_episode: None,
            audit_started_at: Instant::now(),
            audit: VecDeque::with_capacity(RETRY_AUDIT_CAPACITY),
            audit_dropped: 0,
        }
    }

    pub(crate) fn observed_generation(&self, channel: FreshRetryChannel) -> Option<u64> {
        match channel {
            FreshRetryChannel::SelfChannel => self.observed_self_generation,
            FreshRetryChannel::Peer => self.observed_peer_generation,
        }
    }

    pub(crate) fn observe_generation(
        &mut self,
        channel: FreshRetryChannel,
        generation: Option<u64>,
    ) {
        match channel {
            FreshRetryChannel::SelfChannel => self.observed_self_generation = generation,
            FreshRetryChannel::Peer => self.observed_peer_generation = generation,
        }
    }

    pub(crate) fn schedule(&self, channel: FreshRetryChannel) -> Option<&FreshSchedule> {
        match channel {
            FreshRetryChannel::SelfChannel => self.self_schedule.as_ref(),
            FreshRetryChannel::Peer => self.peer_schedule.as_ref(),
        }
    }

    pub(crate) fn schedule_mut(
        &mut self,
        channel: FreshRetryChannel,
    ) -> Option<&mut FreshSchedule> {
        match channel {
            FreshRetryChannel::SelfChannel => self.self_schedule.as_mut(),
            FreshRetryChannel::Peer => self.peer_schedule.as_mut(),
        }
    }

    pub(crate) fn replace_schedule(
        &mut self,
        channel: FreshRetryChannel,
        schedule: FreshSchedule,
    ) -> Option<FreshSchedule> {
        match channel {
            FreshRetryChannel::SelfChannel => self.self_schedule.replace(schedule),
            FreshRetryChannel::Peer => self.peer_schedule.replace(schedule),
        }
    }

    pub(crate) fn take_schedule(&mut self, channel: FreshRetryChannel) -> Option<FreshSchedule> {
        match channel {
            FreshRetryChannel::SelfChannel => self.self_schedule.take(),
            FreshRetryChannel::Peer => self.peer_schedule.take(),
        }
    }

    pub(crate) fn schedules(&self) -> impl Iterator<Item = &FreshSchedule> {
        [self.self_schedule.as_ref(), self.peer_schedule.as_ref()]
            .into_iter()
            .flatten()
    }

    pub(crate) fn accounting(&self, channel: FreshRetryChannel) -> Option<&EpisodeAccounting> {
        match channel {
            FreshRetryChannel::SelfChannel => self.self_accounting.as_ref(),
            FreshRetryChannel::Peer => self.peer_accounting.as_ref(),
        }
    }

    pub(crate) fn accounting_mut(
        &mut self,
        channel: FreshRetryChannel,
    ) -> Option<&mut EpisodeAccounting> {
        match channel {
            FreshRetryChannel::SelfChannel => self.self_accounting.as_mut(),
            FreshRetryChannel::Peer => self.peer_accounting.as_mut(),
        }
    }

    pub(crate) fn set_accounting(
        &mut self,
        channel: FreshRetryChannel,
        accounting: Option<EpisodeAccounting>,
    ) {
        match channel {
            FreshRetryChannel::SelfChannel => self.self_accounting = accounting,
            FreshRetryChannel::Peer => self.peer_accounting = accounting,
        }
    }

    pub(crate) fn end_accounting(&mut self, channel: FreshRetryChannel) {
        let accounting = match channel {
            FreshRetryChannel::SelfChannel => self.self_accounting.take(),
            FreshRetryChannel::Peer => self.peer_accounting.take(),
        };
        if let Some(value) = accounting {
            self.set_ended_episode(
                channel,
                Some((value.target_identity, value.phase, value.episode_generation)),
            );
        }
    }

    pub(crate) fn ended_episode(
        &self,
        channel: FreshRetryChannel,
    ) -> Option<&(String, NativeQuietTailPhase, u64)> {
        match channel {
            FreshRetryChannel::SelfChannel => self.self_ended_episode.as_ref(),
            FreshRetryChannel::Peer => self.peer_ended_episode.as_ref(),
        }
    }

    pub(crate) fn set_ended_episode(
        &mut self,
        channel: FreshRetryChannel,
        episode: Option<(String, NativeQuietTailPhase, u64)>,
    ) {
        match channel {
            FreshRetryChannel::SelfChannel => self.self_ended_episode = episode,
            FreshRetryChannel::Peer => self.peer_ended_episode = episode,
        }
    }

    pub(crate) fn reconcile(
        &mut self,
        channel: FreshRetryChannel,
        intent: RetryIntent,
        now: Instant,
        policy: FreshRetryPolicy,
        stream_max_completed: u32,
        transfer_native_cause: bool,
    ) -> RetryReconcileResult {
        let current_schedule = self.schedule(channel).cloned();
        let current_accounting = self.accounting(channel).cloned();
        let generation_changed =
            intent.generation.is_some() && intent.generation != self.observed_generation(channel);
        let episode_changed = current_schedule.as_ref().is_some_and(|schedule| {
            intent.target_identity.as_ref() != Some(&schedule.target_identity)
                || intent.episode.as_ref().is_none_or(|episode| {
                    episode.generation != schedule.episode_generation
                        || episode.phase != schedule.phase
                })
        });
        self.observe_generation(channel, intent.generation);
        let mut result = RetryReconcileResult::default();
        if intent.target_identity.is_none()
            || intent.generation.is_none()
            || intent.episode.is_none()
        {
            self.end_accounting(channel);
            if let Some(cancelled) = self.take_schedule(channel) {
                result.transitions.push(RetryTransition {
                    schedule: cancelled,
                    outcome: "cancelled",
                });
            }
            return result;
        }
        if episode_changed && !generation_changed {
            if let Some(cancelled) = self.take_schedule(channel) {
                result.transitions.push(RetryTransition {
                    schedule: cancelled,
                    outcome: "cancelled",
                });
            }
            return result;
        }
        if !generation_changed {
            return result;
        }
        let episode = intent.episode.expect("checked episode");
        let target_identity = intent.target_identity.expect("checked target identity");
        if self.ended_episode(channel).is_some_and(|ended| {
            ended.0 == target_identity && ended.1 == episode.phase && ended.2 == episode.generation
        }) {
            return result;
        }
        self.set_ended_episode(channel, None);
        let same_episode = current_accounting.as_ref().is_some_and(|accounting| {
            accounting.target_identity == target_identity
                && accounting.episode_generation == episode.generation
                && accounting.phase == episode.phase
        });
        let completed = if same_episode {
            current_accounting
                .as_ref()
                .map_or(0, |value| value.completed)
        } else {
            0
        };
        let deadline = if same_episode {
            current_accounting
                .as_ref()
                .map_or(now, |value| value.deadline)
        } else {
            now + policy.deadline
        };
        let max_completed = if same_episode {
            current_accounting
                .as_ref()
                .map_or(policy.max_completed, |value| value.max_completed)
        } else {
            match episode.phase {
                NativeQuietTailPhase::Stream => stream_max_completed.min(policy.max_completed),
                NativeQuietTailPhase::Final => policy.max_completed,
            }
        };
        let next_due = if same_episode {
            current_schedule
                .as_ref()
                .map_or(now + policy.cadence, |value| value.next_due)
        } else {
            now + policy.cadence
        };
        let next = FreshSchedule {
            channel,
            trigger_generation: intent.generation.expect("checked generation"),
            required_scene_generation: intent.required_scene_generation,
            target_identity,
            completed,
            max_completed,
            phase: episode.phase,
            episode_generation: episode.generation,
            deadline,
            next_due,
        };
        if transfer_native_cause {
            if let Some(current) = current_schedule.clone() {
                result.cause_transfer = Some((current, next.clone()));
            }
        }
        self.set_accounting(
            channel,
            Some(EpisodeAccounting {
                target_identity: next.target_identity.clone(),
                phase: next.phase,
                episode_generation: next.episode_generation,
                completed,
                max_completed,
                deadline,
            }),
        );
        let disabled = max_completed == 0 || completed >= max_completed || now > deadline;
        let replaced = if disabled {
            self.take_schedule(channel)
        } else {
            self.replace_schedule(channel, next.clone())
        };
        if let Some(replaced) = replaced {
            result.transitions.push(RetryTransition {
                schedule: replaced,
                outcome: "replaced",
            });
        }
        result.transitions.push(RetryTransition {
            schedule: next,
            outcome: "scheduled",
        });
        result
    }

    pub(crate) fn set_all_next_due(&mut self, due: Instant) {
        for channel in [FreshRetryChannel::SelfChannel, FreshRetryChannel::Peer] {
            if let Some(schedule) = self.schedule_mut(channel) {
                schedule.next_due = due;
            }
        }
    }

    pub(crate) fn fail_matching(&mut self, captured: &FreshSchedule) -> Option<FreshSchedule> {
        self.set_accounting(captured.channel, None);
        self.take_schedule(captured.channel)
            .filter(|active| active.same_intent(captured))
    }

    pub(crate) fn defer_cached_rehandoff(
        &mut self,
        captured: &FreshSchedule,
        now: Instant,
        cadence: Duration,
    ) -> Option<FreshSchedule> {
        let active = self
            .schedule_mut(captured.channel)
            .filter(|active| active.same_intent(captured))?;
        active.next_due = (now + cadence).min(active.deadline + Duration::from_nanos(1));
        Some(active.clone())
    }

    pub(crate) fn matching_schedule(&self, captured: &FreshSchedule) -> Option<FreshSchedule> {
        self.schedule(captured.channel)
            .filter(|active| active.same_intent(captured))
            .cloned()
    }

    pub(crate) fn complete_matching(
        &mut self,
        captured: &FreshSchedule,
        now: Instant,
        cadence: Duration,
    ) -> Option<FreshSchedule> {
        let active = self.schedule_mut(captured.channel).filter(|active| {
            active.same_intent(captured) || active.accepts_transferred_due_from(captured)
        })?;
        active.completed += 1;
        active.next_due = now + cadence;
        let completed = active.clone();
        if active.completed >= active.max_completed || now > active.deadline {
            self.take_schedule(captured.channel);
        }
        if let Some(accounting) = self.accounting_mut(captured.channel).filter(|accounting| {
            accounting.target_identity == completed.target_identity
                && accounting.phase == completed.phase
                && accounting.episode_generation == completed.episode_generation
        }) {
            accounting.completed = completed.completed;
        }
        Some(completed)
    }

    pub(crate) fn record(&mut self, schedule: &FreshSchedule, outcome: &'static str) {
        if self.audit.len() == RETRY_AUDIT_CAPACITY {
            self.audit.pop_front();
            self.audit_dropped = self.audit_dropped.saturating_add(1);
        }
        self.audit.push_back(FreshAuditFact {
            channel: schedule.channel,
            trigger_generation: schedule.trigger_generation,
            outcome,
            completed: schedule.completed,
            at: self.audit_started_at.elapsed(),
        });
    }

    pub(crate) fn audit(&self) -> Vec<(&'static str, u64, &'static str, u32, Duration)> {
        self.audit
            .iter()
            .map(|fact| {
                (
                    fact.channel.name(),
                    fact.trigger_generation,
                    fact.outcome,
                    fact.completed,
                    fact.at,
                )
            })
            .collect()
    }

    pub(crate) fn audit_dropped(&self) -> u64 {
        self.audit_dropped
    }

    pub(crate) fn audit_len(&self) -> usize {
        self.audit.len()
    }

    pub(crate) fn clear(&mut self) -> Vec<FreshSchedule> {
        let schedules = [self.self_schedule.take(), self.peer_schedule.take()]
            .into_iter()
            .flatten()
            .collect();
        self.self_accounting = None;
        self.peer_accounting = None;
        self.self_ended_episode = None;
        self.peer_ended_episode = None;
        schedules
    }
}
