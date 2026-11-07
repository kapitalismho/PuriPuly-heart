from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from copy import deepcopy

import pytest

from puripuly_heart.release_evidence.agent_chat_simulator import execution
from puripuly_heart.release_evidence.agent_chat_simulator.execution import (
    AttemptJob,
    AttemptScheduler,
)
from puripuly_heart.release_evidence.agent_chat_simulator.runner import (
    ControlledModels,
    controlled_transport,
    run_attempt,
)
from puripuly_heart.release_evidence.agent_chat_simulator.scenarios import select_scenarios


async def reached(event):
    await asyncio.wait_for(event.wait(), timeout=5)


async def collect_checkpoint(job, result):
    return None


class Session:
    def __init__(self, harness, identity):
        self.harness = harness
        self.identity = identity
        self.prepared = False
        self.closed = False
        self.calls = Counter()
        self._attempts = []

    @property
    def attempts(self):
        return deepcopy(self._attempts)

    async def prepare(self):
        self.prepared = True

    async def complete(self, *, role, instructions, payload, schema):
        self.calls[role] += 1
        key = (self.identity, role, self.calls[role])
        self.harness.active[role] += 1
        self.harness.peak[role] = max(self.harness.peak[role], self.harness.active[role])
        self.harness.entered[key].set()
        try:
            await self.harness.allowed[key].wait()
            if key in self.harness.call_errors:
                raise self.harness.call_errors[key]
            evidence = {
                "role": role,
                "session": self.identity,
                "usage": {"input_tokens": 7, "output_tokens": 3},
            }
            self._attempts.append(deepcopy(evidence))
            return {"value": {"ok": True}, **evidence}
        finally:
            self.harness.active[role] -= 1

    async def close(self):
        self.closed = True
        self.harness.owned -= 1
        self.harness.closed[self.identity].set()
        if self.identity in self.harness.close_errors:
            raise self.harness.close_errors[self.identity]


class Harness:
    def __init__(self):
        self.sessions = {}
        self.entered = defaultdict(asyncio.Event)
        self.allowed = defaultdict(asyncio.Event)
        self.dialogue_started = defaultdict(asyncio.Event)
        self.judge_requested = defaultdict(asyncio.Event)
        self.closed = defaultdict(asyncio.Event)
        self.active = Counter()
        self.peak = Counter()
        self.group_active = Counter()
        self.group_peak = Counter()
        self.starts = []
        self.owned = 0
        self.peak_owned = 0
        self.call_errors = {}
        self.close_errors = {}
        self.checkpoints = []

    def factory(self, job):
        assert job.identity not in self.sessions
        session = Session(self, job.identity)
        self.sessions[job.identity] = session
        self.owned += 1
        self.peak_owned = max(self.peak_owned, self.owned)
        return session

    def job(self, identity, group="shared", *, judge_calls=1):
        async def execute(models):
            await models.prepare()
            self.starts.append(identity)
            self.group_active[group] += 1
            self.group_peak[group] = max(self.group_peak[group], self.group_active[group])
            self.dialogue_started[identity].set()
            try:
                await models.complete(role="user", instructions="user", payload={}, schema={})
            finally:
                self.group_active[group] -= 1
            for number in range(1, judge_calls + 1):
                self.judge_requested[(identity, number)].set()
                await models.complete(role="judge", instructions="judge", payload={}, schema={})
            return {"id": identity, "access_attempts": models.attempts}

        return AttemptJob(identity=identity, group=group, execute=execute)

    async def checkpoint(self, job, result):
        assert result["id"] == job.identity
        assert self.sessions[job.identity].closed
        assert "execution_timing" in result
        self.checkpoints.append(deepcopy(result))

    def allow(self, identity, role, number=1):
        self.allowed[(identity, role, number)].set()

    def allow_jobs(self, identities, *, judge_calls=1):
        for identity in identities:
            self.allow(identity, "user")
            for number in range(1, judge_calls + 1):
                self.allow(identity, "judge", number)


@pytest.mark.asyncio
async def test_frozen_run_attempt_uses_isolated_sessions_and_overlaps_judgment():
    judge_entered = {identity: asyncio.Event() for identity in ("first", "second")}
    finish_judges = asyncio.Event()
    sessions = {}
    checkpoints = []

    class ControlledSession(ControlledModels):
        def __init__(self, identity):
            self.identity = identity
            self._attempts = []
            self.closed = False

        @property
        def attempts(self):
            return deepcopy(self._attempts)

        async def complete(self, **request):
            if request["role"] == "judge":
                judge_entered[self.identity].set()
                await finish_judges.wait()
            response = await super().complete(**request)
            response["session"] = self.identity
            self._attempts.append({"role": request["role"], "session": self.identity})
            return response

        async def close(self):
            self.closed = True

    def factory(job):
        session = ControlledSession(job.identity)
        sessions[job.identity] = session
        return session

    jobs = []
    for identity, scenario_id in (
        ("first", "debug-translation-auth"),
        ("second", "extra-recovered-auth-overlay"),
    ):
        case = select_scenarios([scenario_id])[0]

        async def execute(models, case=case, identity=identity):
            return await run_attempt(
                case, models, controlled_transport, identity=identity, controlled=True
            )

        jobs.append(AttemptJob(identity, "one-arm", execute))

    async def checkpoint(job, result):
        assert sessions[job.identity].closed
        checkpoints.append(deepcopy(result))

    scheduler = AttemptScheduler(
        factory, dialogue_concurrency=1, judge_concurrency=2, group_concurrency={"one-arm": 1}
    )
    running = asyncio.create_task(scheduler.run(jobs, checkpoint=checkpoint))
    try:
        await reached(judge_entered["first"])
        await reached(judge_entered["second"])
        assert not running.done()
        finish_judges.set()
        results = await running
    finally:
        finish_judges.set()
        await asyncio.gather(running, return_exceptions=True)
    assert [result["id"] for result in results] == ["first", "second"]
    assert all(result["status"] == "passed" for result in results)
    assert len(checkpoints) == 2
    assert all(session.closed for session in sessions.values())
    for result in results:
        assert {call["session"] for call in result["access_attempts"]} == {result["id"]}
        assert {call["session"] for call in result["model_calls"]} == {result["id"]}
    results[0]["access_attempts"][0]["session"] = "changed"
    assert sessions["first"].attempts[0]["session"] == "first"
    assert all(call["session"] == "second" for call in results[1]["access_attempts"])


@pytest.mark.asyncio
async def test_dialogue_group_judge_and_total_admission_bounds_with_fair_groups():
    harness = Harness()
    identities = ["a1", "a2", "a3", "b1", "b2", "b3"]
    jobs = [harness.job(identity, identity[0]) for identity in identities]
    scheduler = AttemptScheduler(
        harness.factory,
        dialogue_concurrency=2,
        judge_concurrency=1,
        group_concurrency={"a": 1, "b": 1},
    )
    running = asyncio.create_task(scheduler.run(jobs, checkpoint=harness.checkpoint))
    try:
        await reached(harness.entered[("a1", "user", 1)])
        await reached(harness.entered[("b1", "user", 1)])
        assert harness.starts == ["a1", "b1"]
        harness.allow("a1", "user")
        await reached(harness.entered[("a1", "judge", 1)])
        await reached(harness.entered[("a2", "user", 1)])
        harness.allow("b1", "user")
        harness.allow("a2", "user")
        await reached(harness.judge_requested[("b1", 1)])
        await reached(harness.judge_requested[("a2", 1)])
        assert harness.starts == ["a1", "b1", "a2"]
        harness.allow_jobs(identities)
        results = await running
    finally:
        harness.allow_jobs(identities)
        await asyncio.gather(running, return_exceptions=True)
    assert [result["id"] for result in results] == identities
    assert harness.peak["user"] == 2
    assert harness.peak["judge"] == 1
    assert harness.group_peak == {"a": 1, "b": 1}
    assert harness.peak_owned == 3
    assert harness.owned == 0
    assert len(harness.checkpoints) == len(identities)


@pytest.mark.asyncio
async def test_rejudgment_stays_in_judge_pool_without_reclaiming_dialogue_capacity():
    harness = Harness()
    identities = ["first", "second", "third"]
    jobs = [harness.job(identity, judge_calls=2) for identity in identities]
    scheduler = AttemptScheduler(harness.factory, dialogue_concurrency=1, judge_concurrency=1)
    running = asyncio.create_task(scheduler.run(jobs, checkpoint=harness.checkpoint))
    try:
        await reached(harness.entered[("first", "user", 1)])
        harness.allow("first", "user")
        await reached(harness.entered[("first", "judge", 1)])
        await reached(harness.entered[("second", "user", 1)])
        harness.allow("first", "judge", 1)
        await reached(harness.entered[("first", "judge", 2)])
        assert harness.active == {"user": 1, "judge": 1}
        harness.allow_jobs(identities, judge_calls=2)
        results = await running
    finally:
        harness.allow_jobs(identities, judge_calls=2)
        await asyncio.gather(running, return_exceptions=True)
    assert len(results) == 3
    assert harness.peak["user"] == harness.peak["judge"] == 1
    assert all(
        [call["role"] for call in result["execution_timing"]["calls"]] == ["user", "judge", "judge"]
        for result in results
    )
    assert all(session.closed for session in harness.sessions.values())


@pytest.mark.asyncio
async def test_local_queue_and_completion_timings_are_separate_from_native_usage(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(execution, "perf_counter", lambda: clock[0])
    harness = Harness()
    jobs = [harness.job(identity) for identity in ("first", "second")]
    running = asyncio.create_task(
        AttemptScheduler(harness.factory, dialogue_concurrency=1, judge_concurrency=1).run(
            jobs, checkpoint=harness.checkpoint
        )
    )
    try:
        await reached(harness.entered[("first", "user", 1)])
        clock[0] = 102.0
        harness.allow("first", "user")
        await reached(harness.entered[("first", "judge", 1)])
        await reached(harness.entered[("second", "user", 1)])
        clock[0] = 105.0
        harness.allow("second", "user")
        await reached(harness.judge_requested[("second", 1)])
        clock[0] = 111.0
        harness.allow("first", "judge")
        await reached(harness.entered[("second", "judge", 1)])
        clock[0] = 118.0
        harness.allow("second", "judge")
        results = await running
    finally:
        harness.allow_jobs(("first", "second"))
        await asyncio.gather(running, return_exceptions=True)
    first, second = [result["execution_timing"] for result in results]
    assert first["dialogue_seconds"] == 2.0
    assert first["judgment_seconds"] == 9.0
    assert first["total_seconds"] == 11.0
    assert second["dialogue_seconds"] == 3.0
    assert second["judgment_seconds"] == 13.0
    assert second["total_seconds"] == 16.0
    assert second["calls"][0]["queue_wait_seconds"] == 0.0
    assert second["calls"][0]["completion_seconds"] == 3.0
    assert second["calls"][1]["queue_wait_seconds"] == 6.0
    assert second["calls"][1]["completion_seconds"] == 7.0
    assert second["judgment_started_seconds"] == second["dialogue_finished_seconds"]
    assert second["judgment_finished_seconds"] <= second["execute_finished_seconds"]
    assert all(call["status"] == "completed" for call in second["calls"])
    assert "not native provider or first-token" in second["basis"]
    assert all(
        call["usage"] == {"input_tokens": 7, "output_tokens": 3}
        for result in results
        for call in result["access_attempts"]
    )
    assert harness.checkpoints == results


@pytest.mark.asyncio
async def test_initial_stop_does_not_construct_execute_or_fabricate_results():
    harness = Harness()
    results = await AttemptScheduler(harness.factory).run(
        [harness.job("unstarted")], checkpoint=harness.checkpoint, stop_requested=lambda: True
    )
    assert results == []
    assert harness.sessions == {}
    assert harness.starts == []
    assert harness.checkpoints == []


@pytest.mark.asyncio
async def test_stop_drains_started_work_and_plan_difference_resumes_only_unstarted():
    harness = Harness()
    jobs = [harness.job(identity) for identity in ("first", "second", "third")]
    stopped = [False]
    scheduler = AttemptScheduler(harness.factory, dialogue_concurrency=1, judge_concurrency=1)
    running = asyncio.create_task(
        scheduler.run(jobs, checkpoint=harness.checkpoint, stop_requested=lambda: stopped[0])
    )
    try:
        await reached(harness.entered[("first", "user", 1)])
        stopped[0] = True
        harness.allow("first", "user")
        await reached(harness.entered[("first", "judge", 1)])
        assert not running.done()
        harness.allow("first", "judge")
        results = await running
    finally:
        harness.allow_jobs(("first", "second", "third"))
        await asyncio.gather(running, return_exceptions=True)
    assert [result["id"] for result in results] == ["first"]
    assert harness.starts == ["first"]
    assert list(harness.sessions) == ["first"]
    assert harness.sessions["first"].closed
    completed = {result["id"] for result in harness.checkpoints}
    pending = [job for job in jobs if job.identity not in completed]
    stopped[0] = False
    resumed = await scheduler.run(
        pending, checkpoint=harness.checkpoint, stop_requested=lambda: stopped[0]
    )
    assert [result["id"] for result in resumed] == ["second", "third"]
    assert harness.starts == ["first", "second", "third"]
    assert {result["id"] for result in harness.checkpoints} == {job.identity for job in jobs}
    assert harness.owned == 0


@pytest.mark.asyncio
async def test_stop_between_model_creation_and_claim_closes_without_executing():
    harness = Harness()
    stopped = [False]

    def factory(job):
        models = harness.factory(job)
        stopped[0] = True
        return models

    results = await AttemptScheduler(factory).run(
        [harness.job("not-claimed")],
        checkpoint=harness.checkpoint,
        stop_requested=lambda: stopped[0],
    )
    assert results == []
    assert harness.starts == []
    assert harness.checkpoints == []
    assert harness.sessions["not-claimed"].closed


@pytest.mark.asyncio
async def test_cancellation_drains_and_checkpoints_without_starting_pending_work():
    harness = Harness()
    jobs = [harness.job(identity) for identity in ("started", "pending")]
    running = asyncio.create_task(
        AttemptScheduler(harness.factory, dialogue_concurrency=1, judge_concurrency=1).run(
            jobs, checkpoint=harness.checkpoint
        )
    )
    try:
        await reached(harness.entered[("started", "user", 1)])
        running.cancel()
        await asyncio.sleep(0)
        assert not running.done()
        harness.allow("started", "user")
        await reached(harness.entered[("started", "judge", 1)])
        assert not running.done()
        harness.allow("started", "judge")
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        harness.allow_jobs(("started", "pending"))
        await asyncio.gather(running, return_exceptions=True)
    assert harness.starts == ["started"]
    assert [result["id"] for result in harness.checkpoints] == ["started"]
    assert harness.owned == 0
    assert harness.active == {"user": 0, "judge": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["completion", "factory"])
async def test_failure_stops_admission_drains_siblings_and_propagates(source):
    harness = Harness()
    failure = RuntimeError("owned execution failed")
    if source == "completion":
        harness.call_errors[("bad", "user", 1)] = failure

    def factory(job):
        if job.identity == "bad" and source == "factory":
            raise failure
        return harness.factory(job)

    jobs = [harness.job(identity) for identity in ("good", "bad", "pending")]
    running = asyncio.create_task(
        AttemptScheduler(factory, dialogue_concurrency=2, judge_concurrency=1).run(
            jobs, checkpoint=harness.checkpoint
        )
    )
    try:
        await reached(harness.entered[("good", "user", 1)])
        if source == "completion":
            await reached(harness.entered[("bad", "user", 1)])
            harness.allow("bad", "user")
            await reached(harness.closed["bad"])
        harness.allow("good", "user")
        await reached(harness.entered[("good", "judge", 1)])
        assert not running.done()
        harness.allow("good", "judge")
        with pytest.raises(RuntimeError) as raised:
            await running
        assert raised.value is failure
    finally:
        harness.allow_jobs(("good", "bad", "pending"))
        await asyncio.gather(running, return_exceptions=True)
    assert "pending" not in harness.starts
    assert [result["id"] for result in harness.checkpoints] == ["good"]
    assert harness.owned == 0
    assert all(session.closed for session in harness.sessions.values())


@pytest.mark.asyncio
async def test_completed_result_is_checkpointed_even_when_close_and_checkpoint_fail():
    harness = Harness()
    close_error = RuntimeError("close failed")
    checkpoint_error = OSError("durable write failed")
    harness.close_errors["completed"] = close_error
    observed = []

    async def execute(models):
        return {"id": "completed", "status": "passed"}

    async def checkpoint(job, result):
        observed.append(deepcopy(result))
        assert harness.sessions[job.identity].closed
        raise checkpoint_error

    with pytest.raises(ExceptionGroup) as raised:
        await AttemptScheduler(harness.factory).run(
            [AttemptJob("completed", "arm", execute)], checkpoint=checkpoint
        )
    assert raised.value.exceptions == (close_error, checkpoint_error)
    assert observed[0]["status"] == "passed"
    assert observed[0]["execution_timing"]["judgment_started_seconds"] is None
    assert observed[0]["execution_timing"]["calls"] == []
    assert harness.owned == 0


@pytest.mark.asyncio
async def test_checkpoint_completion_is_awaited_and_serialized():
    harness = Harness()
    checkpoint_entered = asyncio.Event()
    checkpoint_allowed = asyncio.Event()
    active = 0
    peak = 0
    saved = []

    async def execute(models):
        return {"status": "passed"}

    async def checkpoint(job, result):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        checkpoint_entered.set()
        try:
            await checkpoint_allowed.wait()
            await asyncio.sleep(0)
            saved.append(job.identity)
        finally:
            active -= 1

    jobs = [AttemptJob(identity, "arm", execute) for identity in ("first", "second")]
    running = asyncio.create_task(
        AttemptScheduler(harness.factory).run(jobs, checkpoint=checkpoint)
    )
    try:
        await reached(checkpoint_entered)
        assert not running.done()
        checkpoint_allowed.set()
        results = await running
    finally:
        checkpoint_allowed.set()
        await asyncio.gather(running, return_exceptions=True)
    assert len(results) == 2
    assert saved == ["first", "second"]
    assert peak == 1
    assert harness.owned == 0


@pytest.mark.asyncio
async def test_models_without_session_resources_remain_compatible():
    async def execute(models):
        return await run_attempt(
            select_scenarios(["debug-translation-auth"])[0],
            models,
            controlled_transport,
            identity="controlled",
            controlled=True,
        )

    results = await AttemptScheduler(lambda job: ControlledModels()).run(
        [AttemptJob("controlled", "arm", execute)], checkpoint=collect_checkpoint
    )
    assert results[0]["status"] == "passed"
    assert results[0]["access_attempts"] == []


@pytest.mark.parametrize(
    "limits",
    [
        {"dialogue_concurrency": 0},
        {"judge_concurrency": -1},
        {"dialogue_concurrency": True},
        {"group_concurrency": {"arm": 0}},
    ],
)
def test_invalid_limits_are_rejected_before_execution(limits):
    with pytest.raises(ValueError, match="positive integers"):
        AttemptScheduler(lambda job: ControlledModels(), **limits)


@pytest.mark.asyncio
async def test_duplicate_identity_is_rejected_without_claiming_either_job():
    harness = Harness()
    with pytest.raises(ValueError, match="Duplicate attempt identity"):
        await AttemptScheduler(harness.factory).run(
            [harness.job("duplicate", "a"), harness.job("duplicate", "b")],
            checkpoint=harness.checkpoint,
        )
    assert harness.sessions == {}
    assert harness.starts == []


@pytest.mark.asyncio
async def test_caught_model_failure_keeps_failed_call_timing_and_completed_evidence():
    harness = Harness()
    failure = RuntimeError("model call failed")
    harness.call_errors[("blocked", "user", 1)] = failure
    harness.allow("blocked", "user")

    async def execute(models):
        try:
            await models.complete(role="user", instructions="user", payload={}, schema={})
        except RuntimeError as error:
            assert error is failure
            return {"id": "blocked", "status": "blocked", "error": str(error)}
        raise AssertionError("The model failure must remain visible to the attempt")

    results = await AttemptScheduler(harness.factory).run(
        [AttemptJob("blocked", "arm", execute)], checkpoint=harness.checkpoint
    )
    assert results[0]["status"] == "blocked"
    timing = results[0]["execution_timing"]
    assert timing["judgment_started_seconds"] is None
    assert timing["dialogue_finished_seconds"] == timing["execute_finished_seconds"]
    assert timing["calls"][0]["status"] == "failed"
    assert timing["calls"][0]["error_type"] == "RuntimeError"
    assert timing["calls"][0]["completion_seconds"] >= 0
    assert harness.checkpoints == results
    assert harness.owned == 0


@pytest.mark.asyncio
async def test_checkpoint_failure_drains_completed_sibling_without_admitting_pending_work():
    harness = Harness()
    first_checkpoint = asyncio.Event()
    fail_checkpoint = asyncio.Event()
    failure = OSError("checkpoint unavailable")
    saved = []

    async def checkpoint(job, result):
        if job.identity == "first":
            first_checkpoint.set()
            await fail_checkpoint.wait()
            raise failure
        saved.append(deepcopy(result))

    def job(identity):
        async def execute(models):
            harness.starts.append(identity)
            return {"id": identity, "status": "passed"}

        return AttemptJob(identity, "arm", execute)

    running = asyncio.create_task(
        AttemptScheduler(harness.factory, dialogue_concurrency=1, judge_concurrency=1).run(
            [job(identity) for identity in ("first", "second", "pending")],
            checkpoint=checkpoint,
        )
    )
    try:
        await reached(first_checkpoint)
        await reached(harness.closed["second"])
        assert harness.starts == ["first", "second"]
        assert not running.done()
        fail_checkpoint.set()
        with pytest.raises(OSError) as raised:
            await running
        assert raised.value is failure
    finally:
        fail_checkpoint.set()
        await asyncio.gather(running, return_exceptions=True)
    assert [result["id"] for result in saved] == ["second"]
    assert harness.starts == ["first", "second"]
    assert harness.owned == 0
