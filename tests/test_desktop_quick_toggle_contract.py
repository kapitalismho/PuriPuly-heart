import asyncio
import json
from uuid import uuid4

from puripuly_heart.app.services.quick_translate import FRAME_PREFIX, QuickTranslateOwner
from puripuly_heart.ui.quick_window_layout import clamp_window_bounds


class Process:
    def __init__(self):
        self.returncode = None
        self.stdout = asyncio.StreamReader()
        self.stdin = self
        self.messages = []
        self.done = asyncio.Event()

    def write(self, data):
        message = json.loads(data)
        self.messages.append(message)
        if message["type"] == "exit":
            self.terminate()

    async def drain(self):
        await asyncio.sleep(0)

    def emit(self, message):
        self.stdout.feed_data((FRAME_PREFIX + json.dumps(message) + "\n").encode())

    def terminate(self):
        self.returncode = 0
        self.stdout.feed_eof()
        self.done.set()

    kill = terminate

    async def wait(self):
        await self.done.wait()
        return self.returncode


def make_owner(submit=None):
    processes = []

    async def spawn():
        process = Process()
        processes.append(process)
        return process

    async def save_preferences(values):
        return None

    async def default_submit(text):
        return "utterance", "translated"

    owner = QuickTranslateOwner(
        spawn=spawn,
        submit=submit or default_submit,
        preferences=lambda: {"width": 420, "height": 680},
        save_preferences=save_preferences,
    )
    return owner, processes


async def eventually(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("Observable state was not reached")


def test_toggle_reuses_one_child():
    async def scenario():
        owner, processes = make_owner()
        try:
            await owner.open()
            await owner.open()
            await owner.open()
            assert len(processes) == 1
            assert [m["type"] for m in processes[0].messages] == ["show", "hide", "show"]
            assert owner.visible is True
        finally:
            await owner.close()
        assert processes[0].returncode == 0
        assert owner.reader_task.done()
    asyncio.run(scenario())


def test_child_hide_allows_next_click_to_show():
    async def scenario():
        owner, processes = make_owner()
        try:
            await owner.open()
            process = processes[0]
            process.emit(dict(type="visibility", visible=False, visibility_revision=1))
            await eventually(lambda: owner.visible is False)
            await owner.open()
            assert process.messages[-1]["type"] == "show"
            assert len(processes) == 1
        finally:
            await owner.close()
    asyncio.run(scenario())


def test_stale_hide_ack_does_not_override_reopen():
    async def scenario():
        owner, processes = make_owner()
        try:
            for _ in range(3):
                await owner.open()
            processes[0].emit(dict(type="visibility", visible=False, visibility_revision=2))
            for _ in range(3):
                await asyncio.sleep(0)
            assert owner.visible is True
            await owner.open()
            assert processes[0].messages[-1]["type"] == "hide"
        finally:
            await owner.close()
    asyncio.run(scenario())


def test_hidden_window_still_receives_admitted_result():
    async def scenario():
        release = asyncio.Event()
        calls = []

        async def submit(text):
            calls.append(text)
            await release.wait()
            return "utterance", "translated"

        owner, processes = make_owner(submit)
        try:
            await owner.open()
            process = processes[0]
            identity = str(uuid4())
            process.emit(dict(type="submit", id=identity, text="hello"))
            await eventually(lambda: calls == ["hello"])
            await owner.open()
            release.set()
            await eventually(lambda: any(m["type"] == "result" for m in process.messages))
            result = next(m for m in process.messages if m["type"] == "result")
            assert result["id"] == identity
            assert result["translation"] == "translated"
            assert owner.visible is False
            assert calls == ["hello"]
        finally:
            await owner.close()
    asyncio.run(scenario())


def test_concurrent_clicks_do_not_spawn_duplicate_children():
    async def scenario():
        owner, processes = make_owner()
        try:
            await asyncio.gather(owner.open(), owner.open())
            assert len(processes) == 1
            assert [m["type"] for m in processes[0].messages] == ["show", "hide"]
        finally:
            await owner.close()
    asyncio.run(scenario())


def test_closed_owner_cannot_reopen():
    async def scenario():
        owner, processes = make_owner()
        await owner.close()
        await owner.open()
        assert not processes
    asyncio.run(scenario())


def test_exited_child_is_replaced_and_shown():
    async def scenario():
        owner, processes = make_owner()
        try:
            await owner.open()
            processes[0].terminate()
            await owner.reader_task
            await owner.open()
            assert len(processes) == 2
            assert processes[1].messages[0]["type"] == "show"
            assert owner.visible is True
        finally:
            await owner.close()
    asyncio.run(scenario())


def test_default_bounds_are_portrait():
    bounds = clamp_window_bounds({})
    assert (bounds["width"], bounds["height"]) == (420, 680)


def test_legacy_default_bounds_are_migrated():
    bounds = clamp_window_bounds({"width": 480, "height": 520})
    assert (bounds["width"], bounds["height"]) == (420, 680)


def test_custom_size_is_preserved():
    bounds = clamp_window_bounds({"width": 700, "height": 550})
    assert (bounds["width"], bounds["height"]) == (700, 550)


def test_offscreen_position_is_clamped():
    bounds = clamp_window_bounds({"left": 9999, "top": -9999}, work_area=(-1280, 0, 0, 720))
    assert bounds["left"] == -420
    assert bounds["top"] == 0


def test_geometry_does_not_exceed_small_work_area():
    bounds = clamp_window_bounds({}, work_area=(100, 200, 350, 450))
    assert bounds == {"width": 250, "height": 250, "left": 100, "top": 200}
