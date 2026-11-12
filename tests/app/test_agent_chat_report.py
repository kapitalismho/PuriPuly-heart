from __future__ import annotations

import asyncio
import inspect
import io
import json
import re
import threading

import httpx
import pytest

from puripuly_heart.app.adapters.agent_relay_transport import AgentRelayTransport
from puripuly_heart.app.ports.agent_chat import AgentServiceError, AgentTransportError
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.app.services.agent_chat.report import build_support_bundle
from puripuly_heart.core.diagnostic_codes import DOMAIN_CODES
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.ui.views.agent_chat import AgentChatHooks, AgentChatView
from puripuly_heart.ui.views.agent_chat import _texts as agent_chat_texts
from tests.app.test_agent_chat import Records, Responses, Tokens, answer, context_of, tool


def proposal(summary="The supported next steps did not fix the synthetic failure."):
    return {**tool(), "name": "propose_report", "arguments": json.dumps({"summary": summary})}


def contents(bundle):
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == ["meta.json", "puri_chat.json", "runtime.log"]
        assert all(member.isfile() for member in members)
        return {member.name: archive.extractfile(member).read() for member in members}


class ReportResponses(Responses):
    def __init__(self, *responses, upload=None):
        super().__init__(*responses)
        self.uploads = []
        self.upload = upload

    async def upload_support(self, **request):
        self.uploads.append(request)
        if self.upload is not None:
            return await self.upload(request)
        return {"ticket": request["ticket"]}


def owner_for(transport, snapshot=lambda: b"synthetic runtime log\n"):
    return AgentChatOwner(
        Records(), transport, localize=lambda key: key, support_log_snapshot=snapshot
    )


async def make_offer(owner):
    await owner.start()
    result = await owner.send("Please contact the developer about this app failure.")
    assert result.status == "completed"
    (offer,) = owner.snapshot.report_offers
    return offer


@pytest.mark.asyncio
async def test_inline_consent_sends_complete_archive_and_static_receipt(tmp_path, monkeypatch):
    current = tmp_path / "current.log"
    current.write_bytes(b"recognized synthetic sentence\ntranslated synthetic sentence\n")
    (tmp_path / "backup.log").write_bytes(b"BACKUP-MUST-NOT-BE-SENT")
    log_reads = []

    def read_log():
        log_reads.append(threading.get_ident())
        return current.read_bytes()

    transport = ReportResponses(
        answer("Earlier conversation."),
        {
            "status": "completed",
            "output": [
                {"type": "reasoning", "encrypted_content": "REASONING-NOT-ARCHIVED"},
                proposal(),
            ],
        },
        answer("Send the evidence to the developer if you want."),
    )
    owner = owner_for(transport, read_log)
    await owner.start()
    tasks = []

    def run_task(factory):
        assert inspect.iscoroutinefunction(factory)
        task = asyncio.get_running_loop().create_task(factory())
        tasks.append(task)
        return task

    view = AgentChatView(owner, AgentChatHooks(run_task=run_task))
    view.did_mount()
    opened = []
    monkeypatch.setattr("puripuly_heart.ui.views.agent_chat.webbrowser.open", opened.append)
    try:
        long_message = "Earlier app conversation. " * 4000
        await owner.send(long_message)
        owner.archive_support_report("[app notice] synthetic earlier notice")
        owner.new_conversation()
        await owner.send("Please contact the developer.")
        (offer,) = owner.snapshot.report_offers
        assert offer.ticket is None and offer.status == "offered"
        assert owner.snapshot.messages[offer.message_index].role == "assistant"
        assert not log_reads and not transport.uploads
        assert len(transport.classifications) == 2 and len(transport.requests) == 3
        card = view._switcher.content.content.content.controls[1]
        assert card.on_click is not None
        before = (len(transport.classifications), len(transport.requests))
        card.on_click(None)
        await asyncio.gather(*tasks)
        (sent,) = owner.snapshot.report_offers
        assert sent.status == "sent" and re.fullmatch(r"PP-[0-9A-F]{16}", sent.ticket)
        assert len(log_reads) == 1 and log_reads[0] != threading.get_ident()
        assert len(transport.uploads) == 1
        payload = contents(transport.uploads[0]["bundle"])
        meta = json.loads(payload["meta.json"])
        assert set(meta) == {
            "ticket",
            "app_version",
            "os",
            "agent_protocol",
            "summary",
            "diagnostic_codes",
        }
        assert meta["ticket"] == sent.ticket and meta["agent_protocol"] == 3
        assert meta["app_version"] == transport.uploads[0]["app_version"]
        assert meta["diagnostic_codes"] == list(transport.uploads[0]["diagnostic_codes"])
        archived = json.loads(payload["puri_chat.json"])
        assert {"role": "user", "content": long_message} in archived
        assert any(item.get("type") == "conversation_separator" for item in archived)
        assert any(item.get("type") == "function_call" for item in archived)
        assert any(item.get("type") == "function_call_output" for item in archived)
        assert any(item.get("type") == "support_report" for item in archived)
        assert "REASONING-NOT-ARCHIVED" not in payload["puri_chat.json"].decode()
        assert not any(item.get("type") == "configuration_update" for item in archived)
        assert payload["runtime.log"] == current.read_bytes()
        notice = f"[app notice] support_report sent ticket={sent.ticket}"
        assert owner.report_archive[-1] == {
            "type": "support_report",
            "role": "system",
            "content": notice,
        }
        assert not any(item.get("content") == notice for item in owner._items)
        assert before == (len(transport.classifications), len(transport.requests))
        receipt = view._switcher.content.content.content.controls[1]
        assert sent.ticket in [text.value for text in agent_chat_texts(receipt)]
        receipt.on_click(None)
        assert opened == ["https://x.com/kapitalismho"]
        assert await owner.send_report(sent.id) == {"status": "sent", "ticket": sent.ticket}
        assert len(transport.uploads) == 1
    finally:
        view.will_unmount()
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", (False, True))
@pytest.mark.parametrize("reset", (False, True))
async def test_next_user_request_distinguishes_sending_and_completed_support_receipts(
    failed, reset
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def upload(request):
        entered.set()
        await release.wait()
        if failed:
            raise AgentTransportError("support_upload_failed")
        return {"ticket": request["ticket"]}

    transport = ReportResponses(
        {"status": "completed", "output": [proposal("PRIVATE-SUPPORT-SUMMARY")]},
        answer("A report can be sent with the button."),
        answer("The upload is still in progress."),
        answer("Current app evidence."),
        upload=upload,
    )
    owner = owner_for(transport, lambda: b"PRIVATE-UPLOADED-RUNTIME-LOG")
    sending = None
    try:
        offer = await make_offer(owner)
        calls = len(transport.classifications), len(transport.requests)
        sending = asyncio.create_task(owner.send_report(offer.id))
        await asyncio.wait_for(entered.wait(), 2)
        assert calls == (len(transport.classifications), len(transport.requests))
        assert (await owner.send("Has the upload completed yet?")).status == "completed"
        pending = context_of(transport.requests[-1])["core"]["support_reports"]
        assert pending == [
            {"status": "sending", "offer_id": offer.id, "displayed": False, "clicked": True}
        ]
        if reset:
            owner.new_conversation()
        calls = len(transport.classifications), len(transport.requests)
        release.set()
        result = await sending
        assert result["status"] == ("failed" if failed else "sent")
        assert calls == (len(transport.classifications), len(transport.requests))
        assert (await owner.send("What actually happened to the report?")).status == "completed"
        context = context_of(transport.requests[-1])
        expected = (
            []
            if reset
            else (
                [{"status": "failed", "offer_id": offer.id, "displayed": False, "clicked": True}]
                if failed
                else [
                    {
                        "status": "sent",
                        "ticket": result["ticket"],
                        "offer_id": offer.id,
                        "displayed": False,
                        "clicked": True,
                    }
                ]
            )
        )
        assert context["core"]["support_reports"] == expected
        assert "PRIVATE-" not in json.dumps(context)
        assert all(item.get("type") != "support_report" for item in owner._items)
        assert len(transport.uploads) == 1
        if not failed:
            assert any(
                item.get("type") == "support_report" and result["ticket"] in item["content"]
                for item in owner.report_archive
            )
    finally:
        release.set()
        if sending is not None:
            await sending
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"summary": " "},
        {"summary": 42},
        {"summary": "x" * 2001},
        {"summary": "valid", "extra": "not allowed"},
    ],
)
async def test_invalid_proposal_never_creates_offer_or_reads_log(arguments):
    transport = ReportResponses(
        {"status": "completed", "output": [{**proposal(), "arguments": json.dumps(arguments)}]},
        answer("No report offer."),
    )
    owner = owner_for(transport, lambda: pytest.fail("Proposal read the log"))
    await owner.start()
    try:
        result = await owner.send("App help")
        assert result.tool_results[0].result == {"status": "rejected", "code": "invalid_tool_call"}
        assert not owner.snapshot.report_offers and not transport.uploads
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [False, True], ids=["completion", "new_input"])
async def test_report_waits_for_committed_diagnosis_and_never_reattaches_stale_offer(interrupt):
    entered, release = asyncio.Event(), asyncio.Event()
    snapshots = []

    class Held(ReportResponses):
        async def respond(self, **request):
            if len(self.requests) == 1:
                self.requests.append(request)
                yield {"type": "text_delta", "text": "The diagnostic evidence is incomplete. "}
                entered.set()
                await release.wait()
                yield {
                    "type": "completed",
                    "response": answer("The diagnostic evidence is incomplete. You can send it."),
                }
                return
            async for event in super().respond(**request):
                yield event

    transport = Held(
        {"status": "completed", "output": [proposal()]},
        answer("Unrelated next answer"),
    )
    owner = owner_for(transport)
    await owner.start()
    owner.observe(snapshots.append)
    first = asyncio.create_task(owner.send("First unresolved problem"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        (pending,) = owner.snapshot.report_offers
        assert pending.status == "pending"
        assert pending.message_index is None and pending.response_index is None
        assert not pending.displayed and not pending.clicked
        assert owner.snapshot.presentation.sentences
        assert not any(message.role == "assistant" for message in owner.snapshot.messages)
        assert owner.mark_report_displayed(pending.id)["status"] == "rejected"
        assert (await owner.send_report(pending.id))["status"] == "rejected"
        assert not transport.uploads
        if interrupt:
            await owner.send("A different request")
            assert (await first).status == "cancelled"
            (cancelled,) = owner.snapshot.report_offers
            assert cancelled.id == pending.id and cancelled.status == "cancelled"
            assert cancelled.message_index is None
            assert (await owner.send_report(cancelled.id))["status"] == "rejected"
            assert not transport.uploads
        else:
            release.set()
            result = await first
            assert result.status == "completed"
            (offer,) = owner.snapshot.report_offers
            assert offer.id == pending.id and offer.status == "offered"
            assert owner.snapshot.messages[offer.message_index].text == result.text
            assert offer.generation == pending.generation
            assert offer.response_index == owner.snapshot.presentation.response_index
            assert not offer.displayed and not offer.clicked
            assert owner.mark_report_displayed(offer.id)["status"] == "displayed"
            assert (await owner.send_report(offer.id))["status"] == "sent"
            assert owner.snapshot.report_offers[0].clicked
            assert len(transport.uploads) == 1
        assert all(
            report.message_index is None
            or snapshot.messages[report.message_index].role == "assistant"
            for snapshot in snapshots
            for report in snapshot.report_offers
        )
    finally:
        release.set()
        await owner.close()
        await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_proposal_has_one_action_and_rejection_never_uploads():
    transport = ReportResponses(
        {"status": "completed", "output": [proposal()]},
        {
            "status": "completed",
            "output": [{**proposal("Additional evidence"), "call_id": "second-proposal"}],
        },
        answer("The failure could not be resolved from available evidence."),
    )
    owner = owner_for(transport)
    offer = await make_offer(owner)
    try:
        assert len(owner.snapshot.report_offers) == 1
        results = owner.snapshot.tool_results
        assert results[0].result == {"status": "pending", "offer_id": offer.id}
        assert results[1].status == "not_executed"
        assert owner.reject_report(offer.id) == {"status": "rejected", "offer_id": offer.id}
        assert (await owner.send_report(offer.id))["status"] == "rejected"
        assert not transport.uploads
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", [False, True])
async def test_declined_report_reaches_next_message_without_resuming_or_display(reset):
    transport = ReportResponses(
        {"status": "completed", "output": [proposal()]},
        answer("The failure could not be resolved from available evidence."),
        answer("Understood."),
    )
    owner = owner_for(transport)
    offer = await make_offer(owner)
    try:
        requests, messages = len(transport.requests), owner.snapshot.messages
        assert owner.reject_report(offer.id)["status"] == "rejected"
        await asyncio.sleep(0)
        assert owner._notice_task is None
        assert len(transport.requests) == requests
        assert owner.snapshot.messages == messages
        assert "result=declined" in json.dumps(owner.report_archive)
        if reset:
            owner.new_conversation()
        await owner.send("Never mind the report.")
        assert ("result=declined" in json.dumps(transport.requests[-1]["input"])) is not reset
        assert not any("result=declined" in row.text for row in owner.snapshot.messages)
        assert not transport.uploads
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_failed_uncertain_delivery_retries_exact_bytes_after_new_input_and_reset():
    entered, release = asyncio.Event(), asyncio.Event()

    async def upload(request):
        if len(transport.uploads) == 1:
            entered.set()
            await release.wait()
            raise AgentServiceError("service_allowance_exhausted")
        return {"ticket": request["ticket"]}

    log = [b"initial clicked log"]
    transport = ReportResponses(
        {"status": "completed", "output": [proposal()]},
        answer("Report offer"),
        answer("New input still works"),
        upload=upload,
    )
    owner = owner_for(transport, lambda: log[0])
    offer = await make_offer(owner)
    caller = asyncio.create_task(owner.send_report(offer.id))
    try:
        await entered.wait()
        pending = owner.snapshot.report_offers[0]
        assert pending.status == "sending" and not owner.snapshot.busy
        assert (await owner.send_report(offer.id))["status"] == "sending"
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        await owner.send("Another question while sending")
        owner.new_conversation()
        assert owner.snapshot.report_offers[0].status == "sending"
        assert not transport.closed
        release.set()
        task = owner._report_tasks[offer.id]
        assert (await task)["status"] == "failed"
        assert not owner._allowance_exhausted
        before = (len(transport.classifications), len(transport.requests))
        await asyncio.sleep(0)
        assert len(transport.uploads) == 1
        log[0] = b"later changed log must not replace retry bytes"
        assert (await owner.send_report(offer.id))["status"] == "sent"
        assert len(transport.uploads) == 2
        assert transport.uploads[0] == transport.uploads[1]
        assert contents(transport.uploads[1]["bundle"])["runtime.log"] == b"initial clicked log"
        assert before == (len(transport.classifications), len(transport.requests))
    finally:
        release.set()
        await owner.close()


@pytest.mark.asyncio
async def test_missing_log_is_retry_not_fabricated_empty_evidence():
    attempts = []

    def read_log():
        attempts.append(1)
        if len(attempts) == 1:
            raise FileNotFoundError("Synthetic delayed-open log")
        return b"now written"

    transport = ReportResponses({"status": "completed", "output": [proposal()]}, answer("Offer"))
    owner = owner_for(transport, read_log)
    offer = await make_offer(owner)
    try:
        assert (await owner.send_report(offer.id))["status"] == "failed"
        failed = owner.snapshot.report_offers[0]
        assert not transport.uploads
        assert (await owner.send_report(offer.id))["status"] == "sent"
        assert owner.snapshot.report_offers[0].ticket == failed.ticket
        assert contents(transport.uploads[0]["bundle"])["runtime.log"] == b"now written"
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_close_freezes_new_clicks_and_settles_owned_thread_and_upload():
    loop = asyncio.get_running_loop()
    entered, uploading, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release_thread = threading.Event()

    def read_log():
        loop.call_soon_threadsafe(entered.set)
        assert release_thread.wait(5)
        return b"written before shutdown"

    async def upload(request):
        uploading.set()
        await finish.wait()
        assert not transport.closed
        return {"ticket": request["ticket"]}

    transport = ReportResponses(
        {"status": "completed", "output": [proposal()]}, answer("Offer"), upload=upload
    )
    owner = owner_for(transport, read_log)
    offer = await make_offer(owner)
    sending = asyncio.create_task(owner.send_report(offer.id))
    closing = None
    try:
        await entered.wait()
        owner.freeze_ingress()
        assert (await owner.send_report(offer.id))["status"] == "rejected"
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        assert not closing.done() and not transport.closed
        release_thread.set()
        await uploading.wait()
        assert not closing.done()
        finish.set()
        assert (await sending)["status"] == "sent"
        await closing
        assert transport.closed and owner.snapshot.phase == "closed"
        assert owner.snapshot.report_offers[0].status == "sent"
        assert owner.report_archive[-1]["content"].startswith(
            "[app notice] support_report sent ticket=PP-"
        )
    finally:
        release_thread.set()
        finish.set()
        await owner.close()
        await asyncio.gather(sending, *([closing] if closing else []), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt", [{"ticket": "PP-0000000000000000"}, {}, {"error": "lost receipt"}]
)
async def test_transport_requires_matching_receipt_and_preserves_binary_wire(receipt):
    ticket = "PP-0123456789ABCDEF"
    bundle = build_support_bundle(
        ticket=ticket,
        app_version="2.10.0",
        summary="synthetic summary inside only",
        diagnostic_codes=("provider.auth",),
        archive=({"role": "user", "content": "synthetic chat"},),
        log_snapshot=lambda: b"synthetic runtime",
    )
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=receipt)

    transport = AgentRelayTransport(
        "http://127.0.0.1:8788/v1/respond", Tokens(), allow_loopback_http=True
    )
    await transport._client.aclose()
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(AgentTransportError, match="invalid_support_receipt"):
            await transport.upload_support(
                bundle=bundle,
                ticket=ticket,
                app_version="2.10.0",
                diagnostic_codes=("provider.auth",),
            )
        (request,) = seen
        assert request.url.path == "/v1/support" and request.content == bundle
        assert request.headers["authorization"] == "Bearer synthetic-installation-token"
        assert request.headers["content-type"] == "application/gzip"
        assert "content-encoding" not in request.headers
        assert request.headers["x-agent-protocol"] == "3"
        assert request.headers["x-support-ticket"] == ticket
        assert request.headers["x-app-version"] == "2.10.0"
        assert json.loads(request.headers["x-diagnostic-codes"]) == ["provider.auth"]
        assert "synthetic summary" not in str(request.headers)
        assert "synthetic chat" not in str(request.headers)
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_bundle_limit_is_compressed_bytes_without_truncation():
    import os

    compressible = b"x" * 25_000_001
    incompressible = os.urandom(25_000_001)
    log = [compressible]
    transport = ReportResponses(
        {"status": "completed", "output": [proposal()]},
        answer("First offer"),
        {"status": "completed", "output": [{**proposal(), "call_id": "second"}]},
        answer("Second offer"),
    )
    owner = owner_for(transport, lambda: log[0])
    first = await make_offer(owner)
    try:
        assert (await owner.send_report(first.id))["status"] == "sent"
        bundle = transport.uploads[0]["bundle"]
        assert len(bundle) < 25_000_000
        assert contents(bundle)["runtime.log"] == compressible
        log[0] = incompressible
        await owner.send("Another unresolved app issue")
        second = owner.snapshot.report_offers[-1]
        assert (await owner.send_report(second.id))["status"] == "failed"
        retained = owner._reports[second.id].bundle
        assert len(retained) > 25_000_000
        log[0] = b"changed after failed build"
        assert (await owner.send_report(second.id))["status"] == "failed"
        assert owner._reports[second.id].bundle is retained
        assert len(transport.uploads) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_report_metadata_keeps_full_bounded_summary_and_latest_ten_codes():
    transport = ReportResponses(
        {"status": "completed", "output": [proposal("s" * 2000)]}, answer("Offer")
    )
    records = Records()
    codes = sorted(DOMAIN_CODES)[:15]
    for code in codes:
        records.history.retain(
            ErrorDiagnostics(
                component="translation_request",
                operation="translate",
                code=code,
                category="provider",
                visibility="basic",
                content_policy="metadata_only",
                status_code=None,
                retry_after_ms=None,
                cause_id="b" * 32,
                fields={},
            ),
            severity="error",
            disposition="failed",
        )
    owner = AgentChatOwner(
        records, transport, localize=lambda key: key, support_log_snapshot=lambda: b"current log"
    )
    offer = await make_offer(owner)
    try:
        assert (await owner.send_report(offer.id))["status"] == "sent"
        sent = transport.uploads[0]
        expected = tuple(reversed(codes[-10:]))
        assert sent["diagnostic_codes"] == expected
        meta = json.loads(contents(sent["bundle"])["meta.json"])
        assert meta["summary"] == "s" * 2000
        assert meta["diagnostic_codes"] == list(expected)
    finally:
        await owner.close()
