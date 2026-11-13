from __future__ import annotations

import json
from uuid import uuid4

import pytest

from puripuly_heart.core.diagnostics import exception_diagnostics
from tests.app import test_control_repair_behavior as control_cases
from tests.app.test_control_repair_behavior import (
    isolated_composed_logging as isolated_composed_logging,
)
from tests.app.test_control_repair_behavior import offline as offline


@pytest.mark.asyncio
async def test_pending_evidence_survives_nonsettings_probe_until_completed_settings_receipt(
    offline,
):
    persisted = json.loads(offline.read_text(encoding="utf-8"))
    persisted["intent"]["telemetry"]["enabled"] = False
    offline.write_text(json.dumps(persisted), encoding="utf-8")
    app = control_cases.compose_headless_application(offline)
    try:
        await app.start()
        control = app.control()
        control.bind_instance(uuid4().hex)
        assert control.results.current is None
        causes = tuple(
            exception_diagnostics(
                PermissionError(13, "synthetic-private-presentation"),
                component="settings_projection",
                operation="render_settings",
                code="settings_render_failed",
            )
            for _ in range(5)
        )
        for cause in causes:
            control.results.add_secondary(cause)
        accepted = await control.submit("translation.probe", {}, request_id=uuid4().hex)
        probe = await control.wait(accepted["operation_id"], timeout=5)
        assert probe["terminal"] is True
        assert "transaction" not in probe
        assert control.results.current is None
        accepted = await control.submit(
            "settings.apply", {"changes": {"locale": "ja"}}, request_id=uuid4().hex
        )
        receipt = await control.wait(accepted["operation_id"], timeout=5)
        assert receipt["status"] == "applied"
        transaction = receipt["transaction"]
        assert [cause["diagnostic_id"] for cause in transaction["secondary_diagnostics"]] == [
            cause.diagnostic_id for cause in causes[:4]
        ]
        assert transaction["secondary_omitted_count"] == 1
        assert app.compatibility_settings().intent.ui.locale == "ja"
        accepted = await control.submit(
            "settings.apply", {"changes": {"locale": "ko"}}, request_id=uuid4().hex
        )
        following = await control.wait(accepted["operation_id"], timeout=5)
        assert following["status"] == "applied"
        assert following["transaction"]["secondary_diagnostics"] == []
        assert following["transaction"]["secondary_omitted_count"] == 0
        assert "synthetic-private" not in str(receipt)
    finally:
        await app.stop()
