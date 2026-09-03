from __future__ import annotations

NON_ACCUMULATING_LIFECYCLES = frozenset({"UNBOUND", "UNCERTAIN", "POISONED"})


class CommonPersistenceDecoder:
    def __init__(
        self,
        frame_ms: int,
        confirmation_ms: int = 500,
        sensitivity_ms: int = 300,
        enable_sensitivity: bool = True,
    ) -> None:
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if confirmation_ms <= 0 or sensitivity_ms <= 0:
            raise ValueError("confirmation_ms and sensitivity_ms must be positive")
        if sensitivity_ms >= confirmation_ms:
            raise ValueError("sensitivity_ms must be below confirmation_ms")
        self.frame_ms = frame_ms
        self.confirmation_ms = confirmation_ms
        self.sensitivity_ms = sensitivity_ms
        self.enable_sensitivity = enable_sensitivity
        self.run_ms = 0
        self.run_start: int | None = None

    def reset(self) -> None:
        self.run_ms = 0
        self.run_start = None

    def update(self, frame: dict, *, tau: float = 0.5) -> dict:
        speech_gt = frame.get("speech_gt")
        anchor = frame["anchor"]
        lifecycle = frame["lifecycle"]
        t = frame["source_time_ms"]
        if not speech_gt:
            self.reset()
            return {"action": "KEEP", "source_boundary_time": None, "decision_time": None,
                    "sensitivity": False}
        if lifecycle in NON_ACCUMULATING_LIFECYCLES:
            self.reset()
            return {"action": "HOLD", "source_boundary_time": None, "decision_time": None,
                    "sensitivity": False}
        if lifecycle != "BOUND":
            raise ValueError(f"unknown lifecycle: {lifecycle!r}")
        if anchor >= tau:
            self.reset()
            return {"action": "KEEP", "source_boundary_time": None, "decision_time": None,
                    "sensitivity": False}
        if self.run_start is None:
            self.run_start = t - self.frame_ms
        self.run_ms += self.frame_ms
        if self.run_ms >= self.confirmation_ms:
            out = {"action": "CUT", "source_boundary_time": self.run_start,
                   "decision_time": t, "sensitivity": False}
            assert out["source_boundary_time"] <= out["decision_time"]
            self.reset()
            return out
        sens = self.enable_sensitivity and self.run_ms >= self.sensitivity_ms
        return {"action": "CUT_SENS" if sens else "HOLD",
                "source_boundary_time": self.run_start if sens else None,
                "decision_time": t if sens else None, "sensitivity": sens}
