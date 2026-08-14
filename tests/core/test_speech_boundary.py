from puripuly_heart.core.speech_boundary import boundary_wait_ms


def test_boundary_wait_ms_preserves_vad_tail_and_unknown_lifecycle_boundary() -> None:
    assert boundary_wait_ms(None, observed_tail_ms=0) is None
    assert boundary_wait_ms("silence", observed_tail_ms=500) == 500
    assert boundary_wait_ms("soft_pause", observed_tail_ms=160) == 160
    assert boundary_wait_ms("max_duration", observed_tail_ms=0) == 0
    assert boundary_wait_ms("max_duration", observed_tail_ms=160) == 160
