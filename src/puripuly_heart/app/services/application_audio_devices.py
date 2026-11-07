from __future__ import annotations

from puripuly_heart.app.ports.audio_test import AudioDiagnosticCandidate
from puripuly_heart.config.audio_host_api import normalize_input_host_api
from puripuly_heart.core.diagnostic_validation import (
    DIAGNOSTIC_SINK_PERSISTED_LOGS,
    redact_text_for_sink,
)

_VR_MICROPHONE_DRIVER_NAMES = (
    ("virtual desktop audio",),
    ("steam streaming microphone",),
    ("oculus virtual audio device",),
    (
        "picostreamingmicrophone",
        "pico streaming virtual audio",
        "pico streaming audio microphone array",
        "pico streaming audio wave microphone array",
    ),
)


def _vr_microphone_family(name: str) -> int | None:
    normalized = name.casefold()
    for family, driver_names in enumerate(_VR_MICROPHONE_DRIVER_NAMES):
        if any(driver_name in normalized for driver_name in driver_names):
            return family
    return None


def safe_audio_device_label(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = "".join(character for character in value if character.isprintable()).strip()[:160]
    return redact_text_for_sink(text, DIAGNOSTIC_SINK_PERSISTED_LOGS).text or ""


def enumerate_microphone_diagnostic_candidates(
    *, input_host_api: str, input_device: str, max_candidates: int
) -> tuple[AudioDiagnosticCandidate, ...]:
    import sounddevice as sd

    from puripuly_heart.core.audio.source import observe_microphone_test_route

    apis = sd.query_hostapis()
    devices = sd.query_devices()
    observation = observe_microphone_test_route(
        saved_host_api=input_host_api, requested_device=input_device
    )
    current_index = observation.resolved_device_idx
    if not input_host_api.strip() and not input_device.strip():
        try:
            current_index = int(sd.default.device[0])
            if not 0 <= current_index < len(devices):
                default = sd.query_devices(kind="input")
                matches = [
                    index
                    for index, device in enumerate(devices)
                    if device.get("name") == default.get("name")
                    and device.get("hostapi") == default.get("hostapi")
                    and int(device.get("max_input_channels", 0) or 0) > 0
                ]
                current_index = matches[0] if len(matches) == 1 else None
        except Exception:
            current_index = None
    candidates: list[AudioDiagnosticCandidate] = []
    profile = normalize_input_host_api(input_host_api)
    for index, device in enumerate(devices):
        channels = int(device.get("max_input_channels", 0) or 0)
        api_index = int(device.get("hostapi", -1))
        if channels <= 0 or not 0 <= api_index < len(apis):
            continue
        host_api = str(apis[api_index].get("name", ""))
        name = str(device.get("name", ""))
        if not name or not host_api:
            continue
        current = index == current_index and observation.should_attempt_open
        candidates.append(
            AudioDiagnosticCandidate(
                input_host_api=(
                    profile.saved_value if current and profile.actual_host_api else host_api
                ),
                input_device=name,
                device_index=index,
                max_input_channels=channels,
                current_route=current,
            )
        )
    candidates.sort(
        key=lambda candidate: (
            not candidate.current_route,
            _vr_microphone_family(candidate.input_device) is None,
            candidate.input_device.casefold() != input_device.casefold(),
            normalize_input_host_api(candidate.input_host_api).actual_host_api
            != profile.actual_host_api,
            candidate.device_index,
        )
    )
    current = [candidate for candidate in candidates if candidate.current_route]
    seen_families = {
        family
        for candidate in current
        if (family := _vr_microphone_family(candidate.input_device)) is not None
    }
    preferred: list[AudioDiagnosticCandidate] = []
    remaining: list[AudioDiagnosticCandidate] = []
    for candidate in candidates:
        if candidate.current_route:
            continue
        family = _vr_microphone_family(candidate.input_device)
        if family is not None and family not in seen_families:
            seen_families.add(family)
            preferred.append(candidate)
        else:
            remaining.append(candidate)
    return tuple((current + preferred + remaining)[:max_candidates])


def enumerate_audio_devices(settings: object | None) -> dict:
    """Enumerate actual input and loopback devices; report unavailable backends explicitly."""
    result: dict = {"host_apis": [], "microphones": [], "loopback_outputs": [], "errors": {}}
    host_api = getattr(
        getattr(getattr(settings, "intent", None), "audio", None), "input_host_api", ""
    )
    result["selected_input_host_api"] = host_api
    try:
        import sounddevice as sd

        apis = sd.query_hostapis()
        profile = normalize_input_host_api(host_api)
        result["host_apis"] = [str(api.get("name", "")) for api in apis]
        for device in sd.query_devices():
            if int(device.get("max_input_channels", 0) or 0) <= 0:
                continue
            index = int(device.get("hostapi", -1) or 0)
            api_name = str(apis[index].get("name", "")) if 0 <= index < len(apis) else ""
            result["microphones"].append(
                {
                    "name": str(device.get("name", "")),
                    "host_api": api_name,
                    "selected_host_api": not profile.actual_host_api
                    or api_name == profile.actual_host_api,
                }
            )
    except Exception as exc:
        result["errors"]["microphones"] = type(exc).__name__
    manager = None
    try:
        import pyaudiowpatch as pyaudio

        manager = pyaudio.PyAudio()
        result["loopback_outputs"] = sorted(
            {
                str(info.get("name", "")).strip()
                for info in manager.get_loopback_device_info_generator()
                if str(info.get("name", "")).strip()
            }
        )
    except Exception as exc:
        result["errors"]["loopback_outputs"] = type(exc).__name__
    finally:
        if manager is not None:
            manager.terminate()
    return result
