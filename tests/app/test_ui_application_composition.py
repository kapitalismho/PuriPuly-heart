from pathlib import Path

import puripuly_heart.composition.ui_application as composition_module


def test_composition_forwards_options_without_passing_a_flet_page(monkeypatch) -> None:
    captured: dict[str, object] = {}
    presentation = object()
    logging_sinks = object()
    presence = object()
    runtime_components = object()
    runtime_logging_owner = object()
    state_owner = object()
    runtime_logging_inputs: dict[str, object] = {}
    boundary_inputs: dict[str, object] = {}

    class Backend:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.installed_runtime = None
            self.installed_runtime_logging = None
            self.installed_startup = None

        def install_runtime_logging_owner(self, owner: object) -> None:
            self.installed_runtime_logging = owner

        @property
        def runtime_logging_owner(self) -> object:
            return self.installed_runtime_logging

        def install_runtime_composition(self, components: object) -> None:
            self.installed_runtime = components

        def install_startup_owner(self, owner: object) -> None:
            self.installed_startup = owner

        def _log_error(self, message: str) -> None:
            _ = message

        async def _emit_overlay_runtime_logging_mode_update(self) -> None:
            return None

        def _get_overlay_application_owner(self) -> object:
            raise AssertionError("overlay availability must remain lazy")

    backends: list[Backend] = []

    def controller_factory(**kwargs):
        backend = Backend(**kwargs)
        backends.append(backend)
        return backend

    monkeypatch.setattr(composition_module, "GuiController", controller_factory)
    monkeypatch.setattr(
        composition_module,
        "compose_gui_runtime_components",
        lambda backend: runtime_components,
    )
    monkeypatch.setattr(
        composition_module,
        "compose_application_runtime_logging",
        lambda **kwargs: (runtime_logging_inputs.update(kwargs) or runtime_logging_owner),
    )

    class Boundary:
        def __init__(
            self,
            runtime: object,
            *,
            state: object,
            runtime_shutdown: object,
            runtime_logging: object,
        ) -> None:
            boundary_inputs.update(
                runtime=runtime,
                state=state,
                runtime_shutdown=runtime_shutdown,
                runtime_logging=runtime_logging,
            )

    monkeypatch.setattr(composition_module, "UiApplicationBoundary", Boundary)
    monkeypatch.setattr(
        composition_module,
        "UiApplicationStateOwner",
        lambda *_args, **_kwargs: state_owner,
    )

    application = composition_module.compose_ui_application(
        presentation=presentation,
        config_path=Path("settings.json"),
        allow_stable_settings_import=True,
        runtime_logging_sinks=logging_sinks,
        vrchat_osc_presence=presence,
    )

    [backend] = backends
    assert isinstance(application, Boundary)
    assert boundary_inputs == {
        "runtime": backend,
        "state": state_owner,
        "runtime_shutdown": backend,
        "runtime_logging": runtime_logging_owner,
    }
    assert backend.installed_runtime_logging is runtime_logging_owner
    assert runtime_logging_inputs["presentation"] is presentation
    assert runtime_logging_inputs["sinks"] is logging_sinks
    assert callable(runtime_logging_inputs["overlay_logging_mode_update"])
    assert callable(runtime_logging_inputs["overlay_logging_mode_update_available"])
    assert backend.installed_runtime is runtime_components
    assert backend.installed_startup is not None
    settings_owner = captured.pop("settings_owner")
    provider_settings_owner = captured.pop("provider_settings_owner")
    assert provider_settings_owner.settings is settings_owner
    assert captured == {
        "page": None,
        "app": presentation,
        "config_path": Path("settings.json"),
        "allow_stable_settings_import": True,
        "runtime_logging_sinks": logging_sinks,
        "vrchat_osc_presence": presence,
    }
