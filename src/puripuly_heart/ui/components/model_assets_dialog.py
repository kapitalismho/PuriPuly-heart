from __future__ import annotations

import asyncio
from pathlib import Path

import flet as ft
from puripuly_heart.core.local_stt_runtime_installer import LocalSTTRuntimeInstallCancelled

from puripuly_heart.app.services.model_assets_catalog import model_asset_choices
from puripuly_heart.core.feature_diagnostics import log_feature
from puripuly_heart.core.local_translation.assets import GEMMA_MODEL_ID, e4b_gemma_spec
from puripuly_heart.core.local_translation.provisioning import GemmaProvisioningCancelled
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import COLOR_NEUTRAL_DARK, COLOR_SURFACE


class ModelAssetsDialog:
    def __init__(self, page, application, run_task):
        self.page = page
        self.application = application
        self.run_task = run_task
        self.busy = False
        self.closed = True
        self.files = ()
        self.cancel_requested = False
        self.active_model_id = None
        self.choices = model_asset_choices()
        self.checkboxes = {}
        self.sources = {}
        self.results = {}
        self.source = ft.Dropdown(
            value=application.gemma_download_source(),
            options=[
                ft.DropdownOption("official", t("model_assets.official")),
                ft.DropdownOption("mirror", t("model_assets.mirror")),
                ft.DropdownOption("local", t("model_assets.local")),
            ],
            label=t("model_assets.source"),
            text_style=ft.TextStyle(color=COLOR_NEUTRAL_DARK),
        )
        self.status = ft.Text(t("model_assets.hint"), selectable=True, color=COLOR_NEUTRAL_DARK)
        self.progress = ft.ProgressBar(value=0)
        self.start_button = ft.Button(
            t("model_assets.start"), on_click=lambda _: self.run_task(self.install)
        )
        self.picker = ft.FilePicker()
        controls = [ft.Text(t("model_assets.selection_hint"), color=COLOR_NEUTRAL_DARK)]
        source_labels = {
            "official": t("model_assets.official"),
            "mirror": t("model_assets.mirror"),
            "modelscope_only": t("model_assets.modelscope"),
            "local": t("model_assets.local"),
        }
        for purpose in ("asr", "translation"):
            controls.append(
                ft.Text(
                    t("model_assets.group." + purpose),
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_NEUTRAL_DARK,
                )
            )
            for choice in (c for c in self.choices if c.purpose == purpose):
                checkbox = ft.Checkbox(
                    label=choice.name,
                    value=False,
                    label_style=ft.TextStyle(color=COLOR_NEUTRAL_DARK),
                )
                source = (
                    self.source
                    if choice.model_id == GEMMA_MODEL_ID
                    else ft.Dropdown(
                        value=choice.default_source,
                        options=[
                            ft.DropdownOption(value, source_labels[value])
                            for value in choice.sources
                        ],
                        label=t("model_assets.source"),
                        text_style=ft.TextStyle(color=COLOR_NEUTRAL_DARK),
                    )
                )
                result = ft.Text("", color=COLOR_NEUTRAL_DARK, selectable=True)
                self.checkboxes[choice.model_id] = checkbox
                self.sources[choice.model_id] = source
                self.results[choice.model_id] = result
                controls.extend(
                    [
                        checkbox,
                        ft.Text(
                            t(choice.description_key) + f" · {choice.size_bytes / 1024**3:.2f} GiB",
                            color=COLOR_NEUTRAL_DARK,
                        ),
                        source,
                        result,
                    ]
                )
        controls.extend(
            [
                ft.Text(
                    "\n".join(a.filename for a in e4b_gemma_spec().assets),
                    selectable=True,
                    color=COLOR_NEUTRAL_DARK,
                ),
                ft.Text(t("model_assets.gemma_import"), color=COLOR_NEUTRAL_DARK),
                ft.Button(t("model_assets.choose"), on_click=lambda _: self.run_task(self.choose)),
                ft.Button(
                    t("model_assets.choose_directory"),
                    on_click=lambda _: self.run_task(self.choose_directory),
                ),
            ]
        )
        self.dialog = ft.AlertDialog(
            title=ft.Text(t("model_assets.title"), color=COLOR_NEUTRAL_DARK),
            bgcolor=COLOR_SURFACE,
            content=ft.Column(
                [
                    ft.Column(controls, height=350, scroll=ft.ScrollMode.AUTO),
                    self.status,
                    self.progress,
                ],
                width=620,
                tight=True,
            ),
            actions=[
                self.start_button,
                ft.TextButton(
                    t("model_assets.cancel"), on_click=lambda _: self.run_task(self.cancel)
                ),
                ft.TextButton(t("model_assets.close"), on_click=lambda _: self.close()),
            ],
            on_dismiss=lambda _: self.close(pop=False),
        )

    def open(self):
        if not self.closed:
            return
        self.closed = False
        if not self.busy:
            selected = self.application.selected_model_asset_ids()
            for model_id, checkbox in self.checkboxes.items():
                checkbox.value = model_id in selected
        self.page.show_dialog(self.dialog)

    def close(self, *, pop=True):
        if self.closed:
            return
        self.closed = True
        if self.busy:
            self.cancel_requested = True
            self.run_task(self.cancel)
        if pop:
            self.page.pop_dialog()

    async def cancel(self):
        self.cancel_requested = True
        if self.active_model_id == GEMMA_MODEL_ID:
            await self.application.handle_managed_gemma_notice_action("cancel")
        elif self.active_model_id is not None:
            await self.application.cancel_asr_assets(model_id=self.active_model_id)

    async def choose(self):
        if self.busy:
            return
        try:
            selected = await self.picker.pick_files(
                allow_multiple=True,
                allowed_extensions=["gguf"],
                file_type=ft.FilePickerFileType.CUSTOM,
            )
            if selected:
                self.files = tuple(Path(file.path) for file in selected if file.path)
                self.source.value = "local"
                self.checkboxes[GEMMA_MODEL_ID].value = True
                self.status.value = "\n".join(path.name for path in self.files)
                self.page.update()
        except Exception as error:
            log_feature("ModelDownload", "preflight", "file_picker_failed", error=error)
            self.status.value = t("model_assets.error.unknown")
            self.page.update()

    async def choose_directory(self):
        if self.busy:
            return
        try:
            selected = await self.picker.get_directory_path()
            if selected:
                self.files = tuple(
                    Path(selected) / asset.filename for asset in e4b_gemma_spec().assets
                )
                self.source.value = "local"
                self.checkboxes[GEMMA_MODEL_ID].value = True
                self.status.value = "\n".join(path.name for path in self.files)
                self.page.update()
        except Exception as error:
            log_feature("ModelDownload", "preflight", "file_picker_failed", error=error)
            self.status.value = t("model_assets.error.unknown")
            self.page.update()

    async def install(self):
        if self.busy or self.closed:
            return
        selected = tuple(
            choice for choice in self.choices if self.checkboxes[choice.model_id].value
        )
        if not selected:
            self.status.value = t("model_assets.select_required")
            self.page.update()
            return
        self.busy = True
        self.cancel_requested = False
        self.start_button.disabled = True
        for control in (*self.sources.values(), *self.checkboxes.values()):
            control.disabled = True
        self.status.value = t("model_assets.connecting")
        self.page.update()

        def update(status):
            if self.closed:
                return
            self.progress.value = (
                status.downloaded_bytes / status.total_bytes if status.total_bytes else 0
            )
            phase = getattr(status, "state", getattr(status, "status", "downloading"))
            self.status.value = f"{choice.name} · {t('model_assets.phase.' + phase)}\n{status.downloaded_bytes / 1024**2:.1f} / {status.total_bytes / 1024**2:.1f} MiB"
            if getattr(status, "filename", None):
                self.status.value += "\n" + status.filename
            self.page.update()

        try:
            failures = False
            failure_message = ""
            for choice in selected:
                if self.closed or self.cancel_requested:
                    raise GemmaProvisioningCancelled("Model task cancelled")
                self.active_model_id = choice.model_id
                source = self.sources[choice.model_id].value
                self.progress.value = 0
                self.status.value = choice.name + " · " + t("model_assets.connecting")
                self.page.update()
                try:
                    if choice.model_id == GEMMA_MODEL_ID:
                        await self.application.configure_gemma_source(source)
                        if self.closed or self.cancel_requested:
                            raise GemmaProvisioningCancelled("Model task cancelled")
                        await self.application.provision_gemma_assets(
                            source=source, files=self.files, on_status=update
                        )
                    else:
                        await self.application.provision_asr_assets(
                            model_id=choice.model_id, source=source, on_status=update
                        )
                    self.results[choice.model_id].value = t("model_assets.phase.ready")
                except (
                    GemmaProvisioningCancelled,
                    LocalSTTRuntimeInstallCancelled,
                    asyncio.CancelledError,
                ):
                    raise
                except Exception as error:
                    failures = True
                    code = getattr(error, "failure_code", "unknown")
                    if (
                        not isinstance(code, str)
                        or not code.isascii()
                        or not code.replace("_", "").isalnum()
                    ):
                        code = "unknown"
                    log_feature(
                        "ModelDownload",
                        "installation",
                        code,
                        error=error,
                        http_status=getattr(error, "status_code", None),
                    )
                    self.results[choice.model_id].value = t(
                        "model_assets.error." + code, default=t("model_assets.error.unknown")
                    )
                    failure_message = choice.name + " · " + self.results[choice.model_id].value
                finally:
                    self.active_model_id = None
            self.status.value = t(
                "model_assets.partial_failure" if failures else "model_assets.complete"
            )
            if failures:
                self.status.value += "\n" + failure_message
        except GemmaProvisioningCancelled, LocalSTTRuntimeInstallCancelled, asyncio.CancelledError:
            self.status.value = t("model_assets.cancelled")
        except Exception as error:
            code = getattr(error, "failure_code", "unknown")
            log_feature("ModelDownload", "installation", code, error=error)
            self.status.value = t(
                "model_assets.error." + code, default=t("model_assets.error.unknown")
            )
        finally:
            self.busy = False
            self.active_model_id = None
            self.start_button.disabled = False
            for control in (*self.sources.values(), *self.checkboxes.values()):
                control.disabled = False
            if not self.closed:
                self.page.update()
