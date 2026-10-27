from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

from puripuly_heart.core.local_asr.local_qwen_runtime import (
    LocalQwenRuntimeBootstrapError,
    ensure_local_qwen_windows_runtime,
)
from puripuly_heart.core.local_asr.local_stt_assets import (
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    LocalParakeetSherpaLoadError,
)
from puripuly_heart.providers.stt.local_qwen_sherpa import (
    LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ,
    LocalQwenSherpaSTTBackend,
)


class LocalParakeetSherpaInferenceError(RuntimeError):
    pass


class _LocalParakeetSherpaImportError(ImportError):
    pass


def _recognizer_class() -> tuple[object, type[object]]:
    ensure_local_qwen_windows_runtime()
    try:
        import sherpa_onnx

        recognizer_module = importlib.import_module("sherpa_onnx.offline_recognizer")
    except ImportError as exc:
        raise _LocalParakeetSherpaImportError from exc
    return sherpa_onnx, getattr(recognizer_module, "_Recognizer")


def create_local_parakeet_v3_sherpa_recognizer(
    *,
    model_dir: Path,
    num_threads: int,
    sample_rate_hz: int = LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ,
    feature_dim: int = 80,
    provider: str = "cpu",
) -> object:
    if sample_rate_hz != LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ:
        raise ValueError(f"sample_rate_hz must be {LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ}")
    sherpa_onnx, recognizer_cls = _recognizer_class()
    transducer_config = sherpa_onnx.OfflineTransducerModelConfig(
        encoder_filename=str(model_dir / "encoder.int8.onnx"),
        decoder_filename=str(model_dir / "decoder.int8.onnx"),
        joiner_filename=str(model_dir / "joiner.int8.onnx"),
    )
    model_config = sherpa_onnx.OfflineModelConfig(
        transducer=transducer_config,
        tokens=str(model_dir / "tokens.txt"),
        num_threads=num_threads,
        debug=False,
        provider=provider,
        model_type="nemo_transducer",
    )
    feat_config = sherpa_onnx.FeatureExtractorConfig(
        sampling_rate=sample_rate_hz,
        feature_dim=feature_dim,
    )
    recognizer_config = sherpa_onnx.OfflineRecognizerConfig(
        feat_config=feat_config,
        model_config=model_config,
        decoding_method="greedy_search",
    )
    return recognizer_cls(recognizer_config)


def create_local_parakeet_japanese_sherpa_recognizer(
    *,
    model_dir: Path,
    num_threads: int,
    sample_rate_hz: int = LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ,
    feature_dim: int = 80,
    provider: str = "cpu",
) -> object:
    if sample_rate_hz != LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ:
        raise ValueError(f"sample_rate_hz must be {LOCAL_QWEN_RECOGNIZER_SAMPLE_RATE_HZ}")
    sherpa_onnx, recognizer_cls = _recognizer_class()
    nemo_config = sherpa_onnx.OfflineNemoEncDecCtcModelConfig(
        model=str(model_dir / "model.int8.onnx")
    )
    model_config = sherpa_onnx.OfflineModelConfig(
        nemo_ctc=nemo_config,
        tokens=str(model_dir / "tokens.txt"),
        num_threads=num_threads,
        debug=False,
        provider=provider,
    )
    feat_config = sherpa_onnx.FeatureExtractorConfig(
        sampling_rate=sample_rate_hz,
        feature_dim=feature_dim,
    )
    recognizer_config = sherpa_onnx.OfflineRecognizerConfig(
        feat_config=feat_config,
        model_config=model_config,
        decoding_method="greedy_search",
    )
    return recognizer_cls(recognizer_config)


@dataclass(slots=True)
class LocalParakeetV3SherpaSTTBackend(LocalQwenSherpaSTTBackend):
    feature_dim: int = field(default=80, init=False)
    model_id: str = field(default=PARAKEET_V3_MODEL_ID, init=False)
    provider_id: str = field(default="local_parakeet_v3", init=False)

    def _create_recognizer(self) -> object:
        try:
            return create_local_parakeet_v3_sherpa_recognizer(
                model_dir=self.model_dir,
                num_threads=self.num_threads,
                sample_rate_hz=self.sample_rate_hz,
                feature_dim=self.feature_dim,
                provider=self.provider,
            )
        except (LocalQwenRuntimeBootstrapError, _LocalParakeetSherpaImportError) as exc:
            raise LocalParakeetSherpaLoadError(str(exc)) from exc
        except Exception as exc:
            raise LocalParakeetSherpaLoadError(str(exc)) from exc

    def _inference_error(self, exc: Exception) -> RuntimeError:
        return LocalParakeetSherpaInferenceError(str(exc))

    def is_known_hallucination(self, text: str) -> bool:
        _ = text
        return False


@dataclass(slots=True)
class LocalParakeetJapaneseSherpaSTTBackend(LocalQwenSherpaSTTBackend):
    feature_dim: int = field(default=80, init=False)
    model_id: str = field(default=PARAKEET_JAPANESE_MODEL_ID, init=False)
    provider_id: str = field(default="local_parakeet_ja", init=False)

    def _create_recognizer(self) -> object:
        try:
            return create_local_parakeet_japanese_sherpa_recognizer(
                model_dir=self.model_dir,
                num_threads=self.num_threads,
                sample_rate_hz=self.sample_rate_hz,
                feature_dim=self.feature_dim,
                provider=self.provider,
            )
        except (LocalQwenRuntimeBootstrapError, _LocalParakeetSherpaImportError) as exc:
            raise LocalParakeetSherpaLoadError(str(exc)) from exc
        except Exception as exc:
            raise LocalParakeetSherpaLoadError(str(exc)) from exc

    def _inference_error(self, exc: Exception) -> RuntimeError:
        return LocalParakeetSherpaInferenceError(str(exc))

    def is_known_hallucination(self, text: str) -> bool:
        _ = text
        return False
