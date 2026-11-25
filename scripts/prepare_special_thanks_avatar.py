from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

MAX_AVATAR_SIZE = 256
AVATAR_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "puripuly_heart"
    / "data"
    / "pictures"
    / "special_thanks"
)


def _normalize_mode(image: Image.Image) -> Image.Image:
    has_alpha = image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info
    if has_alpha:
        rgba = image.convert("RGBA")
        min_alpha, _ = rgba.getchannel("A").getextrema()
        if min_alpha < 255:
            return rgba
        return rgba.convert("RGB")
    return image.convert("RGB")


def _resolve_destination(name: str, output: Path | None) -> Path:
    if output is not None:
        return output
    cleaned = name.removesuffix(".png")
    if not cleaned or Path(cleaned).name != cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Invalid avatar name: {name!r}")
    return AVATAR_DIR / f"{cleaned}.png"


def normalize_avatar(
    source: Path,
    destination: Path,
    *,
    max_size: int = MAX_AVATAR_SIZE,
) -> tuple[int, int, str]:
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    with Image.open(source) as raw:
        transposed = ImageOps.exif_transpose(raw)
        normalized = _normalize_mode(transposed)
        normalized.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        normalized.save(destination, format="PNG", optimize=True)
        return normalized.size[0], normalized.size[1], normalized.mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a Special Thanks avatar into a resized PNG."
    )
    parser.add_argument("source", type=Path, help="Path to the input image file.")
    parser.add_argument(
        "name",
        help="Target avatar filename or stem under data/pictures/special_thanks/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output PNG path.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=MAX_AVATAR_SIZE,
        help=f"Maximum width/height in pixels (default: {MAX_AVATAR_SIZE}).",
    )
    args = parser.parse_args(argv)
    destination = _resolve_destination(args.name, args.output)
    width, height, mode = normalize_avatar(
        args.source,
        destination,
        max_size=args.max_size,
    )
    print(f"Saved {destination} ({width}x{height}, {mode}, {destination.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
