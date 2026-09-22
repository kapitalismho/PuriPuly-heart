from __future__ import annotations

import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
ANALYSIS_PATH = ROOT / "contrast_analysis.json"

TEXTS = {
    "white": "#FFFFFF",
    "gold": "#FFD700",
    "sky": "#33D6FF",
    "mint": "#2DE1A8",
}
SCENES = {
    "dark": "#0A0C10",
    "bright": "#EDEFF2",
    "busy-navy": "#1B2A4A",
    "busy-ochre": "#7A5C2E",
    "busy-teal": "#3AA58B",
    "busy-fog": "#E8E8E8",
}
BACKDROP_ALPHA = 0.0

CVD = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}
CVD_METHOD = "Machado-Oliveira-Fernandes-2009 severity 1.0, applied in linear light"


def hex_to_rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def srgb_to_linear(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def luminance(rgb: tuple) -> float:
    red, green, blue = (srgb_to_linear(c) for c in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(fg: tuple, bg: tuple) -> float:
    first, second = sorted((luminance(fg), luminance(bg)), reverse=True)
    return (first + 0.05) / (second + 0.05)


def blend_over(scene: tuple, alpha: float) -> tuple:
    return tuple((1.0 - alpha) * c for c in scene)


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def linear_to_srgb(channel: float) -> float:
    if channel <= 0.0031308:
        return 12.92 * channel
    return 1.055 * max(channel, 0.0) ** (1.0 / 2.4) - 0.055


def simulate(rgb: tuple, matrix: tuple) -> tuple:
    linear = [srgb_to_linear(c) for c in rgb]
    out = [clamp01(sum(matrix[row][col] * linear[col] for col in range(3))) for row in range(3)]
    return tuple(linear_to_srgb(c) for c in out)


def color_distance(a: tuple, b: tuple) -> float:
    la = tuple(srgb_to_linear(c) for c in a)
    lb = tuple(srgb_to_linear(c) for c in b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(la, lb)))


def main() -> None:
    text_rgb = {name: hex_to_rgb(value) for name, value in TEXTS.items()}
    scene_rgb = {name: hex_to_rgb(value) for name, value in SCENES.items()}
    backdrops = {name: blend_over(rgb, BACKDROP_ALPHA) for name, rgb in scene_rgb.items()}
    text_table = {}
    for name, fg in text_rgb.items():
        row = {}
        for scene, bg in backdrops.items():
            row[scene] = round(contrast(fg, bg), 2)
        text_table[name] = row
    pairs = [
        ("gold", "sky"),
        ("gold", "mint"),
        ("white", "gold"),
        ("white", "sky"),
        ("white", "mint"),
    ]
    role_rows = []
    for first, second in pairs:
        base = {
            "pair": f"{first}-vs-{second}",
            "linear_distance": round(color_distance(text_rgb[first], text_rgb[second]), 3),
            "luminance_delta": round(
                abs(luminance(text_rgb[first]) - luminance(text_rgb[second])), 3
            ),
        }
        for cvd, matrix in CVD.items():
            a = simulate(text_rgb[first], matrix)
            b = simulate(text_rgb[second], matrix)
            base[f"{cvd}_distance"] = round(color_distance(a, b), 3)
            base[f"{cvd}_luminance_delta"] = round(abs(luminance(a) - luminance(b)), 3)
        role_rows.append(base)
    payload = {
        "meta": {
            "method": "WCAG relative luminance on sRGB against the raw scene sample (effective background alpha 0: native effective_background_alpha returns 0.0, preexisting text-only transparent overlay; configured 0.24 not applied). Black 5px outline not modeled numerically, so bright/busy ratios below are a lower bound before the outline contribution.",
            "cvd": CVD_METHOD,
            "cvd_reference": "Machado-Oliveira-Fernandes-2009 severity 1.0, sRGB decode, matrix in linear light, clamp, sRGB encode",
            "note": "Busy scene is a gradient; samples bound it, fog sample approximates the worst case. Bright/busy text contrast without a dimming strip is an honest risk: Gold/Sky/Mint sit near 1.1-1.6 on bright/fog and rely on the black outline plus scene variation.",
        },
        "text_vs_backdrop_contrast": text_table,
        "role_distinction": role_rows,
    }
    ANALYSIS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("text vs backdrop (strip-blended), higher is better:")
    for name, row in text_table.items():
        print(f"  {name:6s} " + " ".join(f"{scene}={value}" for scene, value in row.items()))
    print("role distinction, linear-RGB distance and luminance delta:")
    for row in role_rows:
        print("  " + " ".join(f"{key}={value}" for key, value in row.items()))
    print(f"wrote {ANALYSIS_PATH.name}")


if __name__ == "__main__":
    main()
