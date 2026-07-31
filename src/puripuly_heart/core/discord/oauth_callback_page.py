from __future__ import annotations

from html import escape

OAUTH_CALLBACK_TITLE_KEY = "auth.callback.title"
OAUTH_CALLBACK_COMPLETION_LINE_KEYS = (
    "auth.callback.line1",
    "auth.callback.line2",
    "auth.callback.line3",
)
OAUTH_CALLBACK_COMPLETION_FALLBACK_LINES = (
    "We received your authentication",
    "PuriPuly is finishing the connection in the app.",
    "You can close this tab.",
)
OAUTH_CALLBACK_FONT_FAMILIES = {
    "en": "system-ui, sans-serif",
    "ko": '"NanumSquareRound", system-ui, sans-serif',
    "ja": '"M PLUS Rounded 1c", system-ui, sans-serif',
    "zh-CN": '"ResourceHanRoundedCN", system-ui, sans-serif',
}


def resolve_oauth_callback_locale(locale: str | None) -> str:
    try:
        from puripuly_heart.ui.i18n import get_locale, resolve_locale

        return resolve_locale(locale if locale is not None else get_locale())
    except Exception:
        return "en"


def _callback_completion_line(locale: str, key: str, fallback: str) -> str:
    try:
        from puripuly_heart.ui.i18n import t_for_locale

        return t_for_locale(locale, key, default=fallback)
    except Exception:
        return fallback


def render_oauth_callback_completion_page(locale: str | None = None) -> bytes:
    resolved_locale = resolve_oauth_callback_locale(locale)
    title = _callback_completion_line(
        resolved_locale,
        OAUTH_CALLBACK_TITLE_KEY,
        "PuriPuly",
    )
    lines = [
        _callback_completion_line(resolved_locale, key, fallback)
        for key, fallback in zip(
            OAUTH_CALLBACK_COMPLETION_LINE_KEYS,
            OAUTH_CALLBACK_COMPLETION_FALLBACK_LINES,
            strict=True,
        )
    ]
    lines_html = "<br>\n".join(escape(line) for line in lines)
    font_family = OAUTH_CALLBACK_FONT_FAMILIES.get(
        resolved_locale,
        OAUTH_CALLBACK_FONT_FAMILIES["en"],
    )
    html = f"""<!doctype html>
<html lang="{escape(resolved_locale, quote=True)}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light; }}
    * {{ box-sizing: border-box; }}
    html, body {{ min-height: 100%; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 32px;
      background: #FFF8F6;
      color: #5C4D4C;
      font-family: {font_family};
    }}
    main {{
      max-width: 46rem;
      text-align: center;
      font-size: clamp(24px, 4vw, 32px);
      line-height: 1.6;
      font-weight: 600;
      word-break: keep-all;
      overflow-wrap: break-word;
    }}
    p {{ margin: 0; }}
  </style>
</head>
<body>
  <main>
    <p>{lines_html}</p>
  </main>
</body>
</html>
"""
    return html.encode("utf-8")
