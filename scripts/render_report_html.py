from __future__ import annotations

import argparse
from pathlib import Path

import markdown


STYLE = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif;
  color: #222;
  line-height: 1.65;
  max-width: 980px;
  margin: 32px auto 72px;
  padding: 0 28px;
}
h1, h2, h3 { line-height: 1.25; color: #111; }
h1 { font-size: 30px; border-bottom: 2px solid #222; padding-bottom: 12px; }
h2 { font-size: 22px; border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 36px; }
h3 { font-size: 17px; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; margin: 14px 0 22px; font-size: 14px; }
th, td { border: 1px solid #d9dee7; padding: 7px 9px; vertical-align: top; }
th { background: #f4f6f8; font-weight: 650; }
code { background: #f5f6f8; padding: 1px 4px; border-radius: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { background: #f5f6f8; padding: 12px 14px; border-radius: 6px; overflow-x: auto; }
pre code { background: transparent; padding: 0; }
img { display: block; max-width: 100%; margin: 18px auto 24px; border: 1px solid #e2e5ea; border-radius: 6px; }
a { color: #1f65cc; text-decoration: none; }
a:hover { text-decoration: underline; }
blockquote { border-left: 4px solid #d9dee7; margin: 16px 0; padding: 4px 16px; color: #555; }
@media print {
  body { margin: 0 auto; padding: 0 18px; max-width: 920px; }
  h2 { page-break-after: avoid; }
  img, table { page-break-inside: avoid; }
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="docs/REPORT_DRAFT_V1.md")
    parser.add_argument("--output", default="artifacts/report/REPORT_DRAFT_V1.html")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    text = input_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5",
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>基于 InternVL2.5-2B 的视觉语言 Reward Model 构建与评测</title>
  <style>{STYLE}</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
