"""
convert_to_pdf.py
=================
Converts implementation_plan.md into a beautifully formatted, publication-grade PDF
using Microsoft Edge headless engine.
"""

import os, re, subprocess

REPO_ROOT = r"C:\Barca Hackathon\more-than-a-hack-2027"
MD_PATH = os.path.join(REPO_ROOT, "implementation_plan.md")
HTML_PATH = os.path.join(REPO_ROOT, "implementation_plan.html")
PDF_PATH = os.path.join(REPO_ROOT, "implementation_plan.pdf")
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

with open(MD_PATH, "r", encoding="utf-8") as f:
    md_text = f.read()

# Simple markdown to HTML converter for our document
lines = md_text.splitlines()
html_lines = []
in_code = False
code_lang = ""
code_buf = []
in_table = False

def escape_html(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def inline_format(t):
    # Bold italic
    t = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', t)
    # Bold
    t = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', t)
    # Italic
    t = re.sub(r'\*(.*?)\*', r'<em>\1</em>', t)
    # Inline code
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    # Math inline $...$
    t = re.sub(r'\$([^\$]+)\$', r'<span class="math">\1</span>', t)
    return t

for line in lines:
    stripped = line.strip()
    
    # Code block
    if stripped.startswith("```"):
        if not in_code:
            in_code = True
            code_lang = stripped[3:].strip()
            code_buf = []
        else:
            in_code = False
            html_lines.append(f'<pre class="code-block {code_lang}"><code>' + escape_html("\n".join(code_buf)) + "</code></pre>")
            code_buf = []
        continue
    
    if in_code:
        code_buf.append(line)
        continue
    
    # Table row
    if stripped.startswith("|") and stripped.endswith("|"):
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        if all(re.match(r'^:?-+:?$', c) for c in cells):
            continue  # Separator row
        if not in_table:
            in_table = True
            html_lines.append('<div class="table-wrapper"><table><thead><tr>')
            for c in cells:
                html_lines.append(f'<th>{inline_format(c)}</th>')
            html_lines.append('</tr></thead><tbody>')
        else:
            html_lines.append('<tr>')
            for c in cells:
                html_lines.append(f'<td>{inline_format(c)}</td>')
            html_lines.append('</tr>')
        continue
    else:
        if in_table:
            in_table = False
            html_lines.append('</tbody></table></div>')
    
    if not stripped:
        html_lines.append("<br/>")
        continue
    
    # Headers
    if stripped.startswith("# "):
        html_lines.append(f'<h1 class="doc-title">{inline_format(stripped[2:])}</h1>')
    elif stripped.startswith("## "):
        html_lines.append(f'<h2 class="section-title">{inline_format(stripped[3:])}</h2>')
    elif stripped.startswith("### "):
        html_lines.append(f'<h3 class="subsection-title">{inline_format(stripped[4:])}</h3>')
    elif stripped.startswith("#### "):
        html_lines.append(f'<h4 class="subsubsection-title">{inline_format(stripped[5:])}</h4>')
    elif stripped.startswith("---"):
        html_lines.append('<hr class="divider"/>')
    elif stripped.startswith("* ") or stripped.startswith("- "):
        html_lines.append(f'<li class="bullet-item">{inline_format(stripped[2:])}</li>')
    elif re.match(r'^\d+\.\s+', stripped):
        item_text = re.sub(r'^\d+\.\s+', '', stripped)
        html_lines.append(f'<li class="num-item">{inline_format(item_text)}</li>')
    elif stripped.startswith("> "):
        html_lines.append(f'<blockquote class="callout">{inline_format(stripped[2:])}</blockquote>')
    else:
        html_lines.append(f'<p class="body-para">{inline_format(line)}</p>')

if in_table:
    html_lines.append('</tbody></table></div>')

body_content = "\n".join(html_lines)

styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ball Possession Engine - Implementation Plan</title>
<style>
  @page {{
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
    @bottom-right {{
      content: counter(page);
    }}
  }}
  body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
    color: #1a1f36;
    line-height: 1.6;
    font-size: 10.5pt;
    background: #ffffff;
  }}
  .doc-title {{
    color: #0c2340;
    font-size: 22pt;
    font-weight: 700;
    border-bottom: 3px solid #004d98;
    padding-bottom: 8px;
    margin-bottom: 12px;
  }}
  .section-title {{
    color: #004d98;
    font-size: 14pt;
    font-weight: 600;
    border-bottom: 1.5px solid #e2e8f0;
    padding-bottom: 4px;
    margin-top: 22px;
    margin-bottom: 10px;
    page-break-after: avoid;
  }}
  .subsection-title {{
    color: #a50044;
    font-size: 12pt;
    font-weight: 600;
    margin-top: 16px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}
  .subsubsection-title {{
    color: #2d3748;
    font-size: 11pt;
    font-weight: 600;
    margin-top: 12px;
    margin-bottom: 4px;
  }}
  p.body-para {{
    margin: 6px 0;
  }}
  .math {{
    font-family: 'Cambria Math', 'Latin Modern Math', Georgia, serif;
    font-style: italic;
    background: #f7fafc;
    padding: 1px 4px;
    border-radius: 3px;
  }}
  code {{
    font-family: 'Cascadia Code', Consolas, Monaco, monospace;
    font-size: 9pt;
    background: #edf2f7;
    color: #c53030;
    padding: 2px 4px;
    border-radius: 3px;
  }}
  pre.code-block {{
    background: #1a202c;
    color: #edf2f7;
    font-family: 'Cascadia Code', Consolas, Monaco, monospace;
    font-size: 8.5pt;
    line-height: 1.45;
    padding: 12px 14px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 10px 0;
    page-break-inside: avoid;
  }}
  pre.code-block code {{
    background: transparent;
    color: inherit;
    padding: 0;
  }}
  .table-wrapper {{
    margin: 12px 0;
    page-break-inside: avoid;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
  }}
  th, td {{
    padding: 6px 10px;
    border: 1px solid #cbd5e0;
    text-align: left;
  }}
  th {{
    background: #ebf4ff;
    color: #1a365d;
    font-weight: 600;
  }}
  tr:nth-child(even) td {{
    background: #f7fafc;
  }}
  blockquote.callout {{
    border-left: 4px solid #004d98;
    background: #f0f7ff;
    margin: 10px 0;
    padding: 8px 12px;
    color: #2c5282;
    border-radius: 0 4px 4px 0;
  }}
  hr.divider {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 18px 0;
  }}
  li.bullet-item {{
    margin-left: 20px;
    margin-bottom: 4px;
  }}
  li.num-item {{
    margin-left: 24px;
    margin-bottom: 4px;
  }}
</style>
</head>
<body>
{body_content}
</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(styled_html)

print(f"Generated HTML: {HTML_PATH}")

# Invoke Edge to convert to PDF
cmd = [
    EDGE_PATH,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={PDF_PATH}",
    HTML_PATH
]

print("Executing Microsoft Edge headless PDF engine...")
res = subprocess.run(cmd, capture_output=True, text=True)
if os.path.exists(PDF_PATH):
    size_kb = os.path.getsize(PDF_PATH) / 1024
    print(f"SUCCESS: PDF generated successfully: {PDF_PATH} ({size_kb:.1f} KB)")
else:
    print("PDF generation failed.")
    print("STDERR:", res.stderr)

