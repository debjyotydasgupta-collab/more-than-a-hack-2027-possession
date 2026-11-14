"""
build_pdf.py
============
Generates an executive-grade PDF document of implementation_plan.md
using ReportLab.
"""

import os, re
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted, KeepTogether
)
from reportlab.pdfgen import canvas

REPO_ROOT = r"C:\Barca Hackathon\more-than-a-hack-2027"
MD_PATH = os.path.join(REPO_ROOT, "implementation_plan.md")
PDF_PATH = os.path.join(REPO_ROOT, "implementation_plan.pdf")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Ball Possession Tracking Engine — Implementation Plan")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 747, 540, 747)
        
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(540, 36, page_str)
        self.drawString(54, 36, "CONFIDENTIAL — More Than A Hack 2027 (FC Barcelona Innovation Hub)")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 540, 46)
        
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0C2340'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#004D98'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#A50044'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyMain',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'BulletMain',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor('#E2E8F0'),
        spaceBefore=4,
        spaceAfter=6
    )

    with open(MD_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    story = []
    lines = content.splitlines()
    in_code = False
    code_buf = []

    def clean_text(t):
        # Escape XML entities for ReportLab Paragraph
        t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # Bold
        t = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', t)
        # Italic
        t = re.sub(r'\*(.*?)\*', r'<i>\1</i>', t)
        # Inline code
        t = re.sub(r'`([^`]+)`', r'<font face="Courier" color="#C53030">\1</font>', t)
        # Clean latex dollars for PDF
        t = re.sub(r'\$([^\$]+)\$', r'<b>\1</b>', t)
        return t

    for line in lines:
        s = line.strip()
        
        # Code fence
        if s.startswith('```'):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                # Add preformatted block inside table for background color
                code_text = "\n".join(code_buf)
                p_code = Preformatted(code_text, code_style)
                code_table = Table([[p_code]], colWidths=[486])
                code_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1A202C')),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ('LEFTPADDING', (0,0), (-1,-1), 8),
                    ('RIGHTPADDING', (0,0), (-1,-1), 8),
                ]))
                story.append(code_table)
                story.append(Spacer(1, 4))
            continue

        if in_code:
            code_buf.append(line)
            continue

        if not s:
            story.append(Spacer(1, 3))
            continue

        if s.startswith('# '):
            story.append(Paragraph(clean_text(s[2:]), title_style))
            story.append(Spacer(1, 2))
        elif s.startswith('## '):
            story.append(Paragraph(clean_text(s[3:]), h1_style))
        elif s.startswith('### '):
            story.append(Paragraph(clean_text(s[4:]), h2_style))
        elif s.startswith('---'):
            # Divider line
            div_table = Table([['']], colWidths=[486])
            div_table.setStyle(TableStyle([
                ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                ('TOPPADDING', (0,0), (-1,-1), 2),
            ]))
            story.append(div_table)
            story.append(Spacer(1, 4))
        elif s.startswith('* ') or s.startswith('- '):
            story.append(Paragraph("• " + clean_text(s[2:]), bullet_style))
        elif re.match(r'^\d+\.\s+', s):
            num = s.split('.')[0]
            rest = s[len(num)+2:]
            story.append(Paragraph(f"<b>{num}.</b> " + clean_text(rest), bullet_style))
        elif s.startswith('> '):
            callout = Paragraph(clean_text(s[2:]), body_style)
            callout_table = Table([[callout]], colWidths=[486])
            callout_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F0F7FF')),
                ('LEFTPADDING', (0,0), (-1,-1), 10),
                ('LINEBEFORE', (0,0), (-1,-1), 3, colors.HexColor('#004D98')),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ]))
            story.append(callout_table)
            story.append(Spacer(1, 4))
        elif s.startswith('|') and s.endswith('|'):
            # We can format simple table rows
            cells = [c.strip() for c in s.split('|')[1:-1]]
            if all(re.match(r'^:?-+:?$', c) for c in cells):
                continue
            p_cells = [Paragraph(clean_text(c), body_style) for c in cells]
            # Simple row
            t_row = Table([p_cells], colWidths=[486/len(cells)]*len(cells))
            t_row.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(t_row)
        else:
            story.append(Paragraph(clean_text(line), body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"SUCCESS: Generated PDF -> {PDF_PATH}")
    print(f"File size: {os.path.getsize(PDF_PATH) / 1024:.1f} KB")

if __name__ == '__main__':
    build_pdf()

