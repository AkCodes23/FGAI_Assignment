"""
scripts/generate_submission_doc.py
Generates the required Step 3 Word document (YourName_RegNum_FGAI_Assignment2.docx)
and automatically converts it to PDF (Step 4) for LMS upload.

Context Acquisition:
- Responsibility: Creation of submission .docx and conversion to .pdf.
- Upstream: python-docx, win32com.client (Microsoft Word automation).
- Downstream: Project root directory submission deliverables.
- Data Ownership: YourName_RegNum_FGAI_Assignment2.docx, YourName_RegNum_FGAI_Assignment2.pdf.
- Runtime Constraints: < 5s execution time.
- Failure Expectations: Graceful warning if Microsoft Word is not installed for COM automation.
"""

import argparse
import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def create_submission_docx(
    student_name: str = "[Student Full Name]",
    reg_number: str = "[Registration Number]",
    repo_url: str = "https://github.com/your-username/fgai_assignment",
    assignment_title: str = "Foundations of Generative AI (ICT4403) - Assignment 2",
    system_title: str = "Autoregressive Text Generation System using Simple RNN, Bidirectional RNN, LSTM, and GRU",
    dataset_name: str = "The Adventures of Sherlock Holmes by Arthur Conan Doyle",
    dataset_link: str = "https://www.gutenberg.org/files/1661/1661-0.txt",
    output_filename: str = None,
) -> str:
    sanitized_name = student_name.replace(" ", "_").replace("[", "").replace("]", "")
    sanitized_reg = reg_number.replace(" ", "_").replace("[", "").replace("]", "")

    if output_filename is None:
        output_filename = f"{sanitized_name}_{sanitized_reg}_FGAI_Assignment2.docx"

    doc = Document()

    # Document margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("Foundations of Generative AI (ICT4403)\nAssignment 2 Submission")
    title_run.font.name = "Arial"
    title_run.font.size = Pt(20)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)  # Navy Blue

    doc.add_paragraph()  # Spacing

    # Subtitle / Header Box
    desc_p = doc.add_paragraph()
    desc_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    desc_run = desc_p.add_run("Submission Details & GitHub Repository Access Link")
    desc_run.font.name = "Arial"
    desc_run.font.size = Pt(13)
    desc_run.font.italic = True
    desc_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph()

    # Table with details
    table = doc.add_table(rows=6, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Shading Accent 1" if "Light Shading Accent 1" in doc.styles else "Table Grid"

    fields = [
        ("Student Name", student_name),
        ("Registration Number", reg_number),
        ("Assignment Title", assignment_title),
        ("System Title", system_title),
        ("Dataset Name & Link", f"{dataset_name}\nURL: {dataset_link}"),
        ("GitHub Repository Link", repo_url),
    ]

    for idx, (label, val) in enumerate(fields):
        row = table.rows[idx]
        cell_label = row.cells[0]
        cell_val = row.cells[1]

        cell_label.width = Inches(2.2)
        cell_val.width = Inches(4.3)

        p_label = cell_label.paragraphs[0]
        r_label = p_label.add_run(label)
        r_label.font.name = "Arial"
        r_label.font.bold = True
        r_label.font.size = Pt(11)

        p_val = cell_val.paragraphs[0]
        r_val = p_val.add_run(val)
        r_val.font.name = "Arial"
        r_val.font.size = Pt(11)
        if "http" in val:
            r_val.font.color.rgb = RGBColor(0x00, 0x55, 0xAA)
            r_val.font.underline = True

    doc.add_paragraph()
    doc.add_paragraph()

    # Architecture Summary Notes
    h2 = doc.add_heading(level=2)
    h2_run = h2.add_run("System Overview & Implementation Summary")
    h2_run.font.name = "Arial"
    h2_run.font.size = Pt(14)
    h2_run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    bullets = [
        "Models Implemented: Simple RNN (Elman), Bidirectional RNN, LSTM, and GRU under matched capacity (embedding dim: 128, hidden dim: 256).",
        "Evaluation Measures: Cross-Entropy Loss, Perplexity (PPL), Top-1 Accuracy, Top-5 Accuracy, Training Time, and Inference Latency.",
        "Qualitative Evaluation: Coherence, Lexical Diversity (Distinct-1 & Distinct-2), and 3-Gram Repetition Rate across sampling temperatures T = 0.2, 0.7, 1.2.",
        "Full Reproducibility: Code, checkpoints, logs, plots, tests, and Jupyter Notebook (.ipynb) are organized in the GitHub repository.",
    ]

    for b in bullets:
        bp = doc.add_paragraph(style="List Bullet")
        brun = bp.add_run(b)
        brun.font.name = "Arial"
        brun.font.size = Pt(10.5)

    doc.save(output_filename)
    print(f"[Word Doc] Created successfully: {output_filename}")
    return output_filename


def convert_docx_to_pdf(docx_path: str, pdf_path: str = None) -> str:
    """
    Converts .docx to .pdf using Windows COM (Microsoft Word).
    """
    if pdf_path is None:
        pdf_path = os.path.splitext(docx_path)[0] + ".pdf"

    abs_docx = os.path.abspath(docx_path)
    abs_pdf = os.path.abspath(pdf_path)

    try:
        import win32com.client
        import pythoncom

        pythoncom.CoInitialize()
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False

        doc = word.Documents.Open(abs_docx)
        # 17 corresponds to wdFormatPDF
        doc.SaveAs(abs_pdf, FileFormat=17)
        doc.Close()
        word.Quit()
        print(f"[PDF Doc] Converted successfully: {abs_pdf}")
        return abs_pdf
    except Exception as e:
        print(f"[PDF Conversion Note] Automatic COM conversion failed: {e}")
        print(f"Please open '{docx_path}' in Microsoft Word and use File -> Save As -> PDF.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Assignment 2 Submission DOCX & PDF")
    parser.add_argument("--name", type=str, default="Student_Name", help="Student Full Name")
    parser.add_argument("--reg_num", type=str, default="REG_123456", help="Student Registration Number")
    parser.add_argument("--repo_url", type=str, default="https://github.com/your-username/fgai_assignment", help="GitHub Repo URL")
    parser.add_argument("--output_docx", type=str, default=None, help="Output .docx file path")

    args = parser.parse_args()

    docx_file = create_submission_docx(
        student_name=args.name,
        reg_number=args.reg_num,
        repo_url=args.repo_url,
        output_filename=args.output_docx,
    )
    convert_docx_to_pdf(docx_file)
