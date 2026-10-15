"""
scripts/generate_submission_doc.py
Generates the comprehensive Step 3 Word document (YourName_RegNum_FGAI_Assignment2.docx)
and automatically converts it to PDF (Step 4) for LMS upload.

Context Acquisition:
- Responsibility: Creation of submission .docx and conversion to .pdf with complete tables, metrics, and analysis.
- Upstream: python-docx, win32com.client (Microsoft Word automation), results/ CSV files.
- Downstream: Project root directory submission deliverables.
- Data Ownership: Varanasi_Naga_Akhil_230953496_FGAI_Assignment2.docx, Varanasi_Naga_Akhil_230953496_FGAI_Assignment2.pdf.
- Runtime Constraints: < 10s execution time.
- Failure Expectations: Graceful warning if Microsoft Word is not installed for COM automation.
"""

import argparse
import os
import sys
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def add_heading_with_color(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(level=level)
    run = h.add_run(text)
    run.font.name = "Arial"
    if level == 1:
        run.font.size = Pt(15)
        run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)  # Navy
        run.font.bold = True
    elif level == 2:
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)  # Slate
        run.font.bold = True
    else:
        run.font.size = Pt(11.5)
        run.font.bold = True


def style_table(table, col_widths=None) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            if col_widths and c_idx < len(col_widths):
                cell.width = Inches(col_widths[c_idx])
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(3)
                p.paragraph_format.space_after = Pt(3)
                for r in p.runs:
                    r.font.name = "Arial"
                    r.font.size = Pt(9.5)
                    if r_idx == 0:
                        r.font.bold = True
                        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            if r_idx == 0:
                # Header row shading via XML
                from docx.oxml import parse_xml
                from docx.oxml.ns import nsdecls
                shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1B365D"/>')
                cell._tc.get_or_add_tcPr().append(shading)


def create_submission_docx(
    student_name: str = "Varanasi Naga Akhil",
    reg_number: str = "230953496",
    repo_url: str = "https://github.com/AkCodes23/FGAI_Assignment.git",
    assignment_title: str = "Foundations of Generative AI (ICT4403) - Assignment 2",
    system_title: str = "Autoregressive Text Generation System using Simple RNN, Bidirectional RNN, LSTM, and GRU",
    dataset_name: str = "The Adventures of Sherlock Holmes by Sir Arthur Conan Doyle (Project Gutenberg)",
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
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Document Header Title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("Foundations of Generative AI (ICT4403)\nAssignment 2: Comparative Autoregressive Text Generation")
    title_run.font.name = "Arial"
    title_run.font.size = Pt(18)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_p.add_run("System Architecture, Empirical Evaluation, and Temperature Dynamics")
    sub_run.font.name = "Arial"
    sub_run.font.size = Pt(12)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph()

    # Mandatory Submission Metadata Table
    add_heading_with_color(doc, "1. Mandatory Submission Identification", level=1)
    meta_table = doc.add_table(rows=6, cols=2)
    meta_table.style = "Table Grid"

    fields = [
        ("Student Full Name", student_name),
        ("Registration Number", reg_number),
        ("Course & Assignment", assignment_title),
        ("System Architecture Title", system_title),
        ("Dataset Name & Source URL", f"{dataset_name}\nURL: {dataset_link}"),
        ("GitHub Repository Link", repo_url),
    ]

    for idx, (lbl, val) in enumerate(fields):
        row = meta_table.rows[idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.2)
        c1.width = Inches(4.6)

        p0 = c0.paragraphs[0]
        r0 = p0.add_run(lbl)
        r0.font.name = "Arial"
        r0.font.bold = True
        r0.font.size = Pt(10)

        p1 = c1.paragraphs[0]
        r1 = p1.add_run(val)
        r1.font.name = "Arial"
        r1.font.size = Pt(10)
        if "http" in val:
            r1.font.color.rgb = RGBColor(0x00, 0x55, 0xAA)
            r1.font.underline = True

    doc.add_paragraph()

    # Section 2: Architecture Specifications
    add_heading_with_color(doc, "2. System Overview & Comparable Experimental Setup", level=1)
    p_arch = doc.add_paragraph()
    p_arch.paragraph_format.space_after = Pt(6)
    p_arch.add_run(
        "To perform a strictly scientific evaluation, all four recurrent architectures were matched in parameter capacity "
        "and trained under identical conditions on the causal word-level language modeling task (context sequence length L=25 -> target word w_L+1). "
        "The vocabulary comprises 4,378 distinct tokens extracted from Arthur Conan Doyle's The Adventures of Sherlock Holmes. "
        "Shared hyperparameters include embedding dimension = 128, hidden dimension = 256, dropout = 0.2, Adam optimizer (lr=0.001), "
        "batch size = 64, and gradient norm clipping = 5.0."
    )

    doc.add_paragraph()

    # Section 3: Quantitative Performance
    add_heading_with_color(doc, "3. Quantitative Performance & Operational Efficiency", level=1)

    metrics_csv = "results/metrics.csv"
    if os.path.exists(metrics_csv):
        df_m = pd.read_csv(metrics_csv)
        col_names = ["Model", "Parameters", "Train Loss", "Val Loss", "Test Loss", "Perplexity", "Top-1 Acc (%)", "Top-5 Acc (%)", "Latency (ms)"]
        quant_table = doc.add_table(rows=len(df_m) + 1, cols=len(col_names))
        quant_table.style = "Table Grid"

        # Headers
        for c_idx, col in enumerate(col_names):
            quant_table.rows[0].cells[c_idx].paragraphs[0].text = col

        # Data rows
        for r_idx, row in df_m.iterrows():
            quant_table.rows[r_idx + 1].cells[0].paragraphs[0].text = str(row["Model"])
            quant_table.rows[r_idx + 1].cells[1].paragraphs[0].text = f"{int(row['Parameters']):,}"
            quant_table.rows[r_idx + 1].cells[2].paragraphs[0].text = f"{float(row['Train Loss']):.4f}"
            quant_table.rows[r_idx + 1].cells[3].paragraphs[0].text = f"{float(row['Val Loss']):.4f}"
            quant_table.rows[r_idx + 1].cells[4].paragraphs[0].text = f"{float(row['Test Loss']):.4f}"
            quant_table.rows[r_idx + 1].cells[5].paragraphs[0].text = f"{float(row['Test Perplexity']):.2f}"
            quant_table.rows[r_idx + 1].cells[6].paragraphs[0].text = f"{float(row['Top-1 Accuracy (%)']):.2f}%"
            quant_table.rows[r_idx + 1].cells[7].paragraphs[0].text = f"{float(row['Top-5 Accuracy (%)']):.2f}%"
            quant_table.rows[r_idx + 1].cells[8].paragraphs[0].text = f"{float(row['Inference Latency (ms)']):.2f}"

        style_table(quant_table, [1.3, 0.9, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8])

    doc.add_paragraph()

    # Section 4: Qualitative Sampling & Diversity Analysis
    add_heading_with_color(doc, "4. Qualitative Sampling & Temperature Dynamics Analysis", level=1)
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_after = Pt(6)
    p_div.add_run(
        "Autoregressive generation was probed across temperatures T in {0.2, 0.7, 1.2} across 3 fixed seed prompts. "
        "Distinct-1 (unigram diversity), Distinct-2 (bigram diversity), and 3-Gram Repetition Rate (%) evaluate the transition "
        "from greedy exploitation to diverse exploration:"
    )

    div_csv = "results/diversity_metrics.csv"
    if os.path.exists(div_csv):
        df_d = pd.read_csv(div_csv)
        div_table = doc.add_table(rows=len(df_d) + 1, cols=5)
        div_table.style = "Table Grid"

        headers = ["Model", "Temperature (T)", "Distinct-1", "Distinct-2", "3-Gram Repetition Rate (%)"]
        for c_idx, h in enumerate(headers):
            div_table.rows[0].cells[c_idx].paragraphs[0].text = h

        for r_idx, row in df_d.iterrows():
            div_table.rows[r_idx + 1].cells[0].paragraphs[0].text = str(row["Model"])
            div_table.rows[r_idx + 1].cells[1].paragraphs[0].text = f"T = {float(row['Temperature']):.1f}"
            div_table.rows[r_idx + 1].cells[2].paragraphs[0].text = f"{float(row['Distinct-1']):.4f}"
            div_table.rows[r_idx + 1].cells[3].paragraphs[0].text = f"{float(row['Distinct-2']):.4f}"
            div_table.rows[r_idx + 1].cells[4].paragraphs[0].text = f"{float(row['Repetition Rate (%)']):.2f}%"

        style_table(div_table, [1.5, 1.2, 1.1, 1.1, 1.8])

    doc.add_paragraph()

    # Section 5: Text Samples
    add_heading_with_color(doc, "5. Qualitative Text Generation Samples (T = 0.7)", level=1)

    samples_csv = "results/generated_samples.csv"
    if os.path.exists(samples_csv):
        df_s = pd.read_csv(samples_csv)
        # Select representative samples for GRU and LSTM at T=0.7
        t07_samples = df_s[(df_s["Temperature"] == 0.7) & (df_s["Model"].isin(["GRU", "LSTM"]))].head(4)

        for _, s_row in t07_samples.iterrows():
            sp = doc.add_paragraph()
            r_head = sp.add_run(f"[{s_row['Model']} | T=0.7 | {s_row['Seed ID']}]\n")
            r_head.font.bold = True
            r_head.font.size = Pt(10)
            r_head.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

            r_text = sp.add_run(f"\"{s_row['Generated Text']}\"\n")
            r_text.font.italic = True
            r_text.font.size = Pt(9.5)

            r_meta = sp.add_run(f"Distinct-1: {s_row['Distinct-1']:.4f} | Distinct-2: {s_row['Distinct-2']:.4f} | Repetition Rate: {s_row['Repetition Rate (%)']:.2f}%\n")
            r_meta.font.size = Pt(8.5)
            r_meta.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    doc.add_paragraph()

    # Section 6: Architectural Conclusions
    add_heading_with_color(doc, "6. Comparative Architecture Analysis & Findings", level=1)
    bullets = [
        "Superiority of GRU: The GRU achieved the highest modeling fidelity (Test PPL 131.77, Top-1 Acc 18.05%, Top-5 Acc 38.86%). Its streamlined coupled update/reset gating enables robust gradient transmission over sequence horizons without the parameter overhead of an independent memory cell.",
        "LSTM Competitiveness & Speed: LSTM finished a close second (Test PPL 135.06, Top-1 Acc 16.47%) and exhibited the lowest inference latency (1.85 ms/batch).",
        "Simple RNN Limitations: Simple RNN suffered from severe vanishing gradients, resulting in the highest test perplexity (158.44) and a 22.12% repetition rate at T=0.2 due to mode collapse.",
        "Bidirectional RNN Causal Mismatch: While BiRNN achieved reasonable training loss, it is fundamentally ill-suited for pure autoregressive causal text generation because backward recurrent passes look forward during training, creating a teacher-forcing discrepancy during sequential inference.",
        "Temperature Sweet Spot: T = 0.7 demonstrated the ideal operational balance between lexical diversity (Distinct-2 = 0.98) and syntactic coherence, completely eliminating phrase looping (0.00% 3-gram repetition) for GRU and LSTM.",
    ]

    for b in bullets:
        bp = doc.add_paragraph(style="List Bullet")
        b_parts = b.split(":", 1)
        r_bold = bp.add_run(b_parts[0] + ":")
        r_bold.font.name = "Arial"
        r_bold.font.bold = True
        r_bold.font.size = Pt(10)

        r_body = bp.add_run(b_parts[1])
        r_body.font.name = "Arial"
        r_body.font.size = Pt(10)

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
        doc.SaveAs(abs_pdf, FileFormat=17)  # 17 = wdFormatPDF
        doc.Close()
        word.Quit()
        print(f"[PDF Doc] Converted successfully: {abs_pdf}")
        return abs_pdf
    except Exception as e:
        print(f"[PDF Conversion Note] Automatic COM conversion failed: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Assignment 2 Submission DOCX & PDF")
    parser.add_argument("--name", type=str, default="Varanasi Naga Akhil", help="Student Full Name")
    parser.add_argument("--reg_num", type=str, default="230953496", help="Student Registration Number")
    parser.add_argument("--repo_url", type=str, default="https://github.com/AkCodes23/FGAI_Assignment.git", help="GitHub Repo URL")
    parser.add_argument("--output_docx", type=str, default=None, help="Output .docx file path")

    args = parser.parse_args()

    docx_file = create_submission_docx(
        student_name=args.name,
        reg_number=args.reg_num,
        repo_url=args.repo_url,
        output_filename=args.output_docx,
    )
    convert_docx_to_pdf(docx_file)
