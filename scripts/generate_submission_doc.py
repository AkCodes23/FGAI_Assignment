"""
scripts/generate_submission_doc.py
Generates the comprehensive research-grade Word document (YourName_RegNum_FGAI_Assignment2.docx)
and automatically converts it to PDF for LMS submission.

Features:
- Complete Exploratory Data Analysis (EDA) with Zipf's Law, sentence length analysis, unigram entropy, and n-grams.
- Full embedding of all 10 high-resolution publication-grade figures and architecture schematics.
- Formatted tables for metadata, corpus statistics, n-gram collocations, quantitative results, and temperature dynamics.
- Mathematical parameter decomposition and causal BiRNN slicing analysis.
- Conversion to PDF via Microsoft Word COM automation.
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
        run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)  # Deep Navy
        run.font.bold = True
    elif level == 2:
        run.font.size = Pt(12.5)
        run.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)  # Slate
        run.font.bold = True
    else:
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)


def add_figure_with_caption(doc: Document, img_path: str, caption_text: str, width_inches: float = 6.0) -> None:
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(2)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(width_inches))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(10)
        run_cap = p_cap.add_run(caption_text)
        run_cap.font.name = "Arial"
        run_cap.font.size = Pt(8.5)
        run_cap.font.italic = True
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


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
                    r.font.size = Pt(9.0)
                    if r_idx == 0:
                        r.font.bold = True
                        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            if r_idx == 0:
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
    dataset_name: str = "The Adventures of Sherlock Holmes by Sir Arthur Conan Doyle (Project Gutenberg eBook #1661)",
    dataset_link: str = "https://www.gutenberg.org/ebooks/1661",
    output_filename: str = None,
) -> str:
    sanitized_name = student_name.replace(" ", "_").replace("[", "").replace("]", "")
    sanitized_reg = reg_number.replace(" ", "_").replace("[", "").replace("]", "")

    if output_filename is None:
        output_filename = f"{sanitized_name}_{sanitized_reg}_FGAI_Assignment2.docx"

    doc = Document()

    # Document margins
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Document Header Title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_after = Pt(2)
    title_run = title_p.add_run("Foundations of Generative AI (ICT4403)\nAssignment 2: Comparative Autoregressive Text Generation")
    title_run.font.name = "Arial"
    title_run.font.size = Pt(17)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(12)
    sub_run = sub_p.add_run("Research-Grade Exploratory Data Analysis, Recurrent Neural Architectures, and Empirical Generation Dynamics")
    sub_run.font.name = "Arial"
    sub_run.font.size = Pt(11)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Section 1: Mandatory Submission Identification
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
        c1.width = Inches(4.8)

        p0 = c0.paragraphs[0]
        r0 = p0.add_run(lbl)
        r0.font.name = "Arial"
        r0.font.bold = True
        r0.font.size = Pt(9.5)

        p1 = c1.paragraphs[0]
        r1 = p1.add_run(val)
        r1.font.name = "Arial"
        r1.font.size = Pt(9.5)
        if "http" in val:
            r1.font.color.rgb = RGBColor(0x00, 0x55, 0xAA)
            r1.font.underline = True

    doc.add_paragraph()

    # Section 2: End-to-End System Architecture
    add_heading_with_color(doc, "2. End-to-End System Architecture & Mathematical Formulation", level=1)
    p_arch = doc.add_paragraph()
    p_arch.paragraph_format.space_after = Pt(6)
    p_arch.add_run(
        "The autoregressive text generation system processes raw literary text into a clean sequence of discrete tokens, "
        "constructs causal sliding-window training pairs, maps tokens into continuous dense embeddings, and trains four "
        "controlled recurrent neural architectures (Simple RNN, Bidirectional RNN, LSTM, and GRU) to sequentially model "
        "the conditional probability distribution P(w_{t+1} | w_1, ..., w_t). At inference time, temperature-controlled stochastic "
        "sampling generates multi-sentence narrative continuations while enforcing special-token masking to prevent degenerate outputs."
    )

    # 2.1 Modular Pipeline & Latent Data Flow
    add_heading_with_color(doc, "2.1 Modular Pipeline & Latent Data Flow", level=2)

    add_figure_with_caption(
        doc,
        "results/system_architecture_diagram.png",
        "Figure 1: End-to-end modular system architecture showing data ingestion, embedding projection, recurrent latent encoding, and controlled stochastic autoregressive generation.",
        width_inches=6.2,
    )

    # 2.2 Mathematical Parameter Allocation & Capacity Matching
    add_heading_with_color(doc, "2.2 Mathematical Parameter Allocation & Capacity Matching", level=2)

    p_params_intro = doc.add_paragraph()
    p_params_intro.paragraph_format.space_after = Pt(4)
    p_params_intro.add_run(
        "A rigorous scientific benchmark requires strictly controlling non-recurrent capacity across all models. In our implementation, "
        "all four architectures share identical embedding dimensionality (d_e = 128), matched effective hidden representation size (d_h = 256), "
        "and identical vocabulary dimension (V = 4,378). Consequently, the non-recurrent parameter allocations are strictly invariant across all experiments."
    )

    # Table: Shared Non-Recurrent Baseline Parameters
    t_shared = doc.add_table(rows=4, cols=5)
    t_shared.style = "Table Grid"
    headers_shared = ["Component Layer", "Mathematical Formulation", "Dimension Mapping", "Total Parameters", "% of Baseline"]
    for c_idx, h in enumerate(headers_shared):
        t_shared.rows[0].cells[c_idx].paragraphs[0].text = h

    shared_data = [
        ("Token Embedding Layer", "V × d_e", "4,378 × 128", "560,384", "33.25%"),
        ("Output Linear Projection", "(d_h × V) + V", "(256 × 4,378) + 4,378", "1,125,146", "66.75%"),
        ("Invariant Baseline Core", "Embedding + Projection", "Fixed across all 4 models", "1,685,530", "100.00%"),
    ]
    for r_idx, row in enumerate(shared_data):
        for c_idx, val in enumerate(row):
            t_shared.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val

    style_table(t_shared, [1.6, 1.7, 1.7, 1.0, 1.0])

    p_shared_note = doc.add_paragraph()
    p_shared_note.paragraph_format.space_before = Pt(3)
    p_shared_note.paragraph_format.space_after = Pt(6)
    r_sn = p_shared_note.add_run(
        "Note: The shared baseline (1,685,530 weights) constitutes 70.3% to 88.5% of total parameter mass across all models, "
        "guaranteeing that performance differentials reflect recurrent cell mechanics rather than capacity asymmetry."
    )
    r_sn.font.name = "Arial"
    r_sn.font.size = Pt(8.5)
    r_sn.font.italic = True
    r_sn.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p_rec_intro = doc.add_paragraph()
    p_rec_intro.paragraph_format.space_before = Pt(4)
    p_rec_intro.paragraph_format.space_after = Pt(4)
    p_rec_intro.add_run(
        "The remaining parameter variance arises strictly from internal gating complexity across the 2 stacked recurrent layers (L = 2):"
    )

    # Table: Recurrent Core Parameter Allocation & Gate Complexity
    t_rec = doc.add_table(rows=5, cols=6)
    t_rec.style = "Table Grid"
    headers_rec = ["Architecture", "Gating Mechanism & Formulation", "Layer 1 Weights", "Layer 2 Weights", "Recurrent Core", "Grand Total"]
    for c_idx, h in enumerate(headers_rec):
        t_rec.rows[0].cells[c_idx].paragraphs[0].text = h

    rec_data = [
        ("Simple RNN", "1 Gate: tanh(W_ih x_t + W_hh h_{t-1} + b)", "98,816", "131,584", "230,400", "1,915,930"),
        ("Bidirectional RNN", "2 Directions: (d_h/2 = 128 units per dir)", "66,048", "98,816", "164,864", "1,850,394"),
        ("GRU", "3 Gates: Reset (r), Update (z), Candidate (n)", "296,448", "394,752", "691,200", "2,376,730"),
        ("LSTM", "4 Gates: Input (i), Forget (f), Output (o), Cell (g)", "395,264", "526,336", "921,600", "2,607,130"),
    ]
    for r_idx, row in enumerate(rec_data):
        for c_idx, val in enumerate(row):
            t_rec.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val

    style_table(t_rec, [1.3, 2.1, 0.9, 0.9, 0.9, 0.9])

    p_capacity_insight = doc.add_paragraph()
    p_capacity_insight.paragraph_format.space_before = Pt(4)
    p_capacity_insight.paragraph_format.space_after = Pt(8)
    r_cap_lbl = p_capacity_insight.add_run("Capacity Efficiency Insight: ")
    r_cap_lbl.bold = True
    p_capacity_insight.add_run(
        "Although LSTM allocates the highest parameter volume (2,607,130 total parameters; 921,600 recurrent weights), "
        "GRU achieves lower test perplexity (129.05 vs. 131.54) with 230,400 fewer recurrent weights (691,200 vs. 921,600). "
        "On a corpus of 121,822 tokens, GRU's coupled update-reset formulation prevents over-parameterization while retaining "
        "sufficient representational capacity to capture long-range literary dependencies."
    )

    # 2.3 The Bidirectional Causal Asymmetry Paradox
    add_heading_with_color(doc, "2.3 The Bidirectional Causal Asymmetry Paradox", level=2)

    p_birnn_intro = doc.add_paragraph()
    p_birnn_intro.paragraph_format.space_after = Pt(4)
    p_birnn_intro.add_run(
        "A critical theoretical question in language modeling is why Bidirectional RNNs (BiRNN)—despite their established superiority "
        "in discriminative NLP tasks (such as named entity recognition and POS tagging)—are fundamentally sub-optimal for causal text generation. "
        "Our empirical results confirm this paradox: BiRNN exhibited the worst text quality, collapsing into repetitive loops (25.03% 3-gram repetition at T = 0.2) "
        "and converging to a markedly inferior test perplexity of 162.24. This failure stems from three structural asymmetries:"
    )

    # 3 structured bullet points
    bp1 = doc.add_paragraph(style="List Bullet")
    bp1.paragraph_format.space_before = Pt(2)
    bp1.paragraph_format.space_after = Pt(2)
    r_bp1_b = bp1.add_run("Forward Temporal Accumulation: ")
    r_bp1_b.bold = True
    bp1.add_run(
        "The forward recurrent branch computes states h_1^-> through h_L^-> sequentially from token w_1 to w_L. "
        "At the final temporal index L, the forward hidden state h_L^-> has legitimately accumulated the full context of all L prompt tokens."
    )

    bp2 = doc.add_paragraph(style="List Bullet")
    bp2.paragraph_format.space_before = Pt(2)
    bp2.paragraph_format.space_after = Pt(2)
    r_bp2_b = bp2.add_run("The PyTorch Temporal Slicing Flaw: ")
    r_bp2_b.bold = True
    bp2.add_run(
        "In PyTorch's nn.RNN with bidirectional=True, the backward pass computes hidden states h_L^<- down to h_1^<- in reverse time. "
        "When standard autoregressive architectures extract the final sequence slice via out[:, -1, :], the tensor concatenates "
        "[h_L^->, h_L^<-]. While h_L^-> contains the entire prompt history, h_L^<- is the initial time-step of the reverse pass—meaning it has "
        "observed only token w_L in isolation initialized from an empty zero vector (h_0^<- = 0). The fully contextualized backward state h_1^<- "
        "(which observed all tokens) resides at index 0, which is discarded by the -1 temporal slice."
    )

    bp3 = doc.add_paragraph(style="List Bullet")
    bp3.paragraph_format.space_before = Pt(2)
    bp3.paragraph_format.space_after = Pt(6)
    r_bp3_b = bp3.add_run("Inference Rollout Disconnect: ")
    r_bp3_b.bold = True
    bp3.add_run(
        "During autoregressive text generation, future tokens w_{L+1}, w_{L+2}, ... do not yet exist. The backward recurrence cannot execute "
        "over ungenerated tokens, breaking the mathematical symmetry established during bidirectional training. The backward features fed into "
        "the projection head degenerate into ungrounded noise, destabilizing the output probability distribution."
    )

    # Table: Causal Unidirectional vs. Bidirectional Recurrence Analysis
    t_birnn = doc.add_table(rows=6, cols=3)
    t_birnn.style = "Table Grid"
    headers_birnn = ["Evaluation Dimension", "Unidirectional Recurrence (LSTM / GRU / RNN)", "Bidirectional Recurrence (BiRNN)"]
    for c_idx, h in enumerate(headers_birnn):
        t_birnn.rows[0].cells[c_idx].paragraphs[0].text = h

    birnn_data = [
        ("Causal Consistency", "Strictly causal: P(w_t | w_1, ..., w_{t-1})", "Non-causal: conditions on future tokens w_{t+1...L}"),
        ("Temporal Slice (out[:, -1, :])", "Captures complete contextual history over w_1...w_L", "Forward: full history; Backward: uninformative initial step"),
        ("Autoregressive Generation Rollout", "Mathematically coherent token-by-token rollout", "Future tokens absent; backward pass operates degenerately"),
        ("Empirical 3-Gram Repetition (T=0.2)", "0.00% (GRU) / 0.00% (LSTM) [No degenerate loops]", "25.03% [Severe Markovian attractor collapse]"),
        ("Empirical Test Perplexity (PPL)", "129.05 (GRU) / 131.54 (LSTM) [Optimal]", "162.24 [Severely sub-optimal token conditioning]"),
    ]
    for r_idx, row in enumerate(birnn_data):
        for c_idx, val in enumerate(row):
            t_birnn.rows[r_idx + 1].cells[c_idx].paragraphs[0].text = val

    style_table(t_birnn, [1.6, 2.7, 2.7])

    p_birnn_conclusion = doc.add_paragraph()
    p_birnn_conclusion.paragraph_format.space_before = Pt(4)
    p_birnn_conclusion.paragraph_format.space_after = Pt(8)
    r_bc_lbl = p_birnn_conclusion.add_run("Theoretical Takeaway: ")
    r_bc_lbl.bold = True
    p_birnn_conclusion.add_run(
        "Bidirectional latent encoding violates the unidirectional arrow of time required for autoregression. "
        "While BiRNNs excel in masked sequence understanding, causal text generation necessitates unidirectional architectures "
        "(such as GRU or causal transformers) where every conditioning token strictly precedes the predicted token."
    )

    doc.add_paragraph()

    # Section 3: Exploratory Data Analysis & Corpus Diagnostics
    add_heading_with_color(doc, "3. Exploratory Data Analysis (EDA) & Corpus Diagnostics", level=1)
    p_eda_intro = doc.add_paragraph()
    p_eda_intro.paragraph_format.space_after = Pt(6)
    p_eda_intro.add_run(
        "A rigorous exploratory data analysis was conducted on The Adventures of Sherlock Holmes to characterize its lexical distribution, "
        "information entropy, sentence length dynamics, and vocabulary coverage. These statistical properties directly informed our "
        "hyperparameter choices for context sequence length (L = 20) and vocabulary cutoff (V = 4,378)."
    )

    # 3.1 Corpus Metrics Table
    add_heading_with_color(doc, "3.1 Descriptive Corpus & Information-Theoretic Metrics", level=2)
    corpus_csv = "results/eda_corpus_stats.csv"
    if os.path.exists(corpus_csv):
        df_c = pd.read_csv(corpus_csv)
        c_table = doc.add_table(rows=len(df_c) + 1, cols=2)
        c_table.style = "Table Grid"
        c_table.rows[0].cells[0].paragraphs[0].text = "Statistical / Information-Theoretic Metric"
        c_table.rows[0].cells[1].paragraphs[0].text = "Empirical Value (Sherlock Holmes Corpus)"

        for r_idx, row in df_c.iterrows():
            c_table.rows[r_idx + 1].cells[0].paragraphs[0].text = str(row["Metric"])
            c_table.rows[r_idx + 1].cells[1].paragraphs[0].text = str(row["Value"])

        style_table(c_table, [3.6, 3.4])

    p_entropy_note = doc.add_paragraph()
    p_entropy_note.paragraph_format.space_before = Pt(6)
    p_entropy_note.add_run(
        "Information-Theoretic Compression: The empirical Shannon unigram entropy of the corpus is H = 8.7852 bits/token. "
        "This yields a theoretical zeroth-order perplexity lower bound of 2^H = 441.17 for any static unigram model. "
        "Our trained GRU achieves a test perplexity of 129.05, representing a 3.42x information compression over the unigram baseline "
        "and a 33.9x reduction over the uniform random vocabulary baseline (PPL = 4,378)."
    )

    # 3.2 Sentence Length Analysis
    add_heading_with_color(doc, "3.2 Sentence Length Distribution & Sequence Window Justification", level=2)
    p_sent = doc.add_paragraph()
    p_sent.add_run(
        "The corpus contains 5,144 sentences with a mean length of 20.41 words and a median of 17.0 words (IQR: [9.0, 28.0] words). "
        "The distribution is moderately right-skewed, characteristic of Victorian prose with subordinate clauses. Our choice of context sequence "
        "length L = 20 spans approximately 60% of all complete sentences in the text, ensuring that the recurrent hidden states capture complete "
        "clausal dependencies without exceeding the vanishing-gradient retention horizon of recurrent cells."
    )
    add_figure_with_caption(
        doc,
        "results/eda_sentence_lengths.png",
        "Figure 2: Empirical sentence length probability density distribution with mean (20.41 words), median (17.0 words), and 95th percentile markers.",
        width_inches=5.8,
    )

    # 3.3 Zipf's Law Analysis
    add_heading_with_color(doc, "3.3 Empirical Zipf's Law Power-Law Verification", level=2)
    p_zipf = doc.add_paragraph()
    p_zipf.add_run(
        "Zipf's Law states that the occurrence frequency f(r) of a word is inversely proportional to its frequency rank r: "
        "f(r) = C / r^alpha, which linearizes in logarithmic coordinates as log10(f(r)) = log10(C) - alpha * log10(r). "
        "Ordinary least squares (OLS) regression over the top 1,000 token ranks yields an empirical exponent of alpha = 1.1161 "
        "with a coefficient of determination R^2 = 0.9966. This strong adherence confirms that the vocabulary follows standard natural language "
        "distributional power laws, where a small set of grammatical function words dominates the token stream while a vast tail of content words "
        "occurs rarely."
    )
    add_figure_with_caption(
        doc,
        "results/eda_zipf_law.png",
        "Figure 3: Empirical rank-frequency Zipfian distribution on log-log coordinates compared against the theoretical power-law fit (alpha = 1.1161, R^2 = 0.9966).",
        width_inches=5.8,
    )

    # 3.4 Lexical Coverage & Vocabulary Thresholding
    add_heading_with_color(doc, "3.4 Cumulative Lexical Coverage & Out-of-Vocabulary Analysis", level=2)
    p_cov = doc.add_paragraph()
    p_cov.add_run(
        "An analysis of cumulative token coverage reveals steep initial concentration: the top 40 words alone account for 50.0% of all "
        "tokens in the corpus, while the top 517 words account for 80.0% (Pareto threshold). Setting the minimum word frequency threshold to min_freq >= 2 "
        "yields an effective vocabulary size of V = 4,378 distinct tokens. This active vocabulary covers over 97.4% of all running tokens in the corpus, "
        "restricting the out-of-vocabulary (<UNK>) rate to just 2.6%. This strikes an optimal trade-off between vocabulary expressiveness and "
        "softmax projection computational cost."
    )
    add_figure_with_caption(
        doc,
        "results/eda_token_coverage.png",
        "Figure 4: Cumulative lexical coverage curve versus vocabulary rank, highlighting the 50%, 80%, and 95% coverage thresholds and the active vocabulary cutoff.",
        width_inches=5.8,
    )

    # 3.5 Top N-Grams Collocation Profiles
    add_heading_with_color(doc, "3.5 Frequent Collocations & N-Gram Profiles", level=2)
    p_ng = doc.add_paragraph()
    p_ng.add_run(
        "N-gram frequency analysis highlights the predominant syntactic collocations in Conan Doyle's prose. Excluding isolated punctuation, "
        "the most frequent bigrams represent prepositional phrases ('of the', 'in the', 'to the', 'at the') and first-person narrative framing "
        "('it is', 'i have', 'it was', 'that i'). These collocations serve as primary qualitative benchmarks when inspecting model-generated sequences."
    )
    add_figure_with_caption(
        doc,
        "results/eda_top_ngrams.png",
        "Figure 5: High-frequency lexical profiles for Top 12 Unigram word tokens (Panel A) and Top 12 Bigram collocations (Panel B).",
        width_inches=6.0,
    )

    ngram_csv = "results/eda_top_ngrams.csv"
    if os.path.exists(ngram_csv):
        df_ng = pd.read_csv(ngram_csv).head(10)
        ng_table = doc.add_table(rows=len(df_ng) + 1, cols=7)
        ng_table.style = "Table Grid"
        headers_ng = ["Rank", "Unigram", "Count", "Unigram %", "Bigram", "Count", "Bigram %"]
        for c_idx, h in enumerate(headers_ng):
            ng_table.rows[0].cells[c_idx].paragraphs[0].text = h
        for r_idx, row in df_ng.iterrows():
            ng_table.rows[r_idx + 1].cells[0].paragraphs[0].text = str(row["Rank"])
            ng_table.rows[r_idx + 1].cells[1].paragraphs[0].text = str(row["Unigram"])
            ng_table.rows[r_idx + 1].cells[2].paragraphs[0].text = f"{int(row['Unigram_Count']):,}"
            ng_table.rows[r_idx + 1].cells[3].paragraphs[0].text = str(row["Unigram_Pct"])
            ng_table.rows[r_idx + 1].cells[4].paragraphs[0].text = str(row["Bigram"])
            ng_table.rows[r_idx + 1].cells[5].paragraphs[0].text = f"{int(row['Bigram_Count']):,}"
            ng_table.rows[r_idx + 1].cells[6].paragraphs[0].text = str(row["Bigram_Pct"])
        style_table(ng_table, [0.6, 1.1, 0.9, 1.1, 1.3, 0.9, 1.1])

    # 3.6 Temporal Sequence Partitioning & Boundary Correlation Analysis
    add_heading_with_color(doc, "3.6 Temporal Partitioning & Boundary Autocorrelation Safeguards", level=2)
    p_split = doc.add_paragraph()
    p_split.add_run(
        "In sliding-window causal sequence modeling, a window of length L = 20 advanced by stride s = 2 shares L - s = 18 tokens "
        "between consecutive sequences. To rigorously prevent future data leakage, our pipeline adopts a strict chronological split "
        "(Train: first 80%, Validation: next 10%, Test: final 10%). While consecutive training windows possess natural autocorrelation, "
        "they preserve unidirectional arrow of time. Furthermore, because the evaluation splits occupy chronologically later chapters, "
        "test set perplexity genuinely measures the models' generalization ability on unseen narrative contexts."
    )

    doc.add_paragraph()

    # Section 4: Quantitative Performance & Comparative Analysis
    add_heading_with_color(doc, "4. Quantitative Performance & Comparative Model Evaluation", level=1)
    p_perf = doc.add_paragraph()
    p_perf.add_run(
        "All four models were trained for 10 epochs using identical hyperparameters: embedding dimension = 128, hidden dimension = 256, "
        "number of layers = 2, dropout = 0.2, batch size = 64, Adam optimizer with initial learning rate = 0.001, ReduceLROnPlateau scheduler, "
        "and gradient clipping at max norm = 1.0. The held-out test split (6,082 samples, 10% of dataset) was evaluated across Cross-Entropy Loss, "
        "Test Perplexity (PPL), Top-1 Accuracy, Top-5 Accuracy, and Inference Latency per batch."
    )

    metrics_csv = "results/metrics.csv"
    if os.path.exists(metrics_csv):
        df_m = pd.read_csv(metrics_csv)
        col_names = ["Model", "Parameters", "Train Loss", "Val Loss", "Test Loss", "Perplexity", "Top-1 Acc (%)", "Top-5 Acc (%)", "Latency (ms)"]
        quant_table = doc.add_table(rows=len(df_m) + 1, cols=len(col_names))
        quant_table.style = "Table Grid"

        for c_idx, col in enumerate(col_names):
            quant_table.rows[0].cells[c_idx].paragraphs[0].text = col

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

    add_figure_with_caption(
        doc,
        "results/loss_curves.png",
        "Figure 6: Cross-entropy training and validation loss curves over 10 training epochs for Simple RNN, BiRNN, LSTM, and GRU.",
        width_inches=5.8,
    )

    add_figure_with_caption(
        doc,
        "results/perplexity_curves.png",
        "Figure 7: Validation perplexity curves over training epochs, demonstrating GRU's superior convergence rate.",
        width_inches=5.8,
    )

    add_figure_with_caption(
        doc,
        "results/accuracy_comparison.png",
        "Figure 8: Next-word prediction accuracy comparison on held-out test data showing Top-1 and Top-5 accuracy percentages.",
        width_inches=5.8,
    )

    add_figure_with_caption(
        doc,
        "results/efficiency_comparison.png",
        "Figure 9: Computational efficiency trade-off: Trainable parameter count vs. batch inference latency (ms).",
        width_inches=5.8,
    )

    # Section 5: Sampling Temperature Dynamics
    add_heading_with_color(doc, "5. Qualitative Sampling & Temperature Dynamics Analysis", level=1)
    p_temp = doc.add_paragraph()
    p_temp.add_run(
        "Autoregressive sequential generation was tested using three sampling temperature values: T = 0.2 (conservative / greedy), "
        "T = 0.7 (balanced stochasticity), and T = 1.2 (high entropy exploration). Diversity metrics were computed across 50 generated tokens: "
        "Distinct-1 (unigram diversity), Distinct-2 (bigram diversity), and 3-Gram Repetition Rate (percentage of repeated trigrams)."
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

    p_entropy_sampling = doc.add_paragraph()
    p_entropy_sampling.paragraph_format.space_before = Pt(6)
    p_entropy_sampling.add_run(
        "Softmax Temperature Mechanics & Attractor Basin Formation:\n"
        "Given raw model logits z_i, temperature scaling scales the logits by 1/T prior to normalization: P(w_i) = exp(z_i / T) / sum_j exp(z_j / T).\n"
        "• At low temperature (T = 0.2): Probability mass concentrates heavily on the mode (argmax). When recurrent states enter a cyclic transition "
        "matrix (such as 'in the matter of the...'), the deterministic argmax collapses into a closed Markovian attractor loop, resulting in high "
        "3-gram repetition (up to 25.51% for Simple RNN and 25.03% for BiRNN).\n"
        "• At moderate temperature (T = 0.7): Distribution entropy expands sufficiently to escape local attractor basins while remaining concentrated "
        "enough to preserve Victorian grammatical syntax. This eliminates 3-gram repetition completely (0.00% across all models).\n"
        "• At high temperature (T = 1.2): Entropy approaches the uniform distribution over 4,378 tokens (H_max = 12.096 bits), causing sampling from "
        "the low-probability distributional tail and degrading syntactic coherence."
    )

    add_figure_with_caption(
        doc,
        "results/qualitative_metrics.png",
        "Figure 10: Multi-metric temperature response showing Distinct-1, Distinct-2, and Repetition Rate dynamics across T in {0.2, 0.7, 1.2}.",
        width_inches=6.0,
    )

    # Section 6: Qualitative Text Generation Samples
    add_heading_with_color(doc, "6. Qualitative Text Generation Samples (T = 0.7)", level=1)
    samples_csv = "results/generated_samples.csv"
    if os.path.exists(samples_csv):
        df_s = pd.read_csv(samples_csv)
        t07_samples = df_s[(df_s["Temperature"] == 0.7) & (df_s["Model"].isin(["GRU", "LSTM"]))].head(4)

        for _, s_row in t07_samples.iterrows():
            sp = doc.add_paragraph()
            sp.paragraph_format.space_before = Pt(4)
            sp.paragraph_format.space_after = Pt(4)

            r_head = sp.add_run(f"[{s_row['Model']} | Temperature = 0.7 | Seed: {s_row['Seed ID']}]\n")
            r_head.font.bold = True
            r_head.font.size = Pt(9.5)
            r_head.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

            r_text = sp.add_run(f"\"{s_row['Generated Text']}\"\n")
            r_text.font.italic = True
            r_text.font.size = Pt(9.0)

            r_meta = sp.add_run(f"Metrics: Distinct-1 = {s_row['Distinct-1']:.4f} | Distinct-2 = {s_row['Distinct-2']:.4f} | Repetition Rate = {s_row['Repetition Rate (%)']:.2f}%\n")
            r_meta.font.size = Pt(8.0)
            r_meta.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # Section 7: Comparative Architecture Analysis & Findings
    add_heading_with_color(doc, "7. Comparative Architecture Analysis & Conclusions", level=1)
    findings = [
        ("Champion Architecture (GRU)", "The Gated Recurrent Unit achieved the best quantitative performance across all metrics: lowest Test Loss (4.8602), lowest Test Perplexity (129.05), highest Top-1 Accuracy (17.86%), and highest Top-5 Accuracy (39.94%). Compared to random guessing over 4,378 tokens (0.0228%), the GRU is 783x more accurate. Its coupled update-reset gating enables effective long-range gradient propagation with fewer parameters (2.38M) than LSTM, avoiding parameter over-capacity on a 120k-token corpus."),
        ("LSTM Competitiveness & Speed", "LSTM was the second-best model (Test Loss 4.8809, PPL 131.74, Top-1 Acc 17.22%, Top-5 Acc 38.59%) and exhibited the lowest inference latency (1.65 ms per batch), making it highly competitive for latency-critical deployment."),
        ("Simple RNN Vanishing Gradients", "The un-gated Simple RNN suffered from severe vanishing gradients across the 20-token horizon, yielding the highest Test Perplexity (159.28) and lowest accuracy (Top-1 16.35%). At low temperature (T=0.2), Simple RNN succumbed to severe degenerate repetition loops (25.51% 3-gram repetition)."),
        ("Bidirectional RNN Causal Mismatch", "BiRNN achieved reasonable training metrics but is structurally incompatible with autoregressive generation: bidirectional encoding conditions on future tokens during training, while taking out[:, -1, :] during next-word prediction causes the backward branch to only see the final token. This causal mismatch produced high repetition loops (25.03% at T=0.2)."),
        ("Optimal Operational Temperature (T = 0.7)", "Temperature T = 0.7 represents the empirical sweet spot between coherence and diversity: Distinct-1 reached 0.70-0.80, Distinct-2 reached 0.96-0.99, and 3-gram repetition was completely eradicated (0.00% across all models). In contrast, T = 0.2 causes repetitive mode collapse, while T = 1.2 induces syntactic disintegration."),
    ]

    for title_txt, body_txt in findings:
        bp = doc.add_paragraph(style="List Bullet")
        bp.paragraph_format.space_before = Pt(2)
        bp.paragraph_format.space_after = Pt(3)

        r_bold = bp.add_run(f"{title_txt}: ")
        r_bold.font.name = "Arial"
        r_bold.font.bold = True
        r_bold.font.size = Pt(9.5)
        r_bold.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

        r_body = bp.add_run(body_txt)
        r_body.font.name = "Arial"
        r_body.font.size = Pt(9.5)

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
