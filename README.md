# Foundations of Generative AI (ICT4403) — Assignment 2
## Autoregressive Text-Generation System Using Simple RNN, Bidirectional RNN, LSTM, and GRU

---

### Project Metadata
- **Course Title:** Foundations of Generative AI (ICT4403)
- **Assignment:** Assignment 2 — Autoregressive Text Generation
- **Author / Student Name:** [Student Name]
- **Registration Number:** [Registration Number]
- **Developed System Title:** Comparative Autoregressive Word-Level Language Modeling with Recurrent Architectures (Simple RNN, BiRNN, LSTM, GRU)
- **Dataset:** *The Adventures of Sherlock Holmes* by Sir Arthur Conan Doyle (Project Gutenberg eBook #1661)
- **Dataset Link:** [https://www.gutenberg.org/files/1661/1661-0.txt](https://www.gutenberg.org/files/1661/1661-0.txt)
- **GitHub Repository Link:** [https://github.com/your-username/fgai_assignment](https://github.com/your-username/fgai_assignment)

---

## 1. Problem Statement & Objective

### 1.1 Problem Statement
Autoregressive language models predict probability distributions over sequences of discrete tokens by sequentially predicting the next token conditioned on prior context:
$$P(W) = \prod_{t=1}^T P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

Recurrent neural network (RNN) architectures have historically formed the foundation of sequential text processing. However, basic recurrent formulations face fundamental mathematical challenges, notably the **vanishing and exploding gradient problem**, which curtails their capacity to retain long-range semantic context. Modern gated variants such as the **Long Short-Term Memory (LSTM)** network and the **Gated Recurrent Unit (GRU)** introduce internal gating mechanisms designed to maintain stable error gradients across long temporal horizons. Meanwhile, **Bidirectional RNNs (BiRNN)** exploit context in both forward and reverse directions.

### 1.2 Assignment Objective
The objective of this assignment is to:
1. Select a high-quality textual dataset and construct causal input–target sliding windows for autoregressive word-level modeling.
2. Implement four recurrent architectures (**Simple RNN**, **Bidirectional RNN**, **LSTM**, and **GRU**) under strictly comparable experimental conditions (identical vocabulary, matched effective hidden representation size, shared sequence length, and identical optimizer parameters).
3. Systematically evaluate and compare the four models across **quantitative metrics** (Cross-Entropy Loss, Perplexity, Top-1 and Top-5 Next-Word Accuracy, Training Speed, Inference Latency) and **qualitative characteristics** (Coherence, Lexical Diversity via Distinct-1/2, Contextual Consistency, Repetition Rate).
4. Conduct a rigorous **sampling-temperature analysis** across at least three temperature settings ($T \in \{0.2, 0.7, 1.2\}$) on identical seed prompts.
5. Provide a justified conclusion identifying the optimal recurrent model for autoregressive generation.

---

## 2. Dataset Description & Preprocessing

### 2.1 Selected Corpus Characteristics
- **Corpus:** *The Adventures of Sherlock Holmes* by Sir Arthur Conan Doyle.
- **Source Link:** [Project Gutenberg #1661](https://www.gutenberg.org/files/1661/1661-0.txt).
- **Volume:** 593,731 raw characters; 121,822 cleaned words/tokens across 12 short stories.
- **Linguistic Characteristics:** Sophisticated 19th-century narrative prose, complex periodic sentences, formal dialogue exchanges, and varied deductive reasoning vocabulary.
- **Suitability:** Unlike informal social media or code datasets, this literary corpus possesses consistent grammatical syntax, rich lexical diversity (4,378 unique word types in vocabulary), and clear sentence boundaries, making it an ideal benchmark for word-level language modeling.

### 2.2 Text Preprocessing Pipeline
The raw text undergoes a deterministic, multi-stage cleaning and tokenization pipeline:
1. **Header/Footer Stripping:** Automated elimination of Project Gutenberg legal disclaimers, metadata banners, and transcription footers.
2. **Regex Tokenization:** Preserves words, contractions (e.g., `don't`, `it's`, `holmes's`), and punctuation marks (`.`, `,`, `!`, `?`, `;`, `:`) as individual semantic tokens while filtering non-printable artifacts.
3. **Casing Normalization:** All alphabetic tokens are converted to lowercase.
4. **Vocabulary Construction:**
   - Word frequency distribution computed across the cleaned token stream.
   - Special control tokens: `<pad>` (index 0), `<unk>` (index 1), `<sos>` (index 2), `<eos>` (index 3).
   - Minimum frequency threshold $f \ge 2$, capped at top 6,000 most frequent tokens, ensuring $>97\%$ corpus token coverage while bounding output projection matrix size. The resulting vocabulary contains **4,378 unique tokens**.

### 2.3 Causal Sequence Windowing
To enforce causal autoregression without data leakage:
- Context window length is fixed to $L = 25$ tokens.
- For each index $i$, input context is $\mathbf{x}_i = [w_i, w_{i+1}, \dots, w_{i+L-1}]$ and target label is $y_i = w_{i+L}$.
- **Temporal Split:** The chronological sequence is split into:
  - **Training Set:** 48,716 samples (80%)
  - **Validation Set:** 6,079 samples (10%)
  - **Testing Set:** 6,079 samples (10%)
  No future tokens are ever exposed to the training set.

---

## 3. Model Architectures & Controlled Experimental Setup

To ensure strict scientific validity, all models are designed with matched parameter capacities and identical training regimes:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Vocabulary Size ($V$)** | 4,378 | Shared across all 4 models |
| **Embedding Dimension ($d_e$)** | 128 | Dense word embedding vector size |
| **Effective Hidden Dim ($d_h$)** | 256 | Combined recurrent representation size |
| **Recurrent Layers** | 2 | Stacked recurrent layers |
| **Dropout Rate** | 0.2 | Applied after embedding and recurrent layers |
| **Context Length ($L$)** | 25 | Words in sliding context window |
| **Batch Size** | 64 | Minibatch size |
| **Optimizer** | Adam | $\beta_1=0.9, \beta_2=0.999, \text{weight\_decay}=10^{-5}$ |
| **Initial Learning Rate** | $1 \times 10^{-3}$ | Reduced on plateau by factor $0.5$ |
| **Gradient Clipping** | $5.0$ | Max gradient norm to prevent explosion |
| **Loss Criterion** | Cross-Entropy | $\mathcal{L} = -\log P(y \mid \mathbf{x})$, `<pad>` ignored |

### Architectural Formulations:

1. **Simple RNN (Elman):**
   $$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$
   - Parameter Count: **1,915,930**
   - Uses standard recurrent feedback without gating.
   - Susceptible to vanishing gradients as sequence length $L$ grows.

2. **Bidirectional RNN (BiRNN):**
   $$h_t^{\to} = \text{RNN}_{\text{fwd}}(x_t, h_{t-1}^{\to}), \quad h_t^{\leftarrow} = \text{RNN}_{\text{bwd}}(x_t, h_{t+1}^{\leftarrow})$$
   $$h_t = [h_t^{\to} \,\|\, h_t^{\leftarrow}]$$
   - Parameter Count: **1,850,394**
   - For fair capacity comparison, forward and backward branches each use $d_h / 2 = 128$ hidden units, yielding an identical concatenated vector of size $256$.
   - **Autoregressive Context Encoding:** During next-word prediction on context $[w_1, \dots, w_L]$, the bidirectional pass encodes the available prompt from both directions without seeing future target token $w_{L+1}$.

3. **LSTM (Long Short-Term Memory):**
   $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c), \quad C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
   $$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(C_t)$$
   - Parameter Count: **2,607,130**
   - Mitigates vanishing gradients via the constant error carousel of internal cell state $C_t$ governed by multiplicative forget, input, and output gates.

4. **GRU (Gated Recurrent Unit):**
   $$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r), \quad z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$$
   $$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t] + b), \quad h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
   - Parameter Count: **2,376,730**
   - Combines forget and input gates into a single update gate $z_t$, yielding comparable representational capacity to LSTM with approximately 25% fewer recurrent parameters.

---

## 4. Execution Steps & Required Dependencies

### 4.1 Installation & Environment Setup
Clone the repository and install required packages:
```bash
git clone https://github.com/your-username/fgai_assignment.git
cd fgai_assignment
pip install -r requirements.txt
```

### 4.2 Running Automated Unit Tests
Verify model integrity, causality, and sampling mechanisms:
```bash
python -m pytest tests/ -v
```

### 4.3 Running End-to-End Pipeline
Execute data preparation, training across all four models, evaluation, generation, and figure export:
```bash
python main.py --epochs 5 --batch_size 64
```

### 4.4 Generating Submission Word Document & PDF
```bash
python scripts/generate_submission_doc.py --name "Your Full Name" --reg_num "Your_Reg_Number" --repo_url "https://github.com/your-username/fgai_assignment"
```

---

## 5. Quantitative Results & Evaluation

### 5.1 Summary Comparison Table
The quantitative evaluation benchmarks all four models across statistical loss, perplexity, prediction accuracy, and computational runtime:

| Model | Parameters | Train Loss | Val Loss | Test Loss | Test Perplexity (PPL) | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Training Time (s) | Inference Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple RNN** | 1,915,930 | 4.2957 | 5.1250 | 5.0654 | 158.44 | 15.61% | 36.90% | 207.21s | 2.15ms |
| **Bidirectional RNN** | 1,850,394 | 4.2958 | 5.0784 | 5.0280 | 152.63 | 16.07% | 37.97% | 238.62s | 3.20ms |
| **LSTM** | 2,607,130 | 4.6227 | 4.9441 | 4.9057 | 135.06 | 16.47% | 37.95% | 289.18s | 1.85ms |
| **GRU** | 2,376,730 | 3.9710 | 4.9514 | **4.8810** | **131.77** | **18.05%** | **38.86%** | 438.06s | 6.94ms |

### 5.2 Key Quantitative Findings:
1. **Perplexity Hierarchy:**
   $$\text{GRU } (\mathbf{131.77}) < \text{LSTM } (135.06) < \text{BiRNN } (152.63) < \text{Simple RNN } (158.44)$$
   Gated architectures (GRU and LSTM) achieve substantial perplexity reductions ($\approx 23–27$ points lower PPL) relative to Simple RNN, proving the critical advantage of additive gradient pathways.
2. **Next-Word Prediction Accuracy:**
   GRU achieves the highest Top-1 prediction accuracy (**18.05%**) and Top-5 accuracy (**38.86%**), outperforming Simple RNN (15.61% / 36.90%). In word-level generation over 4,378 candidate words, an 18.05% exact top-1 next-word match is strong for a compact recurrent model.
3. **Loss Dynamics:**
   GRU attained the lowest training loss (3.9710) and lowest test loss (4.8810). Simple RNN plateaued early around val loss 5.10, indicating inability to fit nuanced dependencies across the 25-token context.

### 5.3 Comparative Visualization Plots:
- **Training & Validation Loss Curves:** `results/loss_curves.png`
- **Validation Perplexity Curves:** `results/perplexity_curves.png`
- **Top-1 and Top-5 Accuracy Comparison:** `results/accuracy_comparison.png`
- **Computational Efficiency:** `results/efficiency_comparison.png`

---

## 6. Qualitative Evaluation & Sampling-Temperature Analysis

### 6.1 Sampling Mechanisms
Given output logits $\mathbf{z}$, the temperature-scaled token probability distribution is:
$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

We evaluate three temperature regimes:
- **Low Temperature ($T = 0.2$):** Sharpened probability distribution approximating greedy decoding. Maximizes syntactic coherence but prone to repetitive loops.
- **Moderate Temperature ($T = 0.7$):** Balanced distribution balancing fluency, lexical variety, and semantic structure.
- **High Temperature ($T = 1.2$):** Flattened distribution with high entropy. Maximizes vocabulary diversity at the expense of grammatical degradation.

### 6.2 Lexical Diversity & Repetition Quantitative Metrics

| Model | Temperature ($T$) | Distinct-1 Ratio | Distinct-2 Ratio | 3-Gram Repetition Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Simple RNN** | 0.2 | 0.5848 | 0.7499 | 36.89% |
| **Simple RNN** | 0.7 | 0.7302 | 0.9447 | 2.86% |
| **Simple RNN** | 1.2 | 0.8754 | 1.0000 | 0.00% |
| **Bidirectional RNN** | 0.2 | 0.5077 | 0.6585 | 36.23% |
| **Bidirectional RNN** | 0.7 | 0.7416 | 0.9711 | 1.90% |
| **Bidirectional RNN** | 1.2 | 0.8967 | 1.0000 | 0.00% |
| **LSTM** | 0.2 | 0.5421 | 0.7399 | 28.37% |
| **LSTM** | 0.7 | 0.6896 | 0.9644 | **0.00%** |
| **LSTM** | 1.2 | 0.8699 | 1.0000 | **0.00%** |
| **GRU** | 0.2 | 0.5245 | 0.7618 | 22.36% |
| **GRU** | 0.7 | **0.7648** | **0.9812** | **0.00%** |
| **GRU** | 1.2 | 0.8350 | 0.9902 | **0.00%** |

### 6.3 Qualitative Analysis of Generated Samples

#### Seed Prompt: *"it was a dark and stormy night when"*

```
[Simple RNN]
T=0.2: "it was a dark and night when he had gone, and i have not be more successful. it is a very shiny, and the whole was struck with a very shiny..."
T=0.7: "it was a dark and night when she had into an ulster. it was a cheetah ball of the paper. it was a little expense. i found that i had never seen him..."
T=1.2: "it was a dark and night when i arrived through them. there are all! if cried lived did i have excuse bitter herefordshire..."

[Bidirectional RNN]
T=0.2: "it was a dark and night when he was a very deep, and i have heard a very friendly. i have heard a very deep man. i have heard a very friendly..."
T=0.7: "it was a dark and night when you have not been in the evening. you will be able to meet you do not wish to me? i asked in astonishment..."
T=1.2: "it was a dark and night when quick to this room colonies his money in the third as immense in glancing held and, the noble gentleman..."

[LSTM]
T=0.2: "it was a dark and night when the of the of the. i am sure to be a of the. i am a of the. it is a of the of the..."
T=0.7: "it was a dark and night when my time and which might have been address to you. in the bell of an night. i need to know to you, watson..."
T=1.2: "it was a dark and night when never ran across some show a hotel. it saved he at how mr. sparkled, i know, i said that it was..."

[GRU]
T=0.2: "it was a dark and night when she had been a small of red, and the church was in the street, and i was aware of the window..."
T=0.7: "it was a dark and night when she had been married. what was from the crown? i had been beautiful and my stepfather's manner had aroused suspicions..."
T=1.2: "it was a dark and night when my double across the conscience recent, of his appeared, where his up shuttered into the secret chamber..."
```

### 6.4 Key Qualitative Observations:
1. **Low Temperature ($T = 0.2$): Pathological Repetition Loops**
   - All models display pronounced repetition rates (22% to 37%).
   - Simple RNN and BiRNN get trapped in immediate degenerate cycles (*"a very shiny... a very shiny"* / *"a very deep man... a very deep man"*).
   - This occurs because greedy sampling repeatedly chooses the argmax transition without exploratory entropy.
2. **Balanced Temperature ($T = 0.7$): The Sweet Spot for Coherence & Diversity**
   - Distinct-1 jumps to $0.69–0.76$ and Distinct-2 reaches $0.94–0.98$.
   - Repetition rate drops to **0.00%** for both LSTM and GRU.
   - The GRU output exhibits remarkable stylistic fidelity to Arthur Conan Doyle's prose (*"what was from the crown? i had been beautiful and my stepfather's manner had aroused suspicions"*).
3. **High Temperature ($T = 1.2$): Semantic Degradation & Entropy Explosion**
   - Distinct-2 reaches $1.000$ (complete lexical uniqueness, 0 repetition).
   - However, syntactic agreement breaks down into erratic grammatical fragments (*"colonies his money in the third as immense in glancing held"*), illustrating that excessively flat distributions sample low-probability tokens that violate grammatical constraints.

---

## 7. Conclusion & Best-Performing Architecture Justification

### 7.1 Comprehensive Comparison Summary
- **Simple RNN:** Fails to maintain long context due to vanishing gradients; highest perplexity (**158.44**) and lowest top-1 accuracy (**15.61%**); collapses into repetitive loops at low temperature.
- **Bidirectional RNN:** Provides improved context encoding over Simple RNN (PPL **152.63** vs 158.44), but incurs higher inference overhead without matching the long-term gating retention of LSTM/GRU.
- **LSTM:** Excels in long-range stability (PPL **135.06**, Acc **16.47%**), achieves 0% repetition at $T=0.7$, but requires the largest parameter footprint (2.61M parameters).
- **GRU:** Achieves the **lowest test loss (4.8810)**, the **lowest test perplexity (131.77)**, the **highest top-1 accuracy (18.05%)**, the **highest top-5 accuracy (38.86%)**, and the **highest balanced lexical diversity (Distinct-1: 0.7648, Distinct-2: 0.9812)** with **0.0% repetition** at $T=0.7$.

### 7.2 Final Verdict
**GRU (Gated Recurrent Unit)** is unequivocally the **best-performing model** for autoregressive text generation on this dataset. It strikes the optimal trade-off between representational capacity, predictive accuracy, and generative fluency. When paired with stochastic temperature sampling at **$T = 0.7$**, GRU generates rich, coherent, and stylistically authentic English text without falling into degenerate repetitive loops.
