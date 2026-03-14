# 📈 Financial News Causality Detection using FinBERT

A multi-phase NLP research project that detects **causal relationships** in financial news articles and predicts their impact on stock market movements — using transformer-based models, classical ML baselines, and interpretability tools.

---

## 🧠 Project Overview

This project investigates whether financial news headlines/articles contain causal language that predicts future stock returns or volatility. It builds a pipeline from raw data collection to explainable model predictions.

**Core Question:** *Can we detect causality in financial text and use it to anticipate market movements?*

---

## 🗂️ Project Phases

| Phase | Notebook | Description |
|-------|----------|-------------|
| **Phase 1** | `PHASE_1.ipynb` | Dataset collection & exploration (financial news articles) |
| **Phase 2** | `PHASE_2.ipynb` | Classical ML baselines (TF-IDF + Logistic Regression, SVM, Naive Bayes) |
| **Phase 3** | `PHASE_3.ipynb` | FinBERT fine-tuning with multimodal features (text + numerical) |
| **Phase 4** | `PHASE_4.ipynb` | Causality & interpretability analysis (SHAP, attention maps, counterfactuals) |
| **Phase 5** | `PHASE_5.ipynb` | Final evaluation & deployment pipeline (Google Colab + Drive) |

---

## 🏗️ Repository Structure

```
financial-causality-nlp/
│
├── notebooks/
│   ├── PHASE_1.ipynb       # Data collection & EDA
│   ├── PHASE_2.ipynb       # Baseline ML models
│   ├── PHASE_3.ipynb       # FinBERT multimodal model
│   ├── PHASE_4.ipynb       # Interpretability analysis
│   └── PHASE_5.ipynb       # Final pipeline (Colab)
│
├── data/                   # Place your datasets here (see Data Setup)
├── models/                 # Saved model checkpoints (.pkl / .pt)
├── results/                # Evaluation outputs, plots, reports
├── assets/                 # Images and figures for documentation
│
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

- **Language Models:** [FinBERT (ProsusAI)](https://huggingface.co/ProsusAI/finbert) via HuggingFace Transformers
- **Deep Learning:** PyTorch, AdamW optimizer, cosine/linear LR schedulers
- **Classical ML:** scikit-learn (Logistic Regression, LinearSVC, MultinomialNB)
- **Interpretability:** SHAP, Attention visualization
- **Data:** HuggingFace Datasets (`ashraq/financial-news-articles`), custom causality-labeled dataset
- **Training:** Kaggle GPU / Google Colab (T4/A100)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/financial-causality-nlp.git
cd financial-causality-nlp
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Data Setup

- **Phase 1** auto-downloads the dataset via HuggingFace.
- **Phase 2–4** require `final_financial_causality_dataset.csv` — place it in the `data/` folder.
- **Phase 5** expects files in Google Drive at `My Drive/NLP_Phase5/`.

### 4. Run Notebooks in Order

Open notebooks inside the `notebooks/` folder and run them sequentially:

```
PHASE_1 → PHASE_2 → PHASE_3 → PHASE_4 → PHASE_5
```

> 💡 **Tip:** Phases 3 and 5 are GPU-intensive. Run them on Kaggle or Google Colab for best performance.

---

## 📊 Key Features

- **Leakage-free labeling** — future returns computed via temporal shifting to prevent data leakage
- **Class imbalance handling** — Effective Number Sampling weights + `WeightedRandomSampler`
- **Multimodal architecture** — Combines FinBERT text embeddings with numerical financial features
- **Text augmentation** — Random word deletion for minority class oversampling
- **Gradient checkpointing** — Memory-efficient training for large batches
- **Full interpretability suite** — SHAP values, attention heatmaps, counterfactual analysis

---

## 📋 Requirements

See `requirements.txt` for the full list. Key packages:

```
transformers
torch
datasets
scikit-learn
shap
optuna
pandas
numpy
matplotlib
seaborn
```

---

## 📁 Data

| Dataset | Source | Usage |
|---------|--------|-------|
| Financial News Articles | HuggingFace (`ashraq/financial-news-articles`) | Phase 1 EDA |
| Financial Causality Dataset | Custom / Kaggle | Phases 2–5 |

> ⚠️ The labeled causality dataset is not included in this repo due to size. See Phase 1 notebook for dataset construction steps.

---

## 📌 Notes

- Set `DATA_PATH` in each notebook to point to your local or Kaggle dataset path.
- Model checkpoints saved as `.pkl.zip` can be large — use Git LFS or store them in Google Drive.
- For Kaggle runs, datasets should be added as Kaggle input datasets.

---

## 🙌 Acknowledgements

- [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) for the domain-adapted financial BERT model
- [HuggingFace Datasets](https://huggingface.co/datasets/ashraq/financial-news-articles) for open financial news data
- Kaggle for GPU compute during training

---

## 📄 License

This project is for academic/research purposes. See `LICENSE` for details.
