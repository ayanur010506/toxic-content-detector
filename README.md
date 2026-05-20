# 🛡️ Automatic Detection of Toxic Content in Social Networks

> **Bachelor's Diploma Project — Astana IT University (AITU), 2025**  
> *Development and Investigation of Methods for Automatic Detection of Toxic Content in Social Networks Based on Natural Language Processing Technologies in the Context of Cybersecurity*

---

## 📌 Overview

This project develops and compares NLP-based machine learning models for **automatic toxic content detection** in multilingual (Russian + English) social network text. The research covers the full pipeline — from raw data to a deployed demo application.

---

## 📊 Results

| Model | F1-Score | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 0.804 | 0.889 |
| SVM (LinearSVC) | 0.845 | 0.912 |
| Random Forest | 0.875 | 0.921 |
| **ruBERT (fine-tuned)** ✅ | **0.950** | **0.986** |

> Best model: `DeepPavlov/rubert-base-cased` fine-tuned on combined RU+EN dataset

---

## 🗂️ Project Structure

```
toxic-content-detector/
│
├── diploma_full_notebook.ipynb   # Full pipeline: EDA → ML → BERT → SHAP
├── streamlit_app.py              # Demo web application
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 📁 Datasets

| Dataset | Language | Examples | Source |
|---------|----------|----------|--------|
| ru_paradetox | 🇷🇺 Russian | 22 180 | [HuggingFace](https://huggingface.co/datasets/s-nlp/ru_paradetox) |
| Jigsaw Toxic Comments | 🇬🇧 English | 16 000 | [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) |
| **Combined** | RU + EN | **~38 000** | — |

---

## 🔬 NLP Pipeline

```
Raw Text
   │
   ▼
Text Cleaning          → remove URLs, mentions, special characters
   │
   ▼
Stopword Removal       → Russian (NLTK) + English (NLTK)
   │
   ├──► TF-IDF (80k features, ngram 1-2)
   │         │
   │         ▼
   │    Baseline ML Models
   │    ├── Logistic Regression
   │    ├── SVM (LinearSVC)
   │    └── Random Forest
   │
   └──► BERT Tokenizer (rubert-base-cased, max_len=128)
             │
             ▼
        ruBERT Fine-tuning
        ├── 3 epochs
        ├── batch_size = 32
        └── lr = 2e-5
             │
             ▼
        SHAP Interpretability
```

---

## 🚀 Run the Demo App Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/toxic-content-detector.git
cd toxic-content-detector

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download rubert_best.pt from Kaggle and place it in this folder

# 5. Run the app
streamlit run streamlit_app.py
```

App opens at: **http://localhost:8501**

---

## 🖥️ Demo App Features

- ✅ Multilingual input — Russian and English
- ✅ Real-time toxic / non-toxic classification
- ✅ Confidence scores with progress bars
- ✅ Preprocessed text preview
- ✅ Model metrics sidebar

---

## 📈 Visualizations (from notebook)

The notebook generates 11 charts:

1. `class_distribution.png` — class and language distribution
2. `text_length_distribution.png` — text length analysis
3. `wordcloud.png` — 4 word clouds (RU/EN × toxic/neutral)
4. `baseline_comparison.png` — ML model comparison
5. `confusion_matrices_baseline.png` — confusion matrices
6. `roc_curves_baseline.png` — ROC curves
7. `bert_training_history.png` — loss & F1 per epoch
8. `bert_evaluation.png` — BERT confusion matrix & ROC
9. `final_comparison.png` — all models comparison
10. `radar_chart.png` — radar chart
11. `shap_feature_importance.png` — SHAP top-25 features

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-blue)

| Library | Purpose |
|---------|---------|
| PyTorch | Deep learning framework |
| HuggingFace Transformers | ruBERT model & tokenizer |
| scikit-learn | Baseline ML models, TF-IDF |
| SHAP | Model interpretability |
| Streamlit | Demo web application |
| NLTK | Stopword removal |
| Kaggle GPU T4 | Model training environment |

---

## 👩‍🎓 Author

**[Your Name]**  
Bachelor's student, Astana IT University  
Specialty: Information Security / Computer Science  
Year: 2025

---

## 📄 License

This project is created for academic purposes as part of a bachelor's diploma thesis at Astana IT University.
