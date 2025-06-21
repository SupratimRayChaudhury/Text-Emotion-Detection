# Text-Emotion-Detection

A lightweight emotion classification model that detects the emotion expressed in text—such as *Happy*, *Sad*, *Angry*, or *Neutral*. Developed as part of the **AI/ML Developer – Entry-Level Screening Assignment** for Jayadhi Limited.

## 🚀 Project Overview

This project aims to classify text into predefined emotional categories using a machine learning model trained on a labeled dataset. It demonstrates key AI/ML concepts including data preprocessing, model training, evaluation, and optional UI deployment.

## 🧠 Features

- Multi-class emotion classification
- Clean preprocessing pipeline
- Evaluation with confusion matrix and accuracy score
- Optional interactive UI using Streamlit

## 📊 Dataset Used

- **Dataset:** [Emotion Dataset from HuggingFace Datasets](https://huggingface.co/datasets/dair-ai/emotion)  
- **Classes:** Happy, Sad, Angry, Fear, Surprise, Love, Neutral  
- **Source:** Twitter-based labeled data

## ⚙️ Approach Summary

1. **Text Cleaning:** Lowercasing, punctuation removal, tokenization  
2. **Vectorization:** TF-IDF vectorization of text input  
3. **Modeling:** Logistic Regression and/or Random Forest classifier  
4. **Evaluation:** Accuracy Score, Confusion Matrix  
5. **(Optional)** UI built with Streamlit for real-time predictions

## 📝 Results

- **Model:** Logistic Regression  
- **Accuracy:** ~[Insert final accuracy]%  
- **Confusion Matrix:** See `notebook` or `output.png`

## 🛠️ Dependencies

Install required packages with:

```bash
pip install -r requirements.txt


Text-Emotion-Detection/
├── data/                  # Raw or preprocessed data
├── models/                # Trained model files (if any)
├── notebook/              # Jupyter Notebooks
├── app.py                 # Streamlit app
├── requirements.txt
└── README.md
