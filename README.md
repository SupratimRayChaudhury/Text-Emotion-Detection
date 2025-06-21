Text-Emotion-Detection
A lightweight emotion classification model that detects the emotion expressed in text—such as Happy, Sad, Angry, or Neutral. Developed as part of the AI/ML Developer – Entry-Level Screening Assignment for Jayadhi Limited.

🚀 Project Overview
This project aims to classify text into predefined emotional categories using a machine learning model trained on a labeled dataset. It demonstrates key AI/ML concepts including data preprocessing, model training, evaluation, and optional UI deployment.

🧠 Features
Multi-class emotion classification

Clean preprocessing pipeline

Evaluation with confusion matrix and accuracy score

Optional interactive UI using Streamlit

📊 Dataset Used
Dataset: Emotion Dataset from HuggingFace Datasets

Classes: Happy, Sad, Angry, Fear, Surprise, Love, Neutral

Source: Twitter-based labeled data

⚙️ Approach Summary
Text Cleaning: Lowercasing, punctuation removal, tokenization

Vectorization: TF-IDF vectorization of text input

Modeling: Logistic Regression and/or Random Forest classifier

Evaluation: Accuracy Score, Confusion Matrix

(Optional) UI built with Streamlit for real-time predictions

📝 Results
Model: Logistic Regression

Accuracy: ~[Insert final accuracy]%

Confusion Matrix: See notebook/output.png

🛠️ Dependencies
Install required packages with:

bash
Copy
Edit
pip install -r requirements.txt
Key libraries:

Python 3.x

scikit-learn

pandas

numpy

matplotlib / seaborn

streamlit (optional)

💻 Running the Project
Training (Jupyter Notebook):

bash
Copy
Edit
jupyter notebook emotion_detection.ipynb
Streamlit App (Optional):

bash
Copy
Edit
streamlit run app.py
📁 Folder Structure
bash
Copy
Edit
Text-Emotion-Detection/
├── data/                  # Raw or preprocessed data
├── models/                # Trained model files (if any)
├── notebook/              # Jupyter Notebooks
├── app.py                 # Streamlit app
├── requirements.txt
└── README.md
