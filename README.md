# 📰 AI-Powered News Classification Using Self-Attention

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![NLP](https://img.shields.io/badge/NLP-SelfAttention-green)

## 📌 Overview

This project is a Deep Learning-based News Classification System that automatically categorizes news articles into different categories using a Multi-Head Self-Attention mechanism.

The model learns contextual relationships between words, identifies important information within articles, and predicts the most relevant news category. A Streamlit web application provides an interactive interface for real-time predictions.

---

## 🎯 Problem Statement

News organizations receive thousands of articles every day. Manually categorizing articles is time-consuming and difficult to scale.

The goal of this project is to build an intelligent system that can:

* Automatically classify news articles
* Understand contextual word relationships
* Highlight important words using Self-Attention
* Improve classification accuracy and interpretability

---

## 🚀 Solution Workflow

```text
Raw News Article
        ↓
Text Cleaning
        ↓
Tokenization & Padding
        ↓
Embedding Layer (128D)
        ↓
Multi-Head Self-Attention
        ↓
Global Average Pooling
        ↓
Dense Layer + Softmax
        ↓
Predicted News Category
```

---

## 🏗️ Model Architecture

### Components

**1. Embedding Layer**

* Converts words into dense vector representations.
* Embedding Dimension: 128

**2. Multi-Head Self-Attention**

* Learns contextual relationships between words.
* Uses 4 attention heads to capture different patterns simultaneously.

**3. Global Average Pooling**

* Reduces feature dimensions.
* Helps prevent overfitting.

**4. Dense Output Layer**

* Softmax activation for multi-class classification.
* Predicts one of five news categories.

---

## ✨ Features

* Automated News Classification
* Multi-Head Self-Attention Architecture
* Explainable AI (Attention Visualization)
* Real-Time Predictions
* Streamlit Web Dashboard
* Multi-Class Category Detection

---

## 📊 Dataset

The project uses the BBC News Dataset containing articles from five categories:

| Category      | Label |
| ------------- | ----- |
| Business      | 0     |
| Politics      | 1     |
| Sport         | 2     |
| Tech          | 3     |
| Entertainment | 4     |

---

## ⚙️ Methodology

### Data Preprocessing

* Text Cleaning
* Lowercase Conversion
* Removal of Special Characters
* Tokenization
* Sequence Padding

### Exploratory Data Analysis

* Category Distribution
* Article Length Analysis
* Class Balance Visualization

### Model Development

Two models were implemented and compared:

#### Baseline Model

* LSTM Network
* Sequential text processing

#### Proposed Model

* Multi-Head Self-Attention Network
* Parallel contextual learning
* Better interpretability

---

## 📈 Results

The Self-Attention model successfully learns contextual word importance and accurately classifies news articles.

### Application Outputs

* Predicted News Category
* Prediction Confidence Score
* Attention Heatmap Visualization

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Deep Learning & NLP

* TensorFlow
* Keras

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

---

## 📂 Project Structure

```text
AI-Powered-News-Classification/
│
├── data/
│   └── bbc-text.csv
│
├── app.py
├── Articles_Attention.ipynb
├── requirements.txt
│
├── model (1).h5
├── tokenizer (1).pkl
└── label_encoder.pkl
```

---

## 🚀 Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/arthireddy14/AI-Powered-News-Classification-System.git

cd AI-Powered-News-Classification-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit Application

```bash
streamlit run app.py
```

---

## 🎯 Applications

* News Recommendation Systems
* Content Management Platforms
* Media Monitoring
* Automated Article Categorization
* Information Retrieval Systems

---

## 🔮 Future Enhancements

* BERT Integration
* RoBERTa Integration
* Real-Time News API Pipeline
* Multi-Language Classification
* Advanced Explainable AI Visualizations

---

## 👨‍💻 Author

**Arthi Reddy**

Aspiring AI/ML Engineer passionate about:

* Machine Learning
* Natural Language Processing
* Deep Learning
* Explainable AI
* Data Analytics

---

## 📌 Repository Description

AI-powered News Classification System using Multi-Head Self-Attention and Deep Learning to automatically categorize news articles and visualize contextual word importance through an interactive Streamlit dashboard.
