# 📰 AI-Powered News Classification Using Self-Attention

An end-to-end Explainable AI (XAI) text classification pipeline that automatically categorizes news articles into five distinct domains using a custom **Multi-Head Self-Attention Network**. The system includes a comparative baseline LSTM analysis and is fully deployed via an interactive web interface using **Streamlit**.

---

## 🎯 Problem Statement
Modern digital news agencies process thousands of incoming text streams daily. Manual categorization is slow, expensive, and scales poorly. 

The objective of this project is to build an intelligent web-native text sorting system that:
* **Automates Categorization:** Automatically routes articles to their correct domains.
* **Captures Global Context:** Leverages Self-Attention to map long-range connections between words simultaneously.
* **Demystifies the Black Box:** Extracts internal model weights to show exactly *which* words influenced the final classification prediction.

---

## 🏗️ Technical Workflow & Architecture

The system processes text sequentially from raw user string arrays down to dynamic visualization matrix distributions:

```text
Input Raw Text ➔ Clean & Lowercase ➔ Tokenization & Padding ➔ 128D Embedding ➔ Multi-Head Attention ➔ Global Pooling ➔ Softmax Classification

Architectural Breakdown:
Input Layer: Standardizes sentence dimensions down to a clean structural matrix ceiling (MAX_LEN = 150).

Embedding Layer: Projects discrete integer tokens into a continuous 128-dimensional dense vector space.

Multi-Head Attention Layer: Utilizes 4 processing heads to learn complex bidirectional word-to-word contextual dependencies simultaneously.

Global Average Pooling 1D: Compresses the dimensional sequence map to prevent training over-fitting.

Dense Output Head: A 5-unit layer running a Softmax activation to distribute probability scores across target classes.

📋 Project Execution Milestones
📊 Milestone 1: Exploratory Data Analysis (EDA)
Analyzed structural properties of the BBC News dataset to calibrate vocabulary limitations:

Target Categories: Business (0), Politics (1), Sport (2), Tech (3), Entertainment (4).

Feature Engineering: Calculated distribution count plots, pie charts, and article length text histograms to balance classification ratios.

🛠️ Milestone 2 & 3: Deep Preprocessing & Mapping
Normalization steps applied: forced lowercase casting, stripped punctuation indices, and removed specialized symbol syntax using Python regex layers.

Vectorized tokens and padded sequence lengths uniformly using the Keras Tokenizer API.

🧪 Milestone 4, 5, & 6: Architecture Comparison
Built and validated two distinct architectures to track accuracy and computational trade-offs:

Baseline Model (LSTM): Sequential token evaluation. Prone to directional bottlenecks over longer texts.

Advanced Model (Self-Attention): Parallelized context modeling. Achieved identical or superior accuracy boundaries with significantly reduced training latency.

🧠 Milestone 7 & 8: Attention Matrix Extraction (Explainable AI)
The Multi-Head Attention layer exposes internal parameters via the return_attention_scores=True flag. The web workspace isolates these score slices, normalizes their weights, and displays a clean Word-to-Word Attention Heatmap using Seaborn, visually proving why the model marks a specific domain.

🛠️ Technology Stack
Core Language: Python

Deep Learning Framework: TensorFlow / Keras (Functional & Sequential API)

Natural Language Processing: Tokenizer, NLTK Sequence Padding

Web Deployment Framework: Streamlit Core UI

Data Visualization & Analytics: Pandas, NumPy, Matplotlib, Seaborn

📂 Project Directory Structure
Plaintext
AI-Powered-News-Classification/
│
├── data/
│   └── bbc-text.csv          # Raw evaluation source articles
│
├── app.py                    # Streamlit web application dashboard code
├── Articles_Attention.ipynb # Training notebook (EDA, LSTMs, Attention Modeling)
├── requirements.txt          # Python dependency environment packages
│
├── model (1).h5                  # Saved Neural Network weights
├── tokenizer (1).pkl             # Serialized word index token dictionary
└── label_encoder.pkl         # Target categorical index dictionary maps
🚀 Running the Streamlit Web Application Locally
Follow these quick steps to launch the interactive AI news classification dashboard on your computer:

1. Configure Local Project Directory
Clone the repository and move your downloaded asset weights directly into the primary folder:

Bash
git clone [https://github.com/arthireddy14/AI-Powered-News-Classification-System.git](https://github.com/arthireddy14/AI-Powered-News-Classification-System.git)
cd AI-Powered-News-Classification-System
2. Install Project Dependencies
Install the required analytical libraries using the pip environment manager:

Bash
pip install streamlit tensorflow pandas numpy scikit-learn seaborn matplotlib
3. Verify Local Artifact File Names
Ensure that the model weights and data transformation pkl assets match the internal loading script parameters:

model.h5

tokenizer.pkl

label_encoder.pkl

4. Trigger Web Interface App
Execute the Streamlit runtime script via your system console terminal:

Bash
streamlit run app.py
A local browser dashboard tab will launch automatically. Paste any headline or news text to instantly observe class metrics and word-to-word attention heatmap maps!

👨‍💻 Author
Arthi Reddy Aspiring AI/ML Engineer passionate about Natural Language Processing, Deep Learning, Predictive Analytics, and Explainable AI Systems.