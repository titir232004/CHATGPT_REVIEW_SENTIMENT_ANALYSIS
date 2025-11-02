# 💬 ChatGPT Review Sentiment Analysis

This project analyzes user reviews of ChatGPT to understand sentiment trends, key themes, and satisfaction levels across different factors such as rating, platform, and version.  
It uses **BERT-based sentiment classification** and provides an interactive **Streamlit dashboard** for visual insights.

---

## 📁 Project Structure

CHATGPT_REVIEW_SENTIMENT_ANALYSIS/
│
├── app.py # Streamlit dashboard for visualization
├── chatgpt_style_reviews_dataset.xlsx # Raw review dataset
├── cleaned_balanced_reviews.csv # Cleaned and preprocessed dataset
├── data_preprocessing.py # Text cleaning, tokenization, balancing
├── improved_model.py # Fine-tuned BERT model (3-class)
├── evaluate_model.py # Model evaluation and accuracy testing
├── evaluation_results.csv # Predictions and metrics
├── data_analysis.py # Exploratory data analysis and charts
└── README.md # Documentation

markdown
Copy code

---

## ⚙️ Features

✅ **Sentiment Classification (BERT)**  
Classifies reviews as:
- **0 → Negative**
- **1 → Neutral**
- **2 → Positive**

✅ **Key Analysis Questions**
1. Overall sentiment distribution  
2. Sentiment vs. user ratings  
3. Keyword/word cloud analysis per sentiment  
4. Sentiment trends over time  
5. Verified vs. non-verified review sentiment  
6. Review length vs. sentiment  
7. Location and platform-based sentiment  
8. ChatGPT version comparison  
9. Common negative feedback themes  
10. Real-time interactive dashboard (Streamlit)

---

## 🧩 Tech Stack

- **Python**
- **PyTorch** – BERT model training  
- **Transformers (Hugging Face)** – Text embeddings  
- **Pandas, NumPy, Matplotlib, Seaborn, Plotly** – Data analysis  
- **Scikit-learn** – Evaluation metrics  
- **Streamlit** – Dashboard visualization  
- **WordCloud** – Sentiment keyword visualization

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/CHATGPT_REVIEW_SENTIMENT_ANALYSIS.git
cd CHATGPT_REVIEW_SENTIMENT_ANALYSIS
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Train / Evaluate the Model
bash
Copy code
python improved_model.py
python evaluate_model.py
4️⃣ Launch the Dashboard
bash
Copy code
streamlit run app.py
