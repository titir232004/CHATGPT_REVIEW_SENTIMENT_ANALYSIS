import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report

st.set_page_config(
    page_title="ChatGPT Reviews Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = "./bert_sentiment_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

st.sidebar.header("🔍 Explore Insights")

question = st.sidebar.selectbox(
    "Select Question to Explore",
    [
        "1️⃣ What is the distribution of review ratings?",
        "2️⃣ How do positive and negative reviews vary over time?",
        "3️⃣ Show sentiment counts for each category.",
        "4️⃣ What percentage of reviews are positive, neutral, or negative?",
        "5️⃣ Which sentiment category dominates the dataset?",
        "6️⃣ How balanced is the dataset after upsampling?",
        "7️⃣ Display a few predicted results with their reviews.",
        "8️⃣ What is the overall average sentiment score?",
        "9️⃣ Provide the classification evaluation metrics.",
        "🔟 Allow CSV download of predictions."
    ]
)


uploaded_file = st.file_uploader("📂 Upload dataset (CSV or XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, encoding="utf-8", errors="ignore")

    if "review" not in df.columns:
        st.error("❌ Dataset must contain a column named 'review'")
        st.stop()

    st.success("✅ Dataset uploaded successfully!")

    texts = df["review"].astype(str).tolist()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).numpy()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    df["Predicted_Sentiment"] = [label_map[p] for p in preds]


    st.title("📊 ChatGPT Reviews Sentiment Analysis Dashboard")

    # 1️⃣ Distribution of review ratings
    if "1️⃣" in question:
        st.subheader("Distribution of Review Sentiments 📈")
        sentiment_counts = df["Predicted_Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig = px.bar(
            sentiment_counts,
            x="Sentiment", y="Count",
            color="Sentiment",
            text="Count",
            color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
            title="Sentiment Distribution (Positive / Neutral / Negative)"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # 2️⃣ Variation over time (if date column exists)
    elif "2️⃣" in question:
        if "date" in df.columns:
            st.subheader("Sentiment Trend Over Time 🕒")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            trend = df.groupby(["date", "Predicted_Sentiment"]).size().reset_index(name="Count")
            fig = px.line(trend, x="date", y="Count", color="Predicted_Sentiment", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No 'date' column found — upload a dataset with review dates to view trends.")

    # 3️⃣ Sentiment counts
    elif "3️⃣" in question:
        st.subheader("Sentiment Counts Table 📊")
        st.dataframe(df["Predicted_Sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count"))

    # 4️⃣ Percentage breakdown
    elif "4️⃣" in question:
        st.subheader("Percentage Breakdown (%) 📉")
        percent = (df["Predicted_Sentiment"].value_counts(normalize=True) * 100).reset_index()
        percent.columns = ["Sentiment", "Percentage"]
        st.dataframe(percent)
        fig = px.pie(percent, names="Sentiment", values="Percentage", color="Sentiment",
                     color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
                     title="Sentiment Percentage Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # 5️⃣ Dominant sentiment
    elif "5️⃣" in question:
        st.subheader("Dominant Sentiment 🏆")
        dominant = df["Predicted_Sentiment"].value_counts().idxmax()
        st.success(f"🏅 The most frequent sentiment is **{dominant}**.")

    # 6️⃣ Dataset balance
    elif "6️⃣" in question:
        st.subheader("Dataset Balance Check ⚖️")
        counts = df["Predicted_Sentiment"].value_counts()
        st.bar_chart(counts)
        st.write("After upsampling, ideally all classes should have similar counts.")

    # 7️⃣ Preview predictions
    elif "7️⃣" in question:
        st.subheader("Sample Predictions 🧾")
        st.dataframe(df[["review", "Predicted_Sentiment"]].head(10), use_container_width=True)

    # 8️⃣ Average sentiment score
    elif "8️⃣" in question:
        st.subheader("Average Sentiment Score ⭐")
        score_map = {"Negative": 1, "Neutral": 3, "Positive": 5}
        avg_score = df["Predicted_Sentiment"].map(score_map).mean()
        st.metric("⭐ Average Sentiment Score", round(avg_score, 2))

    # 9️⃣ Evaluation metrics
    elif "9️⃣" in question:
        st.subheader("Evaluation Metrics 🧠")
        if "true_sentiment" in df.columns:
            report = classification_report(df["true_sentiment"], df["Predicted_Sentiment"], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        else:
            st.info("⚠️ No ground truth column (`true_sentiment`) found for evaluation.")

    # 🔟 CSV download
    elif "🔟" in question:
        st.subheader("📥 Download Predictions CSV")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions",
            data=csv_data,
            file_name="predicted_sentiments.csv",
            mime="text/csv"
        )

else:
    st.info("📤 Upload a dataset to begin sentiment analysis.")
