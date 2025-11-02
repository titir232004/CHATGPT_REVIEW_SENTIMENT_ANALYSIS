import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ========================
# 🧩 PAGE CONFIGURATION
# ========================
st.set_page_config(
    page_title="ChatGPT Reviews Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# ========================
# 📦 LOAD MODEL FUNCTION
# ========================
@st.cache_resource
def load_model():
    model_path = "./bert_sentiment_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

# ========================
# 🎨 SIDEBAR FILTERS
# ========================
st.sidebar.header("🔍 Filters & Questions")

question = st.sidebar.selectbox(
    "Select question",
    [
        "What is the distribution of review ratings?",
        "How do positive and negative reviews vary over time?",
        "Show sentiment counts for each category."
    ]
)

date_range = st.sidebar.date_input(
    "📅 Date range",
    value=[]
)

selected_ratings = st.sidebar.multiselect(
    "Show ratings",
    ["1 ⭐", "2 ⭐", "3 ⭐", "4 ⭐", "5 ⭐"],
    default=["1 ⭐", "2 ⭐", "3 ⭐", "4 ⭐", "5 ⭐"]
)

# ========================
# 🧠 MAIN DASHBOARD
# ========================
st.title("📊 ChatGPT Reviews Sentiment Dashboard")
st.markdown("""
Use the dropdowns below to explore review sentiment, rating distribution, and overall insights.  
Upload your dataset to analyze *real reviews* in real time.
""")

uploaded_file = st.file_uploader("📂 Upload dataset (CSV or XLSX)", type=["csv", "xlsx"])

# ========================
# 📈 IF FILE UPLOADED
# ========================
if uploaded_file:
    # Read uploaded file
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Expect a column named "review"
    if "review" not in df.columns:
        st.warning("⚠️ No column named 'review' found. Please upload a dataset with a 'review' column.")
        st.stop()

    st.success("✅ Dataset uploaded successfully!")

    # ========================
    # 🔮 SENTIMENT PREDICTION
    # ========================
    texts = df["review"].astype(str).tolist()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).numpy()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    df["Predicted_Sentiment"] = [label_map[p] for p in preds]

    # ========================
    # 📊 SENTIMENT DISTRIBUTION
    # ========================
    st.subheader("1️⃣ Distribution of Review Sentiments 📈")

    sentiment_counts = df["Predicted_Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        text="Count",
        title="Sentiment Distribution (Positive / Neutral / Negative)",
        color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"}
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    total_reviews = len(df)
    avg_rating = round((df["Predicted_Sentiment"].replace({"Negative": 1, "Neutral": 3, "Positive": 5}).mean()), 2)

    col1, col2 = st.columns(2)
    col1.metric("📦 Total Reviews", total_reviews)
    col2.metric("⭐ Average Rating (mapped sentiment)", avg_rating)

    # ========================
    # 🔍 PREVIEW TABLE
    # ========================
    st.subheader("2️⃣ Predictions Preview 🧾")
    st.dataframe(df[["review", "Predicted_Sentiment"]].head(20), use_container_width=True)

    # ========================
    # 📥 DOWNLOAD RESULTS
    # ========================
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv_data,
        file_name="predicted_sentiments.csv",
        mime="text/csv"
    )

    # ========================
    # 🧮 EVALUATION METRICS (if ground truth exists)
    # ========================
    if "true_sentiment" in df.columns:
        from sklearn.metrics import classification_report
        st.subheader("✅ Evaluation Metrics (if ground truth exists)")
        report = classification_report(df["true_sentiment"], df["Predicted_Sentiment"], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("📤 No dataset uploaded. Please upload a CSV or XLSX file to begin.")
    st.markdown("""
    *Using a small synthetic sample for demo.*
    """)

    # small synthetic demo data
    demo_df = pd.DataFrame({
        "review": [
            "I love how fast the response was!",
            "The output didn’t make sense at all.",
            "Average performance — could be better.",
            "Very helpful and clear explanations!",
            "Terrible experience, didn’t answer my question."
        ],
        "Predicted_Sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative"]
    })

    fig_demo = px.bar(
        demo_df["Predicted_Sentiment"].value_counts(),
        color=demo_df["Predicted_Sentiment"].value_counts().index,
        title="Demo Sentiment Distribution"
    )
    st.plotly_chart(fig_demo, use_container_width=True)
