import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="ChatGPT Reviews Sentiment Dashboard", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    model_path = "./bert_sentiment_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

st.sidebar.header("🔍 Explore Key Questions")

question = st.sidebar.selectbox(
    "Select an Analysis Question 👇",
    [
        "1️⃣ Overall sentiment of user reviews",
        "2️⃣ Sentiment variation by rating",
        "3️⃣ Keywords or phrases per sentiment class",
        "4️⃣ Sentiment change over time",
        "5️⃣ Verified vs Non-Verified user sentiment",
        "6️⃣ Sentiment vs Review Length",
        "7️⃣ Sentiment by Location",
        "8️⃣ Sentiment by Platform (Web vs Mobile)",
        "9️⃣ Sentiment by ChatGPT Version",
        "🔟 Common negative feedback themes"
    ]
)

uploaded_file = st.file_uploader("📂 Upload dataset (CSV/XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')

    if "review" not in df.columns:
        st.error("⚠️ Please ensure your dataset has a 'review' column.")
        st.stop()

    st.success("✅ Dataset uploaded successfully!")

    texts = df["review"].astype(str).tolist()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).numpy()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    df["Predicted_Sentiment"] = [label_map[p] for p in preds]

    st.title("📊 ChatGPT Reviews Sentiment Dashboard")
    st.markdown("Explore 10 key insights derived from sentiment analysis of ChatGPT user reviews.")


    # 1️⃣ Overall Sentiment
    if "Overall" in question:
        st.subheader("1️⃣ Overall Sentiment of User Reviews")
        sentiment_counts = df["Predicted_Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            text="Count",
            title="Distribution of Sentiments",
            color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2️⃣ Sentiment by Rating
    elif "rating" in question.lower():
        if "rating" not in df.columns:
            st.warning("⚠️ No 'rating' column found in dataset.")
        else:
            st.subheader("2️⃣ Sentiment Variation by Rating")
            fig = px.histogram(
                df,
                x="rating",
                color="Predicted_Sentiment",
                barmode="group",
                title="Sentiment by Star Rating"
            )
            st.plotly_chart(fig, use_container_width=True)

    # 3️⃣ Keywords or phrases per sentiment
    elif "keywords" in question.lower():
        st.subheader("3️⃣ Common Keywords per Sentiment")
        for sentiment in ["Positive", "Neutral", "Negative"]:
            st.markdown(f"**{sentiment} Reviews Word Cloud**")
            text = " ".join(df[df["Predicted_Sentiment"] == sentiment]["review"].astype(str))
            if text.strip():
                wc = WordCloud(width=600, height=400, background_color="white").generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info(f"No {sentiment} reviews to display.")

    # 4️⃣ Sentiment change over time
    elif "time" in question.lower():
        if "date" not in df.columns:
            st.warning("⚠️ No 'date' column found for time-based analysis.")
        else:
            st.subheader("4️⃣ Sentiment Change Over Time")
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
            trend = df.groupby(pd.Grouper(key="date", freq="W"))["Predicted_Sentiment"].value_counts().unstack().fillna(0)
            fig = px.line(trend, title="Sentiment Trend Over Time")
            st.plotly_chart(fig, use_container_width=True)

    # 5️⃣ Verified vs Non-Verified
    elif "verified" in question.lower():
        if "verified_purchase" not in df.columns:
            st.warning("⚠️ No 'verified_purchase' column found.")
        else:
            st.subheader("5️⃣ Verified vs Non-Verified User Sentiment")
            fig = px.histogram(
                df,
                x="verified_purchase",
                color="Predicted_Sentiment",
                barmode="group",
                title="Sentiment by Verification Status"
            )
            st.plotly_chart(fig, use_container_width=True)

    # 6️⃣ Sentiment vs Review Length
    elif "length" in question.lower():
        st.subheader("6️⃣ Sentiment vs Review Length")
        df["length"] = df["review"].astype(str).apply(len)
        fig = px.box(
            df,
            x="Predicted_Sentiment",
            y="length",
            color="Predicted_Sentiment",
            title="Review Length vs Sentiment"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 7️⃣ Sentiment by Location
    elif "location" in question.lower():
        if "location" not in df.columns:
            st.warning("⚠️ No 'location' column found in dataset.")
        else:
            st.subheader("7️⃣ Sentiment by Location")
            fig = px.bar(
                df.groupby("location")["Predicted_Sentiment"].value_counts().unstack().fillna(0),
                title="Sentiment Across Locations"
            )
            st.plotly_chart(fig, use_container_width=True)

    # 8️⃣ Sentiment by Platform
    elif "platform" in question.lower():
        if "platform" not in df.columns:
            st.warning("⚠️ No 'platform' column found.")
        else:
            st.subheader("8️⃣ Sentiment Across Platforms (Web vs Mobile)")
            fig = px.histogram(
                df,
                x="platform",
                color="Predicted_Sentiment",
                barmode="group",
                title="Sentiment by Platform"
            )
            st.plotly_chart(fig, use_container_width=True)

    # 9️⃣ Sentiment by ChatGPT Version
    elif "version" in question.lower():
        if "version" not in df.columns:
            st.warning("⚠️ No 'version' column found.")
        else:
            st.subheader("9️⃣ Sentiment by ChatGPT Version")
            fig = px.histogram(
                df,
                x="version",
                color="Predicted_Sentiment",
                barmode="group",
                title="Sentiment by ChatGPT Version"
            )
            st.plotly_chart(fig, use_container_width=True)

    # 🔟 Common negative feedback themes
    elif "negative feedback" in question.lower():
        st.subheader("🔟 Common Negative Feedback Themes")
        neg_reviews = df[df["Predicted_Sentiment"] == "Negative"]["review"]
        if len(neg_reviews) == 0:
            st.info("No negative reviews found.")
        else:
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(stop_words='english', max_features=20)
            X = vectorizer.fit_transform(neg_reviews)
            keywords = pd.DataFrame(
                {"Keyword": vectorizer.get_feature_names_out(), "Frequency": X.toarray().sum(axis=0)}
            ).sort_values(by="Frequency", ascending=False)
            fig = px.bar(keywords, x="Keyword", y="Frequency", title="Most Common Negative Feedback Themes")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("📤 Please upload a dataset to explore sentiment insights.")
