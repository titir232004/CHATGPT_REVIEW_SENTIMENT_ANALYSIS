# app.py — Full robust Streamlit dashboard for your 10 questions
import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="ChatGPT Reviews Sentiment Dashboard", page_icon="📊", layout="wide")
st.title("📊 ChatGPT Reviews Sentiment Analysis Dashboard")
st.markdown("Gain insights from user feedback using AI-powered sentiment analytics.")
st.sidebar.header("Data & Filters")

uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV or XLSX) — required", type=["csv", "xlsx"])

date_range = st.sidebar.date_input("Filter by date range (optional)", value=[])

rating_options = ["1", "2", "3", "4", "5"]
selected_ratings = st.sidebar.multiselect("Filter by star rating (if available)", rating_options, default=rating_options)


st.sidebar.markdown("---")
st.sidebar.header("Key Questions")
question = st.sidebar.selectbox(
    "Select an analysis question",
    [
        "1. Overall sentiment of user reviews",
        "2. How does sentiment vary by rating?",
        "3. Keywords/phrases per sentiment class",
        "4. How has sentiment changed over time?",
        "5. Verified vs Non-Verified sentiment",
        "6. Are longer reviews more likely to be negative or positive?",
        "7. Which locations show most positive/negative sentiment?",
        "8. Difference across platforms (Web vs Mobile)?",
        "9. Which ChatGPT versions are associated with higher/lower sentiment?",
        "10. Most common negative feedback themes"
    ],
)


@st.cache_resource
def try_load_model(model_path="./bert_sentiment_model"):
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return model, tokenizer
    except Exception as e:
        return None, None

model, tokenizer = try_load_model()


def safe_read(uploaded_file):
    try:
        if uploaded_file.name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
        else:
            return pd.read_csv(uploaded_file, encoding="utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

def normalize_verified_val(x):
    if pd.isna(x):
        return "Unknown"
    if isinstance(x, str):
        v = x.strip().lower()
        if v in ("yes", "y", "true", "t", "1", "verified"):
            return "Verified"
        if v in ("no", "n", "false", "f", "0", "not verified"):
            return "Not verified"
        return "Unknown"
    if isinstance(x, (int, float)):
        try:
            if int(x) == 1:
                return "Verified"
            if int(x) == 0:
                return "Not verified"
        except:
            pass
    if isinstance(x, bool):
        return "Verified" if x else "Not verified"
    return "Unknown"

def ensure_predicted_sentiment(df):
    """
    Return df with Predicted_Sentiment column. If model loaded, predict,
    else expect user provided Predicted_Sentiment.
    """
    if "Predicted_Sentiment" in df.columns:
        # make sure labels are string
        df["Predicted_Sentiment"] = df["Predicted_Sentiment"].astype(str)
        return df
    if model is None or tokenizer is None:
        st.warning("No local model found and no Predicted_Sentiment column in the dataset. Upload model or include Predicted_Sentiment.")
        return df
    # predict in batches (to avoid OOM for large files)
    texts = df["review"].astype(str).tolist()
    batch_size = 32
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**enc)
            batch_preds = outputs.logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(batch_preds)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    df["Predicted_Sentiment"] = [label_map.get(int(p), str(p)) for p in preds]
    return df

def apply_filters(df):

    if "rating" in df.columns:

        df["rating_str"] = df["rating"].astype(str).str.strip().str.replace(".0", "", regex=False)
        if selected_ratings:
            df = df[df["rating_str"].isin(selected_ratings)]

    if len(date_range) == 2:
        start, end = date_range

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
    return df

if not uploaded_file:
    st.info("Please upload a dataset (CSV or XLSX). Expected column: 'review'. Optional columns: rating, date, verified_purchase, review_length, platform, location, version, true_sentiment, Predicted_Sentiment.")
    st.stop()

df = safe_read(uploaded_file)

if "review" not in df.columns:
    st.error("Your file must include a 'review' column (text of the review).")
    st.stop()

df["review"] = df["review"].fillna("").astype(str)

df = ensure_predicted_sentiment(df)

if "review_length" not in df.columns:
    df["review_length"] = df["review"].astype(str).apply(len)

verified_candidates = [c for c in df.columns if c.lower() in ("verified_purchase", "verified", "is_verified")]
if verified_candidates:
    vcol = verified_candidates[0]
    df["_verified_norm"] = df[vcol].apply(normalize_verified_val)
else:
    df["_verified_norm"] = "Unknown"

df = apply_filters(df)

df = df.reset_index(drop=True)

st.header("Dataset snapshot")
col1, col2, col3 = st.columns(3)
col1.metric("Total reviews (after filters)", len(df))
col2.metric("Unique locations", df["location"].nunique() if "location" in df.columns else "N/A")
col3.metric("Has model", "Yes" if model is not None else "No (using Predicted_Sentiment column)")

# Q1: overall sentiment proportions
if question.startswith("1.") or question.strip() == "1":
    st.subheader("1️⃣ Overall sentiment of user reviews")
    counts = df["Predicted_Sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    df_counts = counts.reset_index()
    df_counts.columns = ["Sentiment", "Count"]  # ensure consistent names
    fig = px.bar(df_counts, x="Sentiment", y="Count", color="Sentiment",
                 color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
                 title="Overall sentiment counts")

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_counts)

# Q2: sentiment variation by rating
elif question.startswith("2"):
    st.subheader("2️⃣ How does sentiment vary by rating?")
    if "rating" not in df.columns:
        st.warning("No 'rating' column found. If you have ratings, include a 'rating' column in the dataset.")
    else:
        # normalize rating values
        df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
        rating_group = df.groupby("rating_num")["Predicted_Sentiment"].value_counts(normalize=False).unstack().fillna(0)
        # bar chart grouped
        fig = px.bar(rating_group, barmode="group", title="Sentiment counts per star rating")
        st.plotly_chart(fig, use_container_width=True)
        # Also show mismatch examples where rating is low but sentiment predicted positive etc.
        st.markdown("#### Examples of possible mismatches (low rating but positive text / high rating but negative text)")
        mismatches = df[
            ((df["rating_num"] <= 2) & (df["Predicted_Sentiment"] == "Positive")) |
            ((df["rating_num"] >= 4) & (df["Predicted_Sentiment"] == "Negative"))
        ][["rating", "review", "Predicted_Sentiment"]].head(10)
        if mismatches.empty:
            st.info("No obvious mismatches found in top 10.")
        else:
            st.dataframe(mismatches)

# Q3: keywords/phrases per sentiment class
elif question.startswith("3"):
    st.subheader("3️⃣ Keywords or phrases most associated with each sentiment class")
    sentiments = ["Positive", "Neutral", "Negative"]
    for s in sentiments:
        st.markdown(f"**{s} reviews**")
        text = " ".join(df[df["Predicted_Sentiment"] == s]["review"].astype(str).tolist())
        if not text.strip():
            st.write("No reviews of this sentiment found.")
            continue
        wc = WordCloud(width=800, height=300, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# Q4: sentiment change over time
elif question.startswith("4"):
    st.subheader("4️⃣ How has sentiment changed over time?")
    if "date" not in df.columns:
        st.warning("No 'date' column found in dataset. Add a 'date' column for time-series analysis.")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df_time = df.dropna(subset=["date"])
        if df_time.empty:
            st.info("No valid dates after parsing.")
        else:
            # allow frequency selector
            freq = st.selectbox("Aggregation frequency", ["W", "M", "D"], index=0, format_func=lambda f: {"W":"Weekly","M":"Monthly","D":"Daily"}[f])
            trend = df_time.groupby(pd.Grouper(key="date", freq=freq))["Predicted_Sentiment"].value_counts().unstack().fillna(0)
            fig = px.line(trend, labels={"index": "date"}, title=f"Sentiment trend ({'Weekly' if freq=='W' else 'Monthly' if freq=='M' else 'Daily'})")
            st.plotly_chart(fig, use_container_width=True)

# Q5: verified vs non-verified
elif question.startswith("5"):
    st.subheader("5️⃣ Do verified users tend to leave more positive or negative reviews?")
    # we normalized earlier to _verified_norm
    if "_verified_norm" not in df.columns:
        st.warning("No verified info available.")
    else:
        c = df["_verified_norm"].value_counts().rename_axis("_verified_norm").reset_index(name="count")
        st.write("Counts of verification categories")
        st.dataframe(c)
        if "Predicted_Sentiment" in df.columns:
            fig = px.histogram(df, x="_verified_norm", color="Predicted_Sentiment", barmode="group",
                               category_orders={"_verified_norm": ["Verified", "Not verified", "Unknown"]},
                               title="Sentiment distribution by verification status")
            st.plotly_chart(fig, use_container_width=True)

            # percentage pivot
            pct = df.groupby("_verified_norm")["Predicted_Sentiment"].value_counts(normalize=True).mul(100).rename("percent").reset_index()
            pivot = pct.pivot(index="_verified_norm", columns="Predicted_Sentiment", values="percent").fillna(0)
            st.markdown("### Percentage breakdown by verification status")
            st.dataframe(pivot.round(2))

# Q6: review length vs sentiment
elif question.startswith("6"):
    st.subheader("6️⃣ Are longer reviews more likely to be negative or positive?")
    # compute char and word lengths (already created review_length; ensure numeric)
    df["_char_len"] = pd.to_numeric(df["review_length"], errors="coerce").fillna(df["review"].astype(str).apply(len))
    df["_word_len"] = df["review"].astype(str).apply(lambda x: len(x.split()))
    if "Predicted_Sentiment" in df.columns:
        fig_box = px.box(df, x="Predicted_Sentiment", y="_char_len", points="outliers",
                         title="Character length distribution by sentiment")
        st.plotly_chart(fig_box, use_container_width=True)
        mean_len = df.groupby("Predicted_Sentiment")["_char_len"].mean().reset_index()
        fig_bar = px.bar(mean_len, x="Predicted_Sentiment", y="_char_len", title="Average review length by sentiment", text="_char_len")
        fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("Top 5 longest reviews (by characters)")
        st.dataframe(df.sort_values("_char_len", ascending=False)[["review", "Predicted_Sentiment", "_char_len"]].head(5))
    else:
        st.warning("No Predicted_Sentiment in data to compare.")

# Q7: sentiment by location
elif question.startswith("7"):
    st.subheader("7️⃣ Which locations show the most positive or negative sentiment?")
    if "location" not in df.columns:
        st.warning("No 'location' column found.")
    else:
        # top N selector
        top_n = st.slider("Top N locations to show", min_value=5, max_value=50, value=10)
        top_locs = df["location"].value_counts().head(top_n).index.tolist()
        df_loc = df[df["location"].isin(top_locs)]
        if df_loc.empty:
            st.info("No data for top locations after filters.")
        else:
            summary = df_loc.groupby("location")["Predicted_Sentiment"].value_counts().unstack().fillna(0)
            fig = px.bar(summary, barmode="group", title="Sentiment by top locations")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(summary)

# Q8: sentiment by platform
elif question.startswith("8"):
    st.subheader("8️⃣ Is there a difference in sentiment across platforms (Web vs Mobile)?")
    if "platform" not in df.columns:
        st.warning("No 'platform' column found.")
    else:
        df["platform_norm"] = df["platform"].astype(str).str.strip().str.title().replace({"Nan": "Unknown"})
        fig = px.histogram(df, x="platform_norm", color="Predicted_Sentiment", barmode="group", title="Sentiment by platform")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.groupby("platform_norm")["Predicted_Sentiment"].value_counts().unstack().fillna(0))

# Q9: sentiment by version
elif question.startswith("9"):
    st.subheader("9️⃣ Which ChatGPT versions are associated with higher/lower sentiment?")
    if "version" not in df.columns:
        st.warning("No 'version' column found.")
    else:
        df["version_norm"] = df["version"].astype(str)
        fig = px.histogram(df, x="version_norm", color="Predicted_Sentiment", barmode="group", title="Sentiment by version")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.groupby("version_norm")["Predicted_Sentiment"].value_counts().unstack().fillna(0))

# Q10: negative feedback themes
elif question.startswith("10") or question.lower().startswith("10"):
    st.subheader("🔟 Most Common Negative Feedback Themes")

    negs = df[df["Predicted_Sentiment"] == "Negative"]["review"].astype(str)
    negs = negs.dropna()
    negs = negs[negs.str.strip() != ""]

    if negs.empty:
        st.info("No negative reviews found.")
    else:
        import re
        from sklearn.feature_extraction.text import CountVectorizer
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        clean_negs = negs.apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x.lower()))

        vec = CountVectorizer(stop_words="english", max_features=30)
        X = vec.fit_transform(clean_negs)
        keywords = pd.DataFrame({
            "keyword": vec.get_feature_names_out(),
            "count": X.toarray().sum(axis=0)
        }).sort_values("count", ascending=False)

        fig = px.bar(
            keywords, x="keyword", y="count", color="count",
            title="Top Keywords in Negative Reviews"
        )
        st.plotly_chart(fig, use_container_width=True)

        text = " ".join(clean_negs)
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.subheader("☁️ Word Cloud of Negative Feedback")
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)

        st.dataframe(keywords.head(20))


# End
st.markdown("---")
st.caption("Dashboard generated by local BERT model")
