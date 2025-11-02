import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters except space
    tokens = text.split()
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_and_balance_data(input_file):
    # Load raw data
    df = pd.read_excel(input_file)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['cleaned_review'] = df['review'].apply(clean_text).fillna('')

    # Encode sentiment labels (update logic to your dataset's rating)
    df['sentiment'] = df['rating'].apply(lambda r: 0 if r <= 2 else 1 if r == 3 else 2)

    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(df['cleaned_review'])
    y = df['sentiment']

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_tfidf, y)

    # Recover rows by matching TF-IDF vectors (approximate)
    indices_resampled = []
    idx_map = {tuple(row): idx for idx, row in enumerate(X_tfidf.toarray())}

    for row in X_res.toarray():
        idx = idx_map.get(tuple(row))
        if idx is not None:
            indices_resampled.append(idx)

    balanced_df = df.iloc[indices_resampled].copy()
    balanced_df['sentiment'] = y_res

    print("Class distribution after balancing:")
    print(balanced_df['sentiment'].value_counts())

    balanced_df.to_csv('cleaned_balanced_reviews.csv', index=False)
    print("Preprocessed and balanced data saved to cleaned_balanced_reviews.csv")

if __name__ == "__main__":
    INPUT_FILE = 'chatgpt_style_reviews_dataset.xlsx'  # Update path if needed
    preprocess_and_balance_data(INPUT_FILE)
