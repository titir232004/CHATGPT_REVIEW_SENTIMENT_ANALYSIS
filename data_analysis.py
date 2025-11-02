import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def categorize_sentiment(rating):
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'

def analyze_data():
    df = pd.read_csv('cleaned_balanced_reviews.csv')
    df['sentiment'] = df['rating'].apply(categorize_sentiment)

    # Sentiment distribution plot
    plt.figure(figsize=(6,4))
    df['sentiment'].value_counts().plot(kind='bar', color=['red', 'gray', 'green'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.show()

    # Word cloud of cleaned reviews
    all_words = ' '.join(df['cleaned_review'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    plt.figure(figsize=(10,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Cleaned Reviews')
    plt.show()

    # Histograms for review length and helpful votes
    fig, axs = plt.subplots(1, 2, figsize=(12,4))
    axs[0].hist(df['review_length'], bins=30, color='skyblue')
    axs[0].set_title('Review Length Distribution')
    axs[0].set_xlabel('Characters')
    axs[0].set_ylabel('Count')

    axs[1].hist(df['helpful_votes'], bins=30, color='lightgreen')
    axs[1].set_title('Helpful Votes Distribution')
    axs[1].set_xlabel('Number of Helpful Votes')
    axs[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_data()
