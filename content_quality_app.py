import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt

# Step 1: Data Loading and Cleaning
file_path = 'UScomments.csv'
youtube_data = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
youtube_data_cleaned = youtube_data.dropna(subset=['comment_text'])
youtube_data_cleaned['likes'] = pd.to_numeric(youtube_data_cleaned['likes'], errors='coerce')
youtube_data_cleaned['replies'] = pd.to_numeric(youtube_data_cleaned['replies'], errors='coerce')
youtube_data_cleaned = youtube_data_cleaned.dropna(subset=['likes', 'replies'])
youtube_data_cleaned.reset_index(drop=True, inplace=True)

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)  # Remove special characters
    return text.lower()

youtube_data_cleaned['cleaned_text'] = youtube_data_cleaned['comment_text'].apply(clean_text)

# Step 2: Sentiment Analysis
download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    return 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'

youtube_data_cleaned['sentiment'] = youtube_data_cleaned['cleaned_text'].apply(vader_sentiment)
youtube_data_cleaned['is_positive'] = youtube_data_cleaned['sentiment'] == 'Positive'

# Step 3: Machine Learning Preparation
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(youtube_data_cleaned['cleaned_text'])
y = youtube_data_cleaned['is_positive']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 5: Save the Model
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Step 6: Streamlit App
st.title("YouTube Comment Sentiment Analysis")
user_input = st.text_area("Enter a YouTube comment:")
if st.button("Analyze"):
    # Load the saved model
    with open('sentiment_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    # Clean user input
    cleaned_input = clean_text(user_input)

    # Transform input using TF-IDF
    user_tfidf = tfidf.transform([cleaned_input])

    # Predict sentiment
    prediction = loaded_model.predict(user_tfidf)
    sentiment = 'Positive' if prediction[0] else 'Negative'

    st.write(f"Sentiment: {sentiment}")

# Step 7: Visualizations
st.write("Sentiment Distribution")
sentiment_counts = youtube_data_cleaned['sentiment'].value_counts()

# Replace Streamlit bar_chart with Matplotlib visualization
fig, ax = plt.subplots()
sentiment_counts.plot(kind='bar', ax=ax, title="Sentiment Distribution")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
st.pyplot(fig)

# Word Frequency for Positive Comments
positive_comments = youtube_data_cleaned[youtube_data_cleaned['sentiment'] == 'Positive']
text = " ".join(positive_comments['cleaned_text'])
text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
word_counts = Counter(text.split())
most_common_words = word_counts.most_common(20)

words, counts = zip(*most_common_words)
fig, ax = plt.subplots()
ax.barh(words, counts)
ax.set_xlabel("Frequency")
ax.set_ylabel("Words")
ax.set_title("Most Frequent Words in Positive Comments")
ax.invert_yaxis()
st.pyplot(fig)


#run in streamlit by navigating to folder in terminal and pasting
#python -m streamlit run content_quality_app.py