import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')

# Load the IMDB dataset
data = pd.read_csv(r'IMDB Dataset.csv')

# Preprocess the text data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

data['cleaned_review'] = data['review'].apply(preprocess_text)

# Train sentiment analysis model
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(data['cleaned_review'])

sentiment_model = LogisticRegression()
sentiment_model.fit(features, data['sentiment'])

# Function to classify user input
def classify_review(review):
    cleaned_review = preprocess_text(review)
    feature = vectorizer.transform([cleaned_review])
    prediction = sentiment_model.predict(feature)[0]
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        sentiment = classify_review(review)
        if sentiment == 'positive':
            result = "The review is positive!! 😀😀"
        else:
            result = "The review is negative 😔😔"
        return render_template('cs_index.html', result=result)
    return render_template('cs_index.html')

if __name__ == "__main__":
    app.run(debug=True)
