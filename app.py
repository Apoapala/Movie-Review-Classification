from flask import Flask, request
import joblib

app = Flask(__name__)

# Load your pre-trained model and vectorizer
model = joblib.load('imdb_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Home route
@app.route('/', methods=['GET', 'POST'])
def predict_sentiment():
    prediction = None
    if request.method == 'POST':
        review = request.form['review']
        review_tfidf = vectorizer.transform([review])
        pred = model.predict(review_tfidf)[0]
        pred_prob = model.predict_proba(review_tfidf)[0][pred]
        sentiment = 'Positive' if pred == 1 else 'Negative'
        prediction = f'{sentiment} ({pred_prob:.2%} Confidence)'

    # Simple embedded HTML
    return '''
    <!doctype html>
    <html>
        <head>
            <title>Sentiment Analysis Predictor</title>
        </head>
        <body>
            <h1>Movie Review Sentiment Predictor</h1>
            <form method="POST">
                <textarea name="review" rows="10" cols="80" placeholder="Enter review here..." required></textarea><br><br>
                <button type="submit">Predict</button>
            </form>
            {result}
        </body>
    </html>
    '''.format(result=f'<h2>Prediction Result: {prediction}</h2>' if prediction else '')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
