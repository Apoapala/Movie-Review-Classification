# IMDB Sentiment Analysis ML Pipeline

## Overview

This project implements an end-to-end machine learning pipeline to predict the sentiment of IMDB movie reviews. The solution encompasses data loading, preprocessing (cleaning, tokenization, and TF-IDF vectorization), model training using Logistic Regression, evaluation using standard classification metrics, and deployment via a Flask API. The objective is to build an accessible and explainable sentiment analysis model that classifies reviews as **Positive** or **Negative**.

## Repository Structure

```
imdb-sentiment-analysis/
├── IMDB Dataset.csv            # IMDB movie reviews dataset (optional)
├── imdb_sentiment_model.pkl    # Trained Logistic Regression model 
│── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
├── train_model.py              # Script for data loading, preprocessing, training, and evaluation
├── sentiment_analysis.ipynb    # Jupyter Notebook for exploration and model training
├── app.py                      # Flask API code for serving predictions
├── README.md                       # This file
└── requirements.txt                # Project dependencies
```

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Apoapala/Movie-Review-Classification.git
   cd Movie-Review-Classification
   ```

2. **(Optional) Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Model Training

The main training script is located in the `src/` directory. This script performs the following:
- Loads the IMDB movie reviews dataset.
- Cleans and preprocesses the text (e.g., removing stop words, punctuation, lemmatization).
- Splits the data into training, validation, and test sets.
- Converts reviews to numerical features using TF-IDF vectorization.
- Trains a Logistic Regression classifier and evaluates it using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Saves the trained model and TF-IDF vectorizer in the `models/` folder.

To train the model, run:

```bash
python train_model.py
```

### API Deployment

The Flask API is located in the `app.py` file. It:
- Loads the pre-trained model and TF-IDF vectorizer.
- Provides a simple web interface for users to enter a movie review.
- Processes the review input and returns the predicted sentiment.

To run the Flask API, execute:

```bash
python app.py
```

Then open your browser and navigate to [http://localhost:5000](http://localhost:5000) (or the port specified in the code).

## How to Use

1. **Data Input:**  
   Enter a movie review into the text box provided on the web interface.

2. **Prediction:**  
   Click the "Predict Sentiment" button. The API will process the review and display whether the sentiment is **Positive** or **Negative** along with a confidence score.

## Evaluation Metrics

The trained model was evaluated on a held-out test set using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

These metrics indicate that the model performs robustly in distinguishing between positive and negative sentiments.

## References

- Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. *Foundations and Trends in Information Retrieval, 2*(1–2), 1–135. DOI: 10.1561/1500000011  
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.  
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Springer.

## License

This project is provided for educational purposes.

## Contact

For any questions or suggestions, please contact gloryazimbe@gmail.com