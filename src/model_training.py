import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load processed dataset
data = pd.read_csv("data/features_news.csv")


# Remove rows with missing text
data = data.dropna(subset=["cleaned_text"])


# Input and output
X = data["cleaned_text"]
y = data["label"]


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=5000
)


# Convert text into numerical features
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000
)


# Train the model
model.fit(X_train_tfidf, y_train)


# Make predictions
y_pred = model.predict(X_test_tfidf)


# Save model and TF-IDF vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")


# Save test data and predictions for evaluation
evaluation_data = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred
})

evaluation_data.to_csv(
    "data/evaluation_results.csv",
    index=False
)


# Display results
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nTF-IDF shape:")
print(X_train_tfidf.shape)

print("\nFirst 10 predictions:")
print(y_pred[:10])

print("\nModel saved successfully!")
print("TF-IDF vectorizer saved successfully!")
print("Evaluation data saved successfully!")