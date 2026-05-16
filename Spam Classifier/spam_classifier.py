import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep important columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels into numbers
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Input and output
X = df['message']
y = df['label']

# Convert text into vectors
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)

# Prediction on test data
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Custom message
msg = ["Congratulations! You won a free iPhone"]

# Convert message into vector
msg_input = vectorizer.transform(msg)

# Predict
prediction = model.predict(msg_input)

# Output result
if prediction[0] == 1:
    print("\nSpam Message")
else:
    print("\nNot Spam")