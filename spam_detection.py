
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Create a small sample dataset
data = {
    "message": [
        "Win a free prize now",
        "Claim your cash reward today",
        "You won a free lottery",
        "Click here to claim your gift",
        "Free entry in a lucky draw",
        "Congratulations you won a prize",
        "Get free money now",
        "Urgent claim your reward",
        "Win a new mobile phone",
        "You have won free tickets",
        "Claim your bonus now",
        "Free offer limited time",
        "Win cash by clicking here",
        "You are selected for a prize",
        "Get your free gift today",
        "Meeting starts at ten",
        "Please call me later",
        "Are you coming to college",
        "I will reach home soon",
        "Please send the notes",
        "Lunch is ready",
        "Can we meet tomorrow",
        "I am going to class",
        "Thank you for your help",
        "Please check your email",
        "I finished my homework",
        "See you this evening",
        "What time is the meeting",
        "Please bring your notebook",
        "I will call you tomorrow"
    ],
    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham"
    ]
}

df = pd.DataFrame(data)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# 3. Create and train the ML model
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

# 4. Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Spam Message Detection")
print("Test Accuracy:", round(accuracy * 100, 2), "%")

# 5. Predict a new message
message = input("\nEnter a message: ")
result = model.predict([message])[0]

if result == "spam":
    print("Prediction: SPAM MESSAGE")
else:
    print("Prediction: NORMAL MESSAGE")
