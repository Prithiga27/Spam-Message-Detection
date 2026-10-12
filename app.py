
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩"
)

st.title("📩 Spam Message Detection")
st.write("Enter a message to check whether it may be spam or normal.")

# Sample training dataset
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
    "label": ["spam"] * 15 + ["ham"] * 15
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.metric("Demo test accuracy", f"{accuracy:.0%}")
st.caption("This score is based on a small sample dataset, not real-world performance.")

user_message = st.text_area(
    "Enter your message:",
    placeholder="Example: Please send me the notes"
)

if st.button("Check Message"):
    if user_message.strip():
        result = model.predict([user_message])[0]

        if result == "spam":
            st.warning("Prediction: Potential SPAM")
        else:
            st.success("Prediction: Likely NORMAL")

        st.caption("This is a basic ML demo. Predictions may be incorrect.")
    else:
        st.info("Please enter a message first.")
