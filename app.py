import pandas as pd
import streamlit as st
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

df = pd.read_csv("faq.csv")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_words)

df["Processed_Question"] = df["Question"].apply(preprocess_text)

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(df["Processed_Question"])

def get_answer(user_question):
    processed_question = preprocess_text(user_question)
    user_vector = vectorizer.transform([processed_question])

    similarities = cosine_similarity(user_vector, faq_vectors)
    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score < 0.2:
        return "Sorry, I could not find a relevant answer. Please try asking in a different way.", best_score

    answer = df.iloc[best_match_index]["Answer"]
    return answer, best_score

st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")

st.title("FAQ Chatbot")
st.write("Ask any university-related question and get the most relevant answer.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input("Ask your question:")

if st.button("Send"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        answer, score = get_answer(user_question)

        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Bot", answer))

        st.success(answer)
        st.info(f"Confidence Score: {round(score * 100, 2)}%")

st.subheader("Chat History")

for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.write(f"**You:** {message}")
    else:
        st.write(f"**Bot:** {message}")