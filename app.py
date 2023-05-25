import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("transcripts.csv")

def url_text_extraction(text):
    text = text.split("/")[-1]
    text = text.replace("_", " ")
    text = re.sub(r"\n", "", text)
    text = text.upper()
    return text

df["url"] = df["url"].apply(url_text_extraction)

transcripts = df["transcript"].tolist()
tfidf = TfidfVectorizer(stop_words="english")
uni_matrix = tfidf.fit_transform(transcripts)
cos_sim = cosine_similarity(uni_matrix)

def recommended_articles(title, n=5):
	article_index = df[df["url"] == title].index[0]
	article_similarities = cos_sim[article_index]
	top_indices = article_similarities.argsort()[::-1][1:n+1]
	recommended_titles = [df["url"].loc[i] for i in top_indices]
	return recommended_titles

st.title("Ted Talks Recommendation System")

user_title = st.text_input("Title")
n = st.number_input("Recommend Count")

if st.button("Recommend"):
	user_title = user_title.upper()
	recommended_titles = recommended_articles(user_title, int(n))
	st.write("Recommended:")
	for title in recommended_titles:
		st.success("- " + title)

