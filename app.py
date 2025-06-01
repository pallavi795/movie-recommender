import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[['title', 'overview', 'genres']]
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].apply(lambda x: ' '.join([g['name'] for g in ast.literal_eval(x)]))
    df['content'] = df['overview'] + ' ' + df['genres']
    return df

@st.cache_resource
def get_cosine_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend(title, df, cosine_sim):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    return df['title'].iloc[[i[0] for i in sim_scores]].tolist()

# UI
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

df = load_data()
cosine_sim = get_cosine_similarity(df)

selected_title = st.selectbox("Choose a movie:", df['title'].sort_values().unique())

if st.button("Recommend"):
    st.subheader("Top Recommendations:")
    for i, movie in enumerate(recommend(selected_title, df, cosine_sim), 1):
        st.write(f"{i}. {movie}")
