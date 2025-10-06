import torch
import pickle
from sentence_transformers import SentenceTransformer
import streamlit as st


model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

with open(r"C:\Users\yigit\Downloads\recommendation_model.pkl", 'rb') as f:
    components = pickle.load(f)

embeddings = components['embeddings']
tfidf_reduced = components['embeddings_reduced']
pca = components['pca']
knn = components['knn']
df_cleaned = components['df_cleaned']

def recommend_by_abstract(input_text, top_n=10):
    input_emb = model.encode([input_text])
    input_red = pca.transform(input_emb)
    distances, indices = knn.kneighbors(input_red, n_neighbors=top_n)
    recommendations = df_cleaned.iloc[indices[0]]
    return recommendations[['title','authors','year']]

st.title("📚 Paper Recommendation System")
st.write("Enter your query and I'll show you the best recommended articles!")


query = st.text_input("Query (abstract, topic etc.):")

top_n = st.slider("How many recommendations you want?", min_value=1, max_value=10, value=5)


if st.button("Get Recommendations") and query:
    results = recommend_by_abstract(query, top_n=top_n)
    
    st.write(f"🔍 **Query:** {query}")
    st.write("="*80)
    
    for i, row in results.iterrows():
        st.write(f"**{i+1}. {row['title']}**")
        st.write(f"👥 Authors: {row['authors']}")
        st.write(f"📅 Year: {row['year']}")
        st.write("---")