import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv("news_data.csv")

# Text preprocessing
# Combine title and text columns
data['combined_text'] = data['title'] + ' ' + data['content']

# Define a function for preprocessing text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into text
    text = ' '.join(tokens)
    return text

# Apply text preprocessing
data['processed_text'] = data['combined_text'].apply(preprocess_text)

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['processed_text'])

# Apply clustering algorithm
num_clusters = 4  # You can adjust the number of clusters as per your requirement
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)
clusters = kmeans.labels_

# Add cluster labels to the dataset
data['cluster'] = clusters

# Streamlit app
st.title('News Clustering App ||R204454G HAI Tungamiraishe Mukwena')

# Display clusters
selected_cluster = st.selectbox('Select a cluster', sorted(set(clusters)))
cluster_data = data[data['cluster'] == selected_cluster]

# Display related stories in the selected cluster
st.subheader('Related Stories:')
for i, row in cluster_data.iterrows():
    st.write(row['title'])
    st.write(row['link'])
    st.write('---')
