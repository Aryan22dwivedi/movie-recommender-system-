# mrs.py

import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --- 1. Data Loading ---

# Note: This script expects 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv'
# to be in the same directory.
try:
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
except FileNotFoundError:
    print("Error: Make sure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in the same folder as this script.")
    exit()

print("--- Initial Data Loaded ---")
print("Movies DataFrame Head:")
print(movies.head())
print("\nOverview column type:")
print(type(movies['overview']))


# --- 2. Data Preprocessing & Merging ---

movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

print("\n--- Merged & Trimmed DataFrame ---")
print(movies.head())

# Handle missing data
print("\nNull values before cleaning:")
print(movies.isnull().sum())
movies.dropna(inplace=True)
print("\nNull values after cleaning (should be all 0):")
print(movies.isnull().sum())

print("\nExample of 'genres' data before conversion:")
print(movies.iloc[0].genres)


# --- 3. Helper Functions ---

def convert(obj):
    """Extracts the 'name' from a list of dictionaries stored as a string."""
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except:
        pass 
    return L

def convert3(obj):
    """Extracts the names of the first 3 cast members."""
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
    except:
        pass
    return L

def fetch_director(obj):
    """Extracts the director's name from the crew list."""
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except:
        pass
    return L

def stem(text):
    """Stems a string of text."""
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# --- 4. Data Transformation ---

print("\n--- Transforming Data ---")

# Apply helper functions to clean JSON-like string columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)

print("DataFrame after 'genres', 'keywords', 'cast', 'crew' conversion:")
print(movies.head())

# Split overview into a list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces from list items (to create single tags like "ScienceFiction")
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

print("\nDataFrame after removing spaces from tags:")
print(movies.head())

# Combine columns into a single 'tags' column
movies['tags'] = movies['overview'] + movies['cast'] + movies['genres'] + movies['keywords'] + movies['crew']

# Create the final DataFrame
# Use .copy() to avoid SettingWithCopyWarning
new_df = movies[['movie_id', 'title', 'tags']].copy()

# Convert 'tags' list into a single string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
# Convert to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

print("\nFinal DataFrame with 'tags' column (string format):")
print(new_df.head())

# Stem the 'tags'
ps = PorterStemmer()
print("\nStemming 'tags' column (this may take a moment)...")
new_df['tags'] = new_df['tags'].apply(stem)
print("Stemming complete.")

print("\nExample of stemmed 'tags' for Avatar:")
print(new_df.iloc[0].tags)


# --- 5. Vectorization & Similarity Model ---

print("\n--- Building Model ---")

# Initialize CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')

# Fit and transform the 'tags' data
vectors = cv.fit_transform(new_df['tags']).toarray()

print("Vector shape:", vectors.shape)
print("Example vector (first movie):", vectors[0])

# Calculate Cosine Similarity
print("\nCalculating cosine similarity (this may take a moment)...")
similarity = cosine_similarity(vectors)
print("Similarity matrix shape:", similarity.shape)

# Test similarity output for the first movie
print("\nTop 5 similar movies to the first movie (by index):")
print(sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:6])


# --- 6. Recommendation Function ---

def recommend(movie):
    """Recommends 5 similar movies based on the input movie title."""
    try:
        movie_index = new_df[new_df['title'] == movie].index[0]
    except IndexError:
        print(f"Error: Movie '{movie}' not found in the dataset.")
        return
        
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    print(f"\n--- Recommendations for {movie} ---")
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    print("---------------------------------")


# --- 7. Run Recommendation & Save Files ---

# Test the recommend function
recommend('Avatar')
recommend('The Dark Knight Rises')

# Save the processed DataFrame and similarity matrix for later use
print("\nSaving movie dictionary and similarity matrix to .pkl files...")
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Script complete. 'movie_dict.pkl' and 'similarity.pkl' saved.")