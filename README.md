# Movie Recommender System

Small Streamlit app that recommends movies using content-based filtering (CountVectorizer + cosine similarity). Built from TMDB 5000 dataset and saved artifacts.

Contents:
- `app.py` - Streamlit application to pick a movie and get recommendations.
- `mrs.ipynb` - Notebook used to prepare data and build the model.
- `movie_dict.pkl`, `similarity.pkl` - Pickled artifacts used by the app.
- `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv` - Original datasets.

Quick start (Windows, virtualenv):

1. Activate your virtualenv:

   - PowerShell: `myenv\Scripts\Activate.ps1`

2. Install dependencies (example):

   - `pip install streamlit pandas scikit-learn nltk`

3. Run the app:

   - `streamlit run app.py`

Notes:
- This repo includes pickled model artifacts (`*.pkl`). If you prefer not to store them in Git, remove them before committing the remote or add them to `.gitignore`.
- To create a GitHub repo and push, see the section below or use the `gh` CLI.

License: Unlicensed — add a LICENSE file if you want an open-source license.
