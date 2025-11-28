# Movie Recommender System

Small Streamlit app that recommends movies using content-based filtering (CountVectorizer + cosine similarity). Built from TMDB 5000 dataset and saved artifacts.

Contents:
- `app.py` - Streamlit application to pick a movie and get recommendations.
- `mrs.ipynb` - Notebook used to prepare data and build the model.
- `movie_dict.pkl`, `similarity.pkl` - Pickled artifacts used by the app.
- `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv` - Original datasets.

Quick start (Windows, virtualenv):to run traditionally


1. Activate your virtualenv:

   - PowerShell: `myenv\Scripts\Activate.ps1`

2. Install dependencies (example):

   - `pip install streamlit pandas scikit-learn nltk`

3. Run the app:

   - `streamlit run app.py`
# Movie Recommender System

## Quick Start : to run through docker

\`\`\`bash
docker compose up
\`\`\`

Open http://localhost:8501

## Docker Hub Images

- `aryan22dwivedi/mrs-preprocessor-app:latest`
- `aryan22dwivedi/mrs-working-app:latest`

## Run Externally
```powershell
docker run -d -p 8501:8501 -v mrs_mrs-data:/app aryan22dwivedi/mrs-app:latest
```
\`\`\`
