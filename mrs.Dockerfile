FROM python:3.10

WORKDIR /app

COPY requirements-preprocessor.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements-preprocessor.txt

COPY mrs.py tmdb_5000_movies.csv tmdb_5000_credits.csv app.py ./

RUN python -m nltk.downloader punkt stopwords averaged_perceptron_tagger || true

CMD ["python", "mrs.py"]