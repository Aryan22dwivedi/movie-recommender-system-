<<<<<<< HEAD
FROM python:3.10-slim

WORKDIR /app

COPY requirements-app.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements-app.txt

COPY app.py ./

EXPOSE 8501

=======
FROM python:3.10

WORKDIR /app

COPY requirements-app.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements-app.txt

COPY app.py ./

EXPOSE 8501

>>>>>>> fd74184109a3b04e80973a22a58d45c165a94567
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]