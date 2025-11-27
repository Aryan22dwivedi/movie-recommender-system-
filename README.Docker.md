# Movie Recommender System

## Quick Start

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