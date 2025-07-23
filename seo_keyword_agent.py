import os, json, base64, requests, pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN

# 1️⃣  Config
load_dotenv()                                           # Carica .env
API_ROOT = "https://tuodominio.it/wp-json/wp/v2/keyword_suggestion"
auth = base64.b64encode(f"{os.getenv('WP_USER')}:{os.getenv('WP_APP_PWD')}".encode()).decode()

# 2️⃣  Modelli
embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# 3️⃣  Carica le 26 pagine
df = pd.read_csv("pages.csv")

# 4️⃣  Ciclo per pagina
for _, row in df.iterrows():
    seed   = row["seed_topic"]
    page   = row["pagina"]
    seeds  = json.loads(row["keywords_di_partenza"])
    # …chiamata DataForSEO 'keywords_for_keywords/live' qui…
    keywords = ["demo", "keyword"]    # placeholder

    # Calcola embeddings e cluster
    emb = embedder.encode(keywords)
    labels = HDBSCAN(min_cluster_size=8).fit_predict(emb)

    # 5️⃣  Invio a WordPress
    for kw, lab in zip(keywords, labels):
        data = {
            "title": kw,
            "status": "publish",
            "fields": {
                "page": page,
                "cluster": int(lab),
                "intent": "informational",         # placeholder LLM
                "difficulty": 32,                  # placeholder API
                "opportunity": 68                  # placeholder formula
            }
        }
        r = requests.post(API_ROOT,
                          headers={"Authorization": f"Basic {auth}"},
                          json=data, timeout=30)
        r.raise_for_status()                       # → HTTP 201