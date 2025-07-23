#!/usr/bin/env python3
# seo_keyword_agent.py
# Analizza le keyword per le pagine del sito e invia i suggerimenti a WordPress

import os, json, base64, time, re, warnings, ast, requests
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from dataforseo_client import RestClient            # pip install dataforseo-client

# ─────────────────── 1️⃣  SILENZIA WARNING RIPETITIVO ────────────────────
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=re.escape("'force_all_finite' was renamed to 'ensure_all_finite'")
)

# ─────────────────── 2️⃣  CONFIGURAZIONE ──────────────────────────────────
load_dotenv()                                       # carica variabili da .env

API_ROOT = "https://www.asidatamyte.it/wp-json/wp/v2/keyword_suggestion"
AUTH_HDR = {
    "Authorization": "Basic " + base64.b64encode(
        f"{os.getenv('WP_USER')}:{os.getenv('WP_APP_PWD')}".encode()
    ).decode()
}

DFS_LOGIN = os.getenv("DFS_LOGIN")
DFS_PWD   = os.getenv("DFS_PWD")
dfs_client = RestClient(login=DFS_LOGIN, password=DFS_PWD)

embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# ─────────────────── 3️⃣  LEGGI CSV ───────────────────────────────────────
df = pd.read_csv("pages.csv", encoding="utf-8")
print(f"Tot pagine da processare: {len(df)}")

# ─────────────────── 4️⃣  LOOP PAGINE ─────────────────────────────────────
for idx, row in enumerate(df.itertuples(), 1):
    print(f"[{idx}/{len(df)}] Pagina: {row.pagina}")

    # --- gestisci lista keyword di partenza robusta ----------------------
    raw = (row.keywords_di_partenza or "[]").strip()
    try:
        seeds = json.loads(raw)
    except json.JSONDecodeError:
        try:
            seeds = ast.literal_eval(raw)           # ['kw1','kw2']
        except Exception:
            seeds = []

    # --- chiamata DataForSEO --------------------------------------------
    post_data = [{
        "keywords": (seeds or [row.seed_topic])[:200],
        "language_name": "Italian",
        "location_code": 2250                       # Italy
    }]
    try:
        resp = dfs_client.post(
            "/v3/keywords_data/google/keywords_for_keywords/live",
            post_data
        )
    except Exception as e:
        print("❌ DataForSEO error:", e)
        continue                                   # passa alla pagina dopo

    time.sleep(1)                                  # rispetto rate-limit
    items = resp["tasks"][0]["result"][0]["items"]
    if not items:
        print("⚠️  Nessuna keyword trovata per", row.pagina)
        continue

    # --- embeddings & clustering ----------------------------------------
    keywords = [it["keyword"] for it in items]
    embeddings = embedder.encode(keywords, show_progress_bar=False)
    labels = HDBSCAN(min_cluster_size=8).fit_predict(embeddings)

    # --- invio a WordPress ----------------------------------------------
    for it, lab in zip(items, labels):
        kd  = int(float(it["competition"]) * 100)  # 0-100
        vol = it["search_volume"]
        opp = round((vol * 0.6 - kd * 0.4) / 100, 2)

        data = {
            "title": it["keyword"],
            "status": "publish",
            "fields": {
                "page":        row.pagina,
                "cluster":     int(lab),
                "intent":      "informational",   # TODO: rilevare intent reale
                "difficulty":  kd,
                "opportunity": opp
            }
        }
        r = requests.post(API_ROOT, headers=AUTH_HDR, json=data, timeout=30)
        try:
            r.raise_for_status()                  # HTTP 201 atteso
            print("✅ POST", it["keyword"][:50], r.status_code)
        except requests.HTTPError as e:
            print("❌ WP error:", e, r.text[:120])

print("🎉 Script completato")
