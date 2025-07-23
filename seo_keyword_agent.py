import os, json, base64, time, requests, re, warnings
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from dataforseo_client import RestClient  # oppure dataforseo_sdk

# --------- 1️⃣  SILENZIA IL WARNING RIPETITIVO ----------------
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=re.escape("'force_all_finite' was renamed to 'ensure_all_finite'")
)

# --------- 2️⃣  CONFIG ----------------------------------------
load_dotenv()
API_ROOT = "https://www.asidatamyte.it/wp-json/wp/v2/keyword_suggestion"
AUTH = base64.b64encode(f"{os.getenv('WP_USER')}:{os.getenv('WP_APP_PWD')}".encode()).decode()

DFS_LOGIN = os.getenv("DFS_LOGIN")
DFS_PWD   = os.getenv("DFS_PWD")
client    = RestClient(login=DFS_LOGIN, password=DFS_PWD)

embedder  = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# --------- 3️⃣  LEGGI IL CSV (UTF-8) ---------------------------
df = pd.read_csv("pages.csv", encoding="utf-8")

# --------- 4️⃣  CICLA SULLE PAGINE -----------------------------
for _, row in df.iterrows():
    page  = row["pagina"]
    topic = row["seed_topic"]

    # gestisci keywords_di_partenza robusto
    raw = (row.get("keywords_di_partenza") or "[]").strip()
    try:
        seeds = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import ast
            seeds = ast.literal_eval(raw)
        except Exception:
            seeds = []

    # ------ (A) CHIAMATA DATAFORSEO ----------------------------
    post_data = [{
        "keywords": (seeds or [topic])[:200],
        "language_name": "Italian",
        "location_code": 2250
    }]
    resp = client.post(
        "/v3/keywords_data/google/keywords_for_keywords/live",
        post_data
    )
    time.sleep(1)                       # rispetto dei rate-limit
    kw_items = resp["tasks"][0]["result"][0]["items"]

    # ------ (B) ELABORA & INVIA A WORDPRESS --------------------
    keywords = [item["keyword"] for item in kw_items]
    emb      = embedder.encode(keywords)
    labels   = HDBSCAN(min_cluster_size=8).fit_predict(emb)

    for item, lab in zip(kw_items, labels):
        kd  = int(float(item["competition"]) * 100)
        vol = item["search_volume"]
        opp = round((vol*0.6 - kd*0.4) / 100, 2)

        data = {
            "title": item["keyword"],
            "status": "publish",
            "fields": {
                "page": page,
                "cluster": int(lab),
                "intent": "informational",  # placeholder
                "difficulty": kd,
                "opportunity": opp
            }
        }
        r = requests.post(API_ROOT,
                          headers={"Authorization": f"Basic {AUTH}"},
                          json=data, timeout=30)
        r.raise_for_status()    # 201 Created
        print("POST", item["keyword"], r.status_code)
