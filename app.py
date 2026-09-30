import os
import json
import time
import sqlite3
from datetime import datetime
from urllib.parse import urlparse

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "meta/llama-3.2-90b-vision-instruct"
DB = "data.db"

st.set_page_config(page_title="AskData", page_icon="🔎", layout="wide")


# ---------- Database ----------
def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT, status TEXT, created_at TEXT,
            plan TEXT, error TEXT)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS rows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER, data TEXT)"""
        )


def create_run(prompt):
    with db() as c:
        cur = c.execute(
            "INSERT INTO runs (prompt, status, created_at) VALUES (?, 'running', ?)",
            (prompt, datetime.now().strftime("%Y-%m-%d %H:%M")),
        )
        return cur.lastrowid


def update_run(run_id, **kw):
    sets = ", ".join(f"{k}=?" for k in kw)
    with db() as c:
        c.execute(f"UPDATE runs SET {sets} WHERE id=?", (*kw.values(), run_id))


def save_rows(run_id, records):
    with db() as c:
        c.executemany(
            "INSERT INTO rows (run_id, data) VALUES (?, ?)",
            [(run_id, json.dumps(r)) for r in records],
        )


def get_runs(limit=20):
    with db() as c:
        return c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def get_run(run_id):
    with db() as c:
        return c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()


def get_rows(run_id):
    with db() as c:
        res = c.execute("SELECT data FROM rows WHERE run_id=?", (run_id,)).fetchall()
    return [json.loads(r["data"]) for r in res]


# ---------- AI helpers ----------
def nvidia_json(client, prompt):
    last = None
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            last = e
            time.sleep(8)
    raise last


def make_plan(client, user_prompt):
    prompt = f"""You design web data-collection workflows.
User request: "{user_prompt}"

Return JSON with exactly:
{{"summary": "one line describing the dataset",
  "queries": ["2-3 distinct web search queries"],
  "fields": ["3-6 snake_case column names to extract, e.g. title, company, location, link"]}}"""
    plan = nvidia_json(client, prompt)
    plan["queries"] = plan.get("queries", [])[:3]
    plan["fields"] = plan.get("fields", [])[:6]
    return plan


def search_web(queries, per_query=6, max_pages=8):
    seen, results = set(), []
    with DDGS() as d:
        for q in queries:
            try:
                for r in d.text(q, max_results=per_query):
                    url = r.get("href")
                    if url and url not in seen:
                        seen.add(url)
                        results.append(
                            {"url": url, "title": r.get("title", ""), "snippet": r.get("body", "")}
                        )
            except Exception:
                continue
    return results[:max_pages]


def fetch_text(item):
    try:
        resp = requests.get(item["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header"]):
            t.decompose()
        text = soup.get_text(" ", strip=True)[:3000]
        if len(text) > 200:
            return text
    except Exception:
        pass
    return f"{item['title']}. {item['snippet']}"


def extract_records(client, user_prompt, fields, pages):
    records = []
    for i in range(0, len(pages), 3):
        batch = pages[i : i + 3]
        prompt = f"""Extract structured records for this request: "{user_prompt}"
Fields: {fields}
Sources (JSON): {json.dumps(batch)}

Rules:
- Return a JSON array of objects. Each object has the fields above plus "source_url"
  (must be exactly one of the source urls given).
- Only use facts present in the source text. Never invent data. Use null if unknown.
- One object per distinct item found (e.g. per job, lead, company)."""
        try:
            out = nvidia_json(client, prompt)
        except Exception:
            continue
        if isinstance(out, dict):
            out = next((v for v in out.values() if isinstance(v, list)), [])
        records.extend([r for r in out if isinstance(r, dict)])
    return records


def clean_records(records, fields):
    cleaned, seen = [], set()
    for r in records:
        row = {f: (str(r.get(f)).strip() if r.get(f) not in (None, "", "null") else None) for f in fields}
        row["source_url"] = r.get("source_url")
        filled = sum(1 for f in fields if row[f])
        if filled < max(2, len(fields) // 2) or not row["source_url"]:
            continue
        key = tuple((row[f] or "").lower() for f in fields[:2])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(row)
    return cleaned


def run_pipeline(client, user_prompt):
    run_id = create_run(user_prompt)
    try:
        with st.status("Running workflow...", expanded=True) as s:
            st.write("🧠 Planning the workflow...")
            plan = make_plan(client, user_prompt)
            update_run(run_id, plan=json.dumps(plan))
            st.write(f"Queries: {plan['queries']}")
            st.write(f"Fields: {plan['fields']}")

            st.write("🌐 Searching the web...")
            results = search_web(plan["queries"])
            if not results:
                raise RuntimeError("No search results found. Try a different prompt.")

            st.write(f"📄 Reading {len(results)} pages...")
            pages = [{"url": r["url"], "text": fetch_text(r)} for r in results]

            st.write("🔍 Extracting structured data...")
            raw = extract_records(client, user_prompt, plan["fields"], pages)

            st.write("🧹 Cleaning and deduplicating...")
            final = clean_records(raw, plan["fields"])
            save_rows(run_id, final)
            update_run(run_id, status="done")
            s.update(label=f"Done: {len(final)} records collected", state="complete")
    except Exception as e:
        update_run(run_id, status="failed", error=str(e))
        st.error(f"Run failed: {e}")
    return run_id


# ---------- UI ----------
init_db()
if "run_id" not in st.session_state:
    st.session_state.run_id = None

st.title("🔎 AskData: AI-Powered Data Intelligence Platform")
st.caption("Describe the data you need in plain English. The AI plans, collects, cleans and presents it.")

with st.sidebar:
    st.header("Settings")
    key = st.text_input("NVIDIA API key", type="password", value=os.getenv("NVIDIA_API_KEY", ""))
    st.header("Task history")
    icons = {"done": "✅", "running": "⏳", "failed": "❌"}
    for r in get_runs():
        label = f"{icons.get(r['status'], '•')} #{r['id']} {r['prompt'][:28]}"
        if st.button(label, key=f"run{r['id']}"):
            st.session_state.run_id = r["id"]

prompt = st.text_area(
    "What data do you need?",
    placeholder="e.g. AI startup internship openings in Bangalore with company and apply link",
    height=90,
)

if st.button("Collect data", type="primary"):
    if not key.strip():
        st.warning("Add your NVIDIA API key in the sidebar or in your .env file.")
    elif not prompt.strip():
        st.warning("Type what you need first.")
    else:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=key.strip(),
        )
        st.session_state.run_id = run_pipeline(client, prompt.strip())

run_id = st.session_state.run_id
if run_id:
    run = get_run(run_id)
    st.subheader(f"Run #{run['id']} · {run['status']} · {run['created_at']}")
    st.write(f"**Request:** {run['prompt']}")
    if run["plan"]:
        with st.expander("Workflow plan"):
            st.json(json.loads(run["plan"]))
    if run["error"]:
        st.error(run["error"])

    rows = get_rows(run_id)
    if rows:
        df = pd.DataFrame(rows)
        df["source_domain"] = df["source_url"].apply(lambda u: urlparse(u).netloc if u else "")

        c1, c2 = st.columns([2, 1])
        q = c1.text_input("Search results")
        domains = c2.multiselect("Filter by source", sorted(df["source_domain"].unique()))
        if q:
            mask = df.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False)).any(axis=1)
            df = df[mask]
        if domains:
            df = df[df["source_domain"].isin(domains)]

        st.dataframe(
            df,
            use_container_width=True,
            column_config={"source_url": st.column_config.LinkColumn("source_url")},
        )
        st.download_button("⬇️ Export CSV", df.to_csv(index=False), f"askdata_run_{run_id}.csv", "text/csv")

        with st.expander("Sources used"):
            for u in sorted(df["source_url"].dropna().unique()):
                st.markdown(f"- {u}")
    elif run["status"] == "done":
        st.info("No usable records found. Try a more specific prompt.")