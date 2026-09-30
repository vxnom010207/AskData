# AskData: AI-Powered Data Intelligence Platform

Describe the data you need in plain English. AskData plans a collection workflow with Llama 3, searches the web, reads pages, extracts structured records with source URLs, cleans and deduplicates them, and shows them in an interactive dashboard.

## Features
- Natural-language prompt to workflow plan (queries + fields)
- Web collection from public sources (DuckDuckGo search + public pages)
- LLM extraction, validation and deduplication
- Every record is traceable to its source URL
- Search, filter and CSV export
- Task history with status (running / done / failed), stored in SQLite

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```
Copy `.env.example` to `.env`, put your free Gemini API key (aistudio.google.com/apikey) in it,.

## Tech stack
Python, Streamlit, Google Gemini API, ddgs, BeautifulSoup, SQLite, pandas