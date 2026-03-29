import json
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Optional, Any
import logging
import os
from datetime import datetime
from dataclasses import dataclass
import requests

from openai import OpenAI
from dotenv import load_dotenv

# =====================
# ENV + GROQ CONFIG
# =====================
import streamlit as st

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    logging.warning("⚠️ GROQ_API_KEY not found. Groq fallback will fail.")

# =====================
# NLTK
# =====================
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
except:
    pass


# =====================
# DATA PROCESSOR
# =====================
class NutritionDataProcessor:
    def __init__(self, use_spacy=True):
        self.use_spacy = use_spacy
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.use_spacy = False

    def clean_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).strip()

    def load_nutrition_data(self, file_paths):
        dfs = []
        for p in file_paths:
            try:
                df = pd.read_excel(p) if p.endswith(".xlsx") else pd.read_csv(p)
                dfs.append(df)
            except Exception as e:
                logging.error(e)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def advanced_food_item_retrieve(self, query, df, topn=3):
        if "food_name_clean" not in df.columns:
            df["food_name_clean"] = df["food_name"].astype(str).apply(self.clean_text)
        q = self.clean_text(query)
        return df[df.food_name_clean.str.contains(q)].head(topn)


# =====================
# QUERY OBJECT
# =====================
@dataclass
class NutritionQuery:
    user_id: str
    query: str
    context: Dict[str, Any]
    timestamp: datetime
    query_type: str


# =====================
# LLM INTEGRATION
# =====================
class NutritionLLMIntegration:
    def __init__(
        self,
        nutrition_data,
        processor,
        use_lmstudio=True,
        lmstudio_url="http://localhost:1234/v1/chat/completions",
        lmstudio_model="openchat-3.6-8b-20240522",
    ):
        self.nutrition_data = nutrition_data
        self.processor = processor
        self.use_lmstudio = use_lmstudio
        self.lmstudio_url = lmstudio_url
        self.lmstudio_model = lmstudio_model

        self.groq_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
        )

        self.conversation_history = {}

    # ---------------------
    # REAL LM STUDIO CHECK
    # ---------------------
    def _lmstudio_available(self):
        try:
            r = requests.post(
                self.lmstudio_url,
                json={
                    "model": self.lmstudio_model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                timeout=5,
            )
            return r.status_code == 200
        except:
            return False

    def _build_context(self, query):
        table = self.processor.advanced_food_item_retrieve(
            query.query, self.nutrition_data
        )
        table_md = table.to_markdown(index=False) if not table.empty else "No data found."
        return f"""
You are a professional nutrition expert.

User Context:
{query.context}

Nutrition Data:
{table_md}
"""

    def _conversation(self, user_id):
        return self.conversation_history.setdefault(user_id, [])

    # ---------------------
    # MAIN HANDLER
    # ---------------------
    def handle_query(self, query: NutritionQuery, stream: bool = False):
        messages = [
            {"role": "system", "content": self._build_context(query)},
            *self._conversation(query.user_id),
            {"role": "user", "content": query.query},
        ]

        # -------------------------
        # STREAMING MODE (LM STUDIO)
        # -------------------------
        if stream and self.use_lmstudio and self._lmstudio_available():
            return self._stream_lmstudio(messages, query.user_id)

        # -------------------------
        # NORMAL MODE (EXISTING)
        # -------------------------
        if self.use_lmstudio and self._lmstudio_available():
            try:
                resp = requests.post(
                    self.lmstudio_url,
                    json={
                        "model": self.lmstudio_model,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 500,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                reply = resp.json()["choices"][0]["message"]["content"]
                source = "LM Studio (Local)"
            except Exception:
                reply, source = self._groq_fallback(messages)
        else:
            reply, source = self._groq_fallback(messages)

        self._conversation(query.user_id).append(
            {"role": "assistant", "content": reply}
        )

        return f"**Source:** {source}\n\n{reply}"


    def _stream_lmstudio(self, messages, user_id):
        """
        Generator for Streamlit token streaming
        """
        payload = {
            "model": self.lmstudio_model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": True,
        }

        with requests.post(
            self.lmstudio_url,
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()

            full_reply = ""
            for line in resp.iter_lines():
                if not line:
                    continue

                if line.startswith(b"data: "):
                    data = line.replace(b"data: ", b"").decode("utf-8")

                    if data == "[DONE]":
                        break

                    try:
                        token = json.loads(data)["choices"][0]["delta"].get("content", "")
                        full_reply += token
                        yield token
                    except Exception:
                        continue

            # Save conversation
            self._conversation(user_id).append(
                {"role": "assistant", "content": full_reply}
            )



        # ---------------------
        # GROQ FALLBACK
    # ---------------------
    def _groq_fallback(self, messages):
        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content, "Groq API (Cloud)"
