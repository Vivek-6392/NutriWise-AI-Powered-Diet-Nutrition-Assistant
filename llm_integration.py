import json
import pandas as pd
import numpy as np
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Optional, Any
import logging
import os
from datetime import datetime
from dataclasses import dataclass
import requests

# New OpenAI import
from openai import OpenAI

# Optional embedding support
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass


class NutritionDataProcessor:
    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logging.warning("spaCy model missing: run → python -m spacy download en_core_web_sm")
                self.use_spacy = False

        if SentenceTransformer:
            try:
                self._embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            except:
                self._embedder = None
        else:
            self._embedder = None

        self._food_embeddings = None

    def clean_text(self, text: str) -> str:
        if pd.isna(text) or not isinstance(text, str): return ""
        return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower()).strip()

    def load_nutrition_data(self, file_paths: List[str]) -> pd.DataFrame:
        dfs = []
        for path in file_paths:
            try:
                df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
                df["source_file"] = os.path.basename(path)
                dfs.append(df)
            except Exception as e:
                logging.error(f"Load error {path}: {e}")

        if not dfs: return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        if "food_name" in combined.columns:
            combined["food_name_clean"] = combined["food_name"].astype(str).apply(self.clean_text)

        return combined

    def advanced_food_item_retrieve(self, query: str, df: pd.DataFrame, topn: int = 1) -> pd.DataFrame:
        if 'food_name_clean' not in df.columns:
            df['food_name_clean'] = df['food_name'].astype(str).apply(self.clean_text)

        cleaned = self.clean_text(query)
        matches = df[df.food_name_clean.str.contains(cleaned)]
        return matches.head(topn) if not matches.empty else pd.DataFrame()


@dataclass
class NutritionQuery:
    user_id: str
    query: str
    context: Dict[str, Any]
    timestamp: datetime
    query_type: str


class NutritionLLMIntegration:
    def __init__(
        self,
        nutrition_data: pd.DataFrame,
        processor,
        api_key: Optional[str] = None,
        use_lmstudio: bool = False,
        lmstudio_url: str = "http://localhost:1234/v1/chat/completions",
        lmstudio_model: str = "openchat-3.6-8b-20240522"
    ):
        self.nutrition_data = nutrition_data
        self.processor = processor
        self.use_lmstudio = use_lmstudio
        self.lmstudio_url = lmstudio_url
        self.lmstudio_model = lmstudio_model

        self.client = OpenAI(api_key=api_key) if api_key and not use_lmstudio else None

        self.conversation_history = {}

        self.system_prompts = {
            'nutritionist': "You are a professional nutritionist.",
            'meal_planner': "You are an expert meal planner.",
            'educator': "You teach nutrition in an easy way.",
            'analysis': "You analyze food nutrition data."
        }

    def _build_context(self, query: NutritionQuery) -> str:
        role = self.system_prompts.get(query.query_type, self.system_prompts['nutritionist'])
        food_table = self._context_from_nutrition_table(query.query)

        return f"""
{role}

User Context: {query.context}

Nutrition Table:
{food_table}
"""

    def _context_from_nutrition_table(self, query: str, max_foods=5):
        matches = self.processor.advanced_food_item_retrieve(query, self.nutrition_data, max_foods)
        return matches.to_markdown(index=False) if not matches.empty else "No data found."

    def _conversation(self, user_id):
        return self.conversation_history.setdefault(user_id, [])

    def handle_query(self, query: NutritionQuery) -> str:
        try:
            context = self._build_context(query)
            messages = [{"role": "system", "content": context}, *self._conversation(query.user_id),
                        {"role": "user", "content": query.query}]

            # LM Studio local call
            if self.use_lmstudio:
                resp = requests.post(self.lmstudio_url, json={
                    "model": self.lmstudio_model,
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.7
                })
                resp.raise_for_status()
                reply = resp.json()["choices"][0]["message"]["content"]

            # OpenAI API call (updated)
            elif self.client:
                response = self.client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                reply = response.choices[0].message.content

            else:
                reply = "No LLM available. Enable LM Studio or OpenAI."

            self._conversation(query.user_id).append({"role": "assistant", "content": reply})
            return reply

        except Exception as e:
            logging.error(f"LLM error: {e}")
            return "Sorry, something went wrong while generating a response."


# ---------------- MAIN ----------------
if __name__ == "__main__":
    file = "Anuvaad_INDB_2024.11.xlsx"

    processor = NutritionDataProcessor()
    data = processor.load_nutrition_data([file])

    llm = NutritionLLMIntegration(
        data,
        processor,
        api_key=None,  # Add your OpenAI key here if needed
        use_lmstudio=True
    )

    q = NutritionQuery(
        user_id="user123",
        query="Show carbs and fat for lemonade and roti",
        context={},
        timestamp=datetime.now(),
        query_type="analysis"
    )

    print(llm.handle_query(q))
