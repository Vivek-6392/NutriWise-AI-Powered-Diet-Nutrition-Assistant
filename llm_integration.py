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
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
from datetime import datetime
from dataclasses import dataclass
import requests  # for LM Studio local API
import openai



# Optional: sentence-transformers for semantic search
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
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logging.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        if SentenceTransformer:
            try:
                self._embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            except Exception:
                self._embedder = None
        else:
            self._embedder = None
        self._food_embeddings = None

    def clean_text(self, text: str) -> str:
        if pd.isna(text) or not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def load_nutrition_data(self, file_paths: List[str]) -> pd.DataFrame:
        dataframes = []
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    if file_path.endswith('.xlsx'):
                        df = pd.read_excel(file_path)
                    else:
                        df = pd.read_csv(file_path)
                    df['source_file'] = os.path.basename(file_path)
                    dataframes.append(df)
                    logging.info(f"Loaded {len(df)} records from {file_path}")
                else:
                    logging.warning(f"File not found: {file_path}")
            except Exception as e:
                logging.error(f"Error loading {file_path}: {str(e)}")
        if not dataframes:
            return pd.DataFrame()
        combined_df = pd.concat(dataframes, ignore_index=True)
        logging.info(f"Combined dataset: {len(combined_df)} total records")
        if 'food_name' in combined_df.columns:
            combined_df['food_name_clean'] = combined_df['food_name'].astype(str).apply(self.clean_text)
        return combined_df

    def advanced_food_item_retrieve(self, query: str, df: pd.DataFrame, topn: int = 1) -> pd.DataFrame:
        if 'food_name_clean' not in df.columns:
            df['food_name_clean'] = df['food_name'].astype(str).apply(self.clean_text)
        food_names = df['food_name_clean'].tolist()
        query_clean = self.clean_text(query)
        matches = [name for name in food_names if name in query_clean]
        if matches:
            return df[df['food_name_clean'].isin(matches)].head(topn)
        if self.use_spacy and hasattr(self, 'nlp'):
            doc = self.nlp(query)
            noun_chunks = [self.clean_text(chunk.text) for chunk in doc.noun_chunks]
            for chunk in noun_chunks:
                matches = [name for name in food_names if chunk in name]
                if matches:
                    return df[df['food_name_clean'].isin(matches)].head(topn)

        if self._embedder:
            try:
                if self._food_embeddings is None:
                    self._food_embeddings = self._embedder.encode(food_names, normalize_embeddings=True)
                query_vec = self._embedder.encode([query_clean], normalize_embeddings=True)
                sims = np.dot(self._food_embeddings, query_vec.T).flatten()
                idx = np.argmax(sims)
                if sims[idx] > 0.65:
                    return df.iloc[[idx]]
            except Exception:
                pass
        return pd.DataFrame()

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
        self.api_key = api_key
        self.use_lmstudio = use_lmstudio
        self.lmstudio_url = lmstudio_url
        self.lmstudio_model = lmstudio_model
        if api_key and not use_lmstudio:
            openai.api_key = api_key
        self.conversation_history = {}
        self.system_prompts = {
            'nutritionist': "You are a professional nutritionist and dietitian...",
            'meal_planner': "You are an expert meal planning specialist...",
            'educator': "You are a nutrition educator who explains complex concepts...",
            'analyzer': "You are a food and nutrition analyst who can break down meals..."
        }
        self.nutrition_knowledge = {
            'macronutrients': {
                'protein': {
                    'functions': 'Building and repairing tissues, enzyme production, immune function',
                    'sources': 'Meat, fish, eggs, dairy, legumes, nuts, seeds',
                    'daily_needs': '0.8-1.6g per kg body weight'
                },
                'carbohydrates': {
                    'functions': 'Primary energy source, brain fuel, muscle fuel',
                    'sources': 'Grains, fruits, vegetables, legumes',
                    'daily_needs': '45-65% of total calories'
                },
                'fats': {
                    'functions': 'Energy storage, vitamin absorption, hormone production',
                    'sources': 'Oils, nuts, seeds, fatty fish, avocado',
                    'daily_needs': '20-35% of total calories'
                }
            },
            'micronutrients': {
                'vitamins': {
                    'A': {'functions': 'Vision, immune function', 'sources': 'Orange vegetables, leafy greens'},
                    'C': {'functions': 'Antioxidant, collagen synthesis', 'sources': 'Citrus fruits, berries'},
                    'D': {'functions': 'Bone health, immune function', 'sources': 'Sunlight, fatty fish, fortified foods'},
                    'E': {'functions': 'Antioxidant, cell protection', 'sources': 'Nuts, seeds, vegetable oils'},
                    'K': {'functions': 'Blood clotting, bone health', 'sources': 'Leafy greens, broccoli'}
                },
                'minerals': {
                    'calcium': {'functions': 'Bone health, muscle function', 'sources': 'Dairy, leafy greens, fortified foods'},
                    'iron': {'functions': 'Oxygen transport, energy metabolism', 'sources': 'Red meat, beans, spinach'},
                    'magnesium': {'functions': 'Enzyme function, muscle relaxation', 'sources': 'Nuts, seeds, whole grains'},
                    'zinc': {'functions': 'Immune function, wound healing', 'sources': 'Meat, shellfish, legumes'},
                    'potassium': {'functions': 'Fluid balance, nerve signaling', 'sources': 'Bananas, potatoes, beans'}
                }
            }
        }

    def _build_context(self, query: NutritionQuery) -> str:
        """Build context string for the LLM based on query, including robust table lookup."""
        role = query.query_type
        system_prompt = self.system_prompts.get(role, self.system_prompts['nutritionist'])
        table_context = self._context_from_nutrition_table(query.query, max_foods=5)
        knowledge_snippets = json.dumps(self.nutrition_knowledge, indent=2)
        context_parts = [
            system_prompt,
            f"User Context: {query.context}",
            f"Nutrition Table (matching items):\n{table_context}",
            f"Nutrition KB Summary:\n{knowledge_snippets}"
        ]
        return "\n\n".join(context_parts)

    def _context_from_nutrition_table(self, user_query: str, max_foods=5) -> str:
        df = self.nutrition_data
        processor = self.processor
        matches = processor.advanced_food_item_retrieve(user_query, df, topn=max_foods)
        if isinstance(matches, pd.DataFrame) and not matches.empty:
            food_match_df = matches
        else:
            food_match_df = df.head(3)
        possible_cols = [col for col in df.columns if col not in ['food_name', 'food_name_clean']]
        requested_cols = []
        q_lower = user_query.lower()
        for c in possible_cols:
            simple_c = c.lower().replace("_", " ")
            if simple_c in q_lower or any(n in q_lower for n in simple_c.split()):
                requested_cols.append(c)
        if not requested_cols:
            requested_cols = [c for c in possible_cols if "calories" in c or "protein" in c or "carb" in c or "fat" in c or "fiber" in c]
            if not requested_cols: requested_cols = possible_cols[:5]
        context_df = food_match_df[["food_name"] + requested_cols] if requested_cols else food_match_df
        if not context_df.empty:
            out = context_df.to_markdown(index=False)
        else:
            out = "No matching foods."
        return out

    def _get_conversation(self, user_id: str) -> List[Dict[str, str]]:
        return self.conversation_history.get(user_id, [])

    def _update_conversation(self, user_id: str, role: str, content: str):
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append({"role": role, "content": content})

    def handle_query(self, query: NutritionQuery) -> str:
        try:
            context = self._build_context(query)
            messages = [{"role": "system", "content": context}]
            history = self._get_conversation(query.user_id)
            messages.extend(history)
            messages.append({"role": "user", "content": query.query})
            if self.use_lmstudio:
                payload = {
                    "model": self.lmstudio_model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 500,
                }
                resp = requests.post(self.lmstudio_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
            elif self.api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                answer = response['choices'][0]['message']['content']
            else:
                return self._rule_based_response(query)
            self._update_conversation(query.user_id, "assistant", answer)
            return answer
        except Exception as e:
            logging.error(f"Error in handle_query: {e}")
            return "Sorry, I encountered an error while processing your request."

    def _rule_based_response(self, query: NutritionQuery) -> str:
        if query.query_type == "analysis":
            return f"Basic analysis: Based on your input {query.context}, aim for a balanced diet of protein, carbs, and fats."
        elif query.query_type == "recommendation":
            return f"Basic recommendation: Include fruits, vegetables, lean protein, and whole grains."
        elif query.query_type == "education":
            return f"Basic education: Nutrition involves macronutrients (protein, carbs, fats) and micronutrients (vitamins, minerals)."
        elif query.query_type == "meal_planning":
            return f"Basic meal plan: Breakfast - oats and fruit, Lunch - grilled chicken salad, Dinner - lentil soup with whole wheat bread."
        else:
            return "I'm not sure, but maintaining a balanced diet with variety is always beneficial."

# -------- MAIN USAGE EXAMPLE ---------
if __name__ == "__main__":
    # Set your file path!
    excel_path = "Anuvaad_INDB_2024.11.xlsx"

    processor = NutritionDataProcessor(use_spacy=True)
    nutrition_data = processor.load_nutrition_data([excel_path])

    # Initialize with processor argument included:
    llm_engine = NutritionLLMIntegration(
        nutrition_data,
        processor,  # <-- REQUIRED argument!
        api_key=None,        # Set if using OpenAI
        use_lmstudio=False,  # Set True if using LM Studio
        lmstudio_url="http://localhost:1234/v1/chat/completions",
        lmstudio_model="openchat-3.6-8b-20240522"
    )

    # Example test query
    query = NutritionQuery(
        user_id="user1",
        query="Show carbs and fat for lemonade and roti.",
        context={},
        timestamp=datetime.now(),
        query_type="analysis"
    )
    answer = llm_engine.handle_query(query)
    print(answer)
