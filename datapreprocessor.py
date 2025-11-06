"""
Data Processing Module for NLP-based Diet and Nutrition Analysis
Handles data loading, preprocessing, and feature engineering
"""

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
from typing import Dict, List, Tuple, Optional
import logging
import os

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class NutritionDataProcessor:
    """
    Comprehensive data processor for nutrition and diet analysis
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the data processor
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP processing
        """
        self.use_spacy = use_spacy
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Load spaCy model if requested
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logging.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        
        # Nutrition-specific keywords and patterns
        self.nutrition_keywords = {
            'macronutrients': ['protein', 'carbohydrate', 'carbs', 'fat', 'fiber', 'sugar'],
            'vitamins': ['vitamin a', 'vitamin c', 'vitamin d', 'vitamin e', 'vitamin k', 'thiamine', 'riboflavin', 'niacin', 'folate'],
            'minerals': ['calcium', 'iron', 'magnesium', 'phosphorus', 'potassium', 'sodium', 'zinc'],
            'diet_types': ['vegan', 'vegetarian', 'keto', 'paleo', 'mediterranean', 'gluten-free', 'dairy-free'],
            'allergies': ['nuts', 'dairy', 'eggs', 'soy', 'wheat', 'shellfish', 'fish'],
            'health_goals': ['weight loss', 'weight gain', 'muscle gain', 'heart health', 'diabetes', 'energy']
        }
        
    def load_nutrition_data(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Load nutrition data from multiple CSV files
        
        Args:
            file_paths: List of paths to CSV files
            
        Returns:
            Combined DataFrame with nutrition data
        """
        dataframes = []
        
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df['source_file'] = os.path.basename(file_path)
                    dataframes.append(df)
                    logging.info(f"Loaded {len(df)} records from {file_path}")
                else:
                    logging.warning(f"File not found: {file_path}")
            except Exception as e:
                logging.error(f"Error loading {file_path}: {str(e)}")
        
        if not dataframes:
            # Create sample data if no files are found
            return self._create_sample_nutrition_data()
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        logging.info(f"Combined dataset: {len(combined_df)} total records")
        
        return combined_df
    
    def _create_sample_nutrition_data(self) -> pd.DataFrame:
        """
        Create sample nutrition data for demonstration
        """
        sample_data = {
            'food_name': [
                'Grilled Chicken Breast', 'Brown Rice', 'Broccoli', 'Salmon Fillet',
                'Greek Yogurt', 'Almonds', 'Sweet Potato', 'Spinach',
                'Quinoa', 'Avocado', 'Eggs', 'Oats'
            ],
            'description': [
                'Lean protein source, grilled without oil',
                'Whole grain rice, high in fiber',
                'Cruciferous vegetable, rich in vitamins',
                'Fatty fish, omega-3 rich',
                'High protein dairy, probiotic',
                'Tree nuts, healthy fats',
                'Root vegetable, beta carotene',
                'Leafy green, iron rich',
                'Ancient grain, complete protein',
                'Fruit, monounsaturated fats',
                'Complete protein, versatile',
                'Whole grain, soluble fiber'
            ],
            'calories_per_100g': [165, 123, 34, 208, 97, 579, 86, 23, 368, 160, 155, 389],
            'protein_g': [31, 2.7, 2.8, 25, 10, 21, 2, 2.9, 14, 2, 13, 17],
            'carbs_g': [0, 25, 7, 0, 3.6, 22, 20, 3.6, 64, 9, 1.1, 66],
            'fat_g': [3.6, 0.9, 0.4, 13, 0.4, 50, 0.1, 0.4, 6, 15, 11, 7],
            'fiber_g': [0, 1.8, 2.6, 0, 0, 12, 3, 2.2, 7, 7, 0, 11],
            'category': ['meat', 'grain', 'vegetable', 'fish', 'dairy', 'nuts', 'vegetable', 'vegetable', 'grain', 'fruit', 'protein', 'grain']
        }
        
        return pd.DataFrame(sample_data)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_nutrition_keywords(self, text: str) -> Dict[str, List[str]]:
        """
        Extract nutrition-related keywords from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with categorized keywords found
        """
        cleaned_text = self.clean_text(text)
        found_keywords = {category: [] for category in self.nutrition_keywords}
        
        for category, keywords in self.nutrition_keywords.items():
            for keyword in keywords:
                if keyword in cleaned_text:
                    found_keywords[category].append(keyword)
        
        return found_keywords
    
    def advanced_nlp_processing(self, text: str) -> Dict[str, any]:
        """
        Perform advanced NLP processing using spaCy
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with NLP features
        """
        if not self.use_spacy:
            return self._basic_nlp_processing(text)
        
        doc = self.nlp(text)
        
        features = {
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
            'sentiment': TextBlob(text).sentiment.polarity,
            'subjectivity': TextBlob(text).sentiment.subjectivity
        }
        
        return features
    
    def _basic_nlp_processing(self, text: str) -> Dict[str, any]:
        """
        Basic NLP processing without spaCy
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with basic NLP features
        """
        blob = TextBlob(text)
        tokens = word_tokenize(self.clean_text(text))
        
        features = {
            'pos_tags': blob.tags,
            'sentiment': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'word_count': len(tokens),
            'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0
        }
        
        return features
    
    def create_nutrition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive nutrition features from the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_processed = df.copy()
        
        # Basic nutrition calculations
        if all(col in df.columns for col in ['protein_g', 'carbs_g', 'fat_g', 'calories_per_100g']):
            df_processed['protein_calories'] = df_processed['protein_g'] * 4
            df_processed['carbs_calories'] = df_processed['carbs_g'] * 4
            df_processed['fat_calories'] = df_processed['fat_g'] * 9
            
            df_processed['protein_ratio'] = df_processed['protein_calories'] / df_processed['calories_per_100g']
            df_processed['carbs_ratio'] = df_processed['carbs_calories'] / df_processed['calories_per_100g']
            df_processed['fat_ratio'] = df_processed['fat_calories'] / df_processed['calories_per_100g']
            
            # Macronutrient balance score
            df_processed['macro_balance_score'] = 1 - np.abs(df_processed[['protein_ratio', 'carbs_ratio', 'fat_ratio']].var(axis=1))
        
        # Text processing for food names and descriptions
        if 'food_name' in df.columns:
            df_processed['food_name_clean'] = df_processed['food_name'].apply(self.clean_text)
            df_processed['food_name_length'] = df_processed['food_name'].str.len()
            df_processed['food_name_words'] = df_processed['food_name'].str.split().str.len()
        
        if 'description' in df.columns:
            df_processed['description_clean'] = df_processed['description'].apply(self.clean_text)
            
            # Extract nutrition keywords
            keyword_features = df_processed['description'].apply(self.extract_nutrition_keywords)
            for category in self.nutrition_keywords:
                df_processed[f'{category}_count'] = keyword_features.apply(lambda x: len(x.get(category, [])))
            
            # NLP features
            nlp_features = df_processed['description'].apply(self.advanced_nlp_processing)
            df_processed['sentiment_score'] = nlp_features.apply(lambda x: x.get('sentiment', 0))
            df_processed['subjectivity_score'] = nlp_features.apply(lambda x: x.get('subjectivity', 0))
        
        # Categorical encoding
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['food_name', 'description', 'food_name_clean', 'description_clean']:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_processed[col].fillna('unknown'))
                else:
                    df_processed[f'{col}_encoded'] = self.label_encoders[col].transform(df_processed[col].fillna('unknown'))
        
        return df_processed
    
    def create_tfidf_features(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Create TF-IDF features from text data
        
        Args:
            texts: List of text documents
            fit: Whether to fit the vectorizer
            
        Returns:
            TF-IDF feature matrix
        """
        clean_texts = [self.clean_text(str(text)) for text in texts]
        
        if fit:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(clean_texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(clean_texts)
        
        return tfidf_matrix.toarray()
    
    def calculate_nutritional_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various nutritional quality scores
        
        Args:
            df: DataFrame with nutrition data
            
        Returns:
            DataFrame with added scoring columns
        """
        df_scored = df.copy()
        
        # Protein quality score (higher protein content = higher score)
        if 'protein_g' in df.columns:
            df_scored['protein_quality_score'] = np.clip(df_scored['protein_g'] / 30, 0, 1)
        
        # Fiber quality score
        if 'fiber_g' in df.columns:
            df_scored['fiber_quality_score'] = np.clip(df_scored['fiber_g'] / 10, 0, 1)
        
        # Overall nutrition score (weighted combination)
        score_columns = [col for col in df_scored.columns if col.endswith('_quality_score')]
        if score_columns:
            df_scored['overall_nutrition_score'] = df_scored[score_columns].mean(axis=1)
        
        # Calorie density classification
        if 'calories_per_100g' in df.columns:
            df_scored['calorie_density'] = pd.cut(
                df_scored['calories_per_100g'],
                bins=[0, 100, 200, 400, float('inf')],
                labels=['low', 'moderate', 'high', 'very_high']
            )
        
        return df_scored
    
    def prepare_ml_features(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features for machine learning models
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (if any)
            
        Returns:
            Feature matrix and target array (if provided)
        """
        # Select numeric columns for ML
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column from features
        if target_column and target_column in numeric_columns:
            numeric_columns.remove(target_column)
        
        X = df[numeric_columns].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        y = None
        if target_column and target_column in df.columns:
            y = df[target_column].values
        
        return X_scaled, y
    
    def generate_data_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate comprehensive data summary
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        }
        
        # Category distribution
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].nunique() < 50:  # Only for columns with reasonable number of categories
                summary[f'{col}_distribution'] = df[col].value_counts().to_dict()
        
        return summary