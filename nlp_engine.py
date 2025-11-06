"""
NLP Engine for Diet and Nutrition Analysis
Handles natural language queries and provides intelligent nutrition recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import logging

class NutritionNLPEngine:
    """
    Advanced NLP engine for processing nutrition queries and generating recommendations
    """
    
    def __init__(self, nutrition_data: pd.DataFrame):
        """
        Initialize the NLP engine with nutrition data
        
        Args:
            nutrition_data: DataFrame containing nutrition information
        """
        self.nutrition_data = nutrition_data
        self.sentence_model = None
        self.food_embeddings = None
        self.food_names = []
        
        # Initialize sentence transformer model
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._create_food_embeddings()
        except Exception as e:
            logging.warning(f"Could not load sentence transformer: {e}")
        
        # Nutrition knowledge base
        self.nutrition_knowledge = {
            'weight_loss': {
                'keywords': ['lose weight', 'weight loss', 'slim down', 'reduce weight', 'burn fat'],
                'recommendations': 'low calorie, high protein, high fiber foods',
                'avoid': 'high calorie, high sugar, processed foods',
                'calorie_range': (1200, 1800),
                'macro_ratio': {'protein': 0.3, 'carbs': 0.4, 'fat': 0.3}
            },
            'weight_gain': {
                'keywords': ['gain weight', 'weight gain', 'bulk up', 'increase weight'],
                'recommendations': 'calorie-dense, protein-rich foods',
                'avoid': 'very low calorie foods',
                'calorie_range': (2500, 3500),
                'macro_ratio': {'protein': 0.25, 'carbs': 0.45, 'fat': 0.3}
            },
            'muscle_gain': {
                'keywords': ['build muscle', 'muscle gain', 'strength', 'bodybuilding'],
                'recommendations': 'high protein, complex carbs, healthy fats',
                'avoid': 'excessive sugar, trans fats',
                'calorie_range': (2200, 3000),
                'macro_ratio': {'protein': 0.35, 'carbs': 0.4, 'fat': 0.25}
            },
            'heart_health': {
                'keywords': ['heart health', 'cardiovascular', 'cholesterol', 'blood pressure'],
                'recommendations': 'omega-3 rich foods, fiber, antioxidants',
                'avoid': 'saturated fats, sodium, processed foods',
                'calorie_range': (1800, 2400),
                'macro_ratio': {'protein': 0.2, 'carbs': 0.5, 'fat': 0.3}
            },
            'diabetes': {
                'keywords': ['diabetes', 'blood sugar', 'glucose', 'insulin'],
                'recommendations': 'low glycemic index, high fiber, lean protein',
                'avoid': 'simple sugars, refined carbs',
                'calorie_range': (1600, 2200),
                'macro_ratio': {'protein': 0.25, 'carbs': 0.45, 'fat': 0.3}
            }
        }
        
        # Diet type patterns
        self.diet_patterns = {
            'vegan': ['vegan', 'plant-based', 'no animal products'],
            'vegetarian': ['vegetarian', 'no meat', 'plant-based'],
            'keto': ['keto', 'ketogenic', 'low carb', 'high fat'],
            'paleo': ['paleo', 'paleolithic', 'ancestral diet'],
            'mediterranean': ['mediterranean', 'olive oil', 'fish'],
            'gluten_free': ['gluten-free', 'celiac', 'no gluten'],
            'dairy_free': ['dairy-free', 'lactose-free', 'no dairy']
        }
        
        # Allergy patterns
        self.allergy_patterns = {
            'nuts': ['nut allergy', 'no nuts', 'tree nuts', 'peanuts'],
            'dairy': ['dairy allergy', 'lactose intolerant', 'no milk'],
            'eggs': ['egg allergy', 'no eggs'],
            'soy': ['soy allergy', 'no soy'],
            'wheat': ['wheat allergy', 'gluten sensitivity'],
            'shellfish': ['shellfish allergy', 'seafood allergy'],
            'fish': ['fish allergy', 'no fish']
        }
    
    def _create_food_embeddings(self):
        """Create embeddings for all food items in the database"""
        if self.sentence_model is None:
            return
        
        try:
            # Combine food name and description for better embeddings
            food_descriptions = []
            self.food_names = []
            
            for _, row in self.nutrition_data.iterrows():
                name = str(row.get('food_name', ''))
                desc = str(row.get('description', ''))
                category = str(row.get('category', ''))
                
                combined_text = f"{name} {desc} {category}".strip()
                food_descriptions.append(combined_text)
                self.food_names.append(name)
            
            # Create embeddings
            self.food_embeddings = self.sentence_model.encode(food_descriptions)
            logging.info(f"Created embeddings for {len(food_descriptions)} food items")
            
        except Exception as e:
            logging.error(f"Error creating food embeddings: {e}")
    
    def parse_user_query(self, query: str) -> Dict[str, Any]:
        """
        Parse and analyze user query to extract intent and entities
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary containing parsed information
        """
        query_lower = query.lower()
        
        parsed_info = {
            'original_query': query,
            'health_goals': [],
            'diet_type': None,
            'allergies': [],
            'food_preferences': [],
            'nutrients_mentioned': [],
            'sentiment': 0,
            'entities': []
        }
        
        # Sentiment analysis
        blob = TextBlob(query)
        parsed_info['sentiment'] = blob.sentiment.polarity
        
        # Extract health goals
        for goal, data in self.nutrition_knowledge.items():
            for keyword in data['keywords']:
                if keyword in query_lower:
                    parsed_info['health_goals'].append(goal)
                    break
        
        # Extract diet type
        for diet_type, patterns in self.diet_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    parsed_info['diet_type'] = diet_type
                    break
            if parsed_info['diet_type']:
                break
        
        # Extract allergies
        for allergy, patterns in self.allergy_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    parsed_info['allergies'].append(allergy)
                    break
        
        # Extract specific nutrients mentioned
        nutrients = ['protein', 'carbs', 'carbohydrates', 'fat', 'fiber', 'calcium', 'iron', 
                    'vitamin', 'mineral', 'omega', 'antioxidant']
        for nutrient in nutrients:
            if nutrient in query_lower:
                parsed_info['nutrients_mentioned'].append(nutrient)
        
        # Extract food preferences using simple pattern matching
        food_words = ['chicken', 'fish', 'beef', 'vegetables', 'fruits', 'rice', 'pasta', 
                     'beans', 'nuts', 'dairy', 'eggs', 'quinoa', 'oats']
        for food in food_words:
            if food in query_lower:
                parsed_info['food_preferences'].append(food)
        
        return parsed_info
    
    def find_similar_foods(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find foods similar to the query using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of similar foods to return
            
        Returns:
            List of similar foods with similarity scores
        """
        if self.sentence_model is None or self.food_embeddings is None:
            return self._fallback_food_search(query, top_k)
        
        try:
            # Encode the query
            query_embedding = self.sentence_model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.food_embeddings)[0]
            
            # Get top k similar foods
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            similar_foods = []
            for idx in top_indices:
                food_data = self.nutrition_data.iloc[idx].to_dict()
                food_data['similarity_score'] = float(similarities[idx])
                similar_foods.append(food_data)
            
            return similar_foods
            
        except Exception as e:
            logging.error(f"Error in semantic search: {e}")
            return self._fallback_food_search(query, top_k)
    
    def _fallback_food_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback search using simple text matching
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching foods
        """
        query_words = set(query.lower().split())
        
        scores = []
        for _, row in self.nutrition_data.iterrows():
            food_text = f"{row.get('food_name', '')} {row.get('description', '')} {row.get('category', '')}".lower()
            food_words = set(food_text.split())
            
            # Calculate simple overlap score
            overlap = len(query_words.intersection(food_words))
            score = overlap / len(query_words) if query_words else 0
            
            scores.append((score, row.to_dict()))
        
        # Sort by score and return top k
        scores.sort(key=lambda x: x[0], reverse=True)
        
        return [{'similarity_score': score, **data} for score, data in scores[:top_k]]
    
    def generate_meal_recommendations(self, parsed_query: Dict[str, Any], 
                                    meal_type: str = "general") -> List[Dict[str, Any]]:
        """
        Generate meal recommendations based on parsed query
        
        Args:
            parsed_query: Parsed user query information
            meal_type: Type of meal (breakfast, lunch, dinner, snack, general)
            
        Returns:
            List of recommended meals/foods
        """
        recommendations = []
        
        # Filter foods based on dietary restrictions
        filtered_data = self.nutrition_data.copy()
        
        # Apply diet type filters
        if parsed_query.get('diet_type'):
            filtered_data = self._apply_diet_filter(filtered_data, parsed_query['diet_type'])
        
        # Apply allergy filters
        if parsed_query.get('allergies'):
            for allergy in parsed_query['allergies']:
                filtered_data = self._apply_allergy_filter(filtered_data, allergy)
        
        # Generate recommendations based on health goals
        if parsed_query.get('health_goals'):
            for goal in parsed_query['health_goals']:
                goal_recommendations = self._get_goal_based_recommendations(filtered_data, goal)
                recommendations.extend(goal_recommendations)
        
        # If no specific goals, provide general recommendations
        if not recommendations:
            recommendations = self._get_general_recommendations(filtered_data, meal_type)
        
        # Score and rank recommendations
        scored_recommendations = self._score_recommendations(recommendations, parsed_query)
        
        return scored_recommendations[:10]  # Return top 10
    
    def _apply_diet_filter(self, data: pd.DataFrame, diet_type: str) -> pd.DataFrame:
        """Apply dietary restriction filters"""
        if diet_type == 'vegan':
            return data[~data['category'].isin(['meat', 'fish', 'dairy', 'eggs'])]
        elif diet_type == 'vegetarian':
            return data[~data['category'].isin(['meat', 'fish'])]
        elif diet_type == 'keto':
            return data[data['carbs_g'] < 10]  # Low carb threshold
        # Add more diet filters as needed
        return data
    
    def _apply_allergy_filter(self, data: pd.DataFrame, allergy: str) -> pd.DataFrame:
        """Apply allergy filters"""
        allergy_filters = {
            'nuts': lambda df: df[~df['food_name'].str.contains('nut|almond|walnut|cashew', case=False, na=False)],
            'dairy': lambda df: df[~df['category'].isin(['dairy'])],
            'eggs': lambda df: df[~df['category'].isin(['eggs'])],
            'soy': lambda df: df[~df['food_name'].str.contains('soy|tofu', case=False, na=False)],
            'wheat': lambda df: df[~df['food_name'].str.contains('wheat|bread|pasta', case=False, na=False)],
            'shellfish': lambda df: df[~df['food_name'].str.contains('shrimp|crab|lobster', case=False, na=False)],
            'fish': lambda df: df[~df['category'].isin(['fish'])]
        }
        
        if allergy in allergy_filters:
            return allergy_filters[allergy](data)
        return data
    
    def _get_goal_based_recommendations(self, data: pd.DataFrame, goal: str) -> List[Dict[str, Any]]:
        """Get recommendations based on health goals"""
        goal_info = self.nutrition_knowledge.get(goal, {})
        recommendations = []
        
        if goal == 'weight_loss':
            # Prioritize low calorie, high protein foods
            suitable_foods = data[
                (data['calories_per_100g'] < 200) & 
                (data['protein_g'] > 10)
            ].head(5)
        elif goal == 'muscle_gain':
            # Prioritize high protein foods
            suitable_foods = data[data['protein_g'] > 15].head(5)
        elif goal == 'heart_health':
            # Prioritize foods with healthy fats and fiber
            suitable_foods = data[
                (data['fiber_g'] > 3) | 
                (data['category'].isin(['fish', 'nuts']))
            ].head(5)
        else:
            suitable_foods = data.head(5)
        
        for _, food in suitable_foods.iterrows():
            rec = food.to_dict()
            rec['recommendation_reason'] = f"Good for {goal}: {goal_info.get('recommendations', 'beneficial')}"
            recommendations.append(rec)
        
        return recommendations
    
    def _get_general_recommendations(self, data: pd.DataFrame, meal_type: str) -> List[Dict[str, Any]]:
        """Get general meal recommendations"""
        meal_preferences = {
            'breakfast': ['eggs', 'oats', 'fruit', 'dairy'],
            'lunch': ['grain', 'vegetable', 'protein'],
            'dinner': ['meat', 'fish', 'vegetable', 'grain'],
            'snack': ['nuts', 'fruit', 'dairy']
        }
        
        if meal_type in meal_preferences:
            preferred_categories = meal_preferences[meal_type]
            suitable_foods = data[data['category'].isin(preferred_categories)].head(5)
        else:
            # Balanced selection
            suitable_foods = data.sample(min(10, len(data)))
        
        recommendations = []
        for _, food in suitable_foods.iterrows():
            rec = food.to_dict()
            rec['recommendation_reason'] = f"Good choice for {meal_type}"
            recommendations.append(rec)
        
        return recommendations
    
    def _score_recommendations(self, recommendations: List[Dict[str, Any]], 
                             parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Score and rank recommendations"""
        for rec in recommendations:
            score = 0
            
            # Base nutritional score
            if 'overall_nutrition_score' in rec:
                score += rec['overall_nutrition_score'] * 30
            
            # Preference matching
            food_name_lower = rec.get('food_name', '').lower()
            for pref in parsed_query.get('food_preferences', []):
                if pref in food_name_lower:
                    score += 20
            
            # Nutrient matching
            for nutrient in parsed_query.get('nutrients_mentioned', []):
                if nutrient == 'protein' and rec.get('protein_g', 0) > 15:
                    score += 15
                elif nutrient == 'fiber' and rec.get('fiber_g', 0) > 5:
                    score += 15
            
            rec['recommendation_score'] = score
        
        return sorted(recommendations, key=lambda x: x.get('recommendation_score', 0), reverse=True)
    
    def generate_nutrition_explanation(self, food_item: Dict[str, Any], 
                                     user_goals: List[str] = None) -> str:
        """
        Generate detailed nutrition explanation for a food item
        
        Args:
            food_item: Dictionary containing food nutrition data
            user_goals: List of user's health goals
            
        Returns:
            Detailed explanation string
        """
        name = food_item.get('food_name', 'Unknown food')
        calories = food_item.get('calories_per_100g', 0)
        protein = food_item.get('protein_g', 0)
        carbs = food_item.get('carbs_g', 0)
        fat = food_item.get('fat_g', 0)
        fiber = food_item.get('fiber_g', 0)
        
        explanation = f"**{name}** is a nutritious choice with {calories} calories per 100g. "
        
        # Macronutrient breakdown
        explanation += f"It contains {protein}g protein, {carbs}g carbohydrates, and {fat}g fat"
        if fiber > 0:
            explanation += f", plus {fiber}g fiber"
        explanation += ". "
        
        # Health benefits based on nutrient content
        benefits = []
        if protein > 15:
            benefits.append("excellent source of protein for muscle maintenance")
        if fiber > 5:
            benefits.append("high in fiber for digestive health")
        if calories < 100:
            benefits.append("low in calories for weight management")
        
        if benefits:
            explanation += f"This food is an {', and '.join(benefits)}. "
        
        # Goal-specific advice
        if user_goals:
            for goal in user_goals:
                if goal in self.nutrition_knowledge:
                    goal_info = self.nutrition_knowledge[goal]
                    explanation += f"For {goal.replace('_', ' ')}, this food supports your goals because it's {goal_info.get('recommendations', 'beneficial')}. "
        
        return explanation
    
    def analyze_meal_balance(self, meal_foods: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze nutritional balance of a complete meal
        
        Args:
            meal_foods: List of food items in the meal
            
        Returns:
            Analysis of meal's nutritional balance
        """
        if not meal_foods:
            return {"error": "No foods provided for analysis"}
        
        total_calories = sum(food.get('calories_per_100g', 0) for food in meal_foods)
        total_protein = sum(food.get('protein_g', 0) for food in meal_foods)
        total_carbs = sum(food.get('carbs_g', 0) for food in meal_foods)
        total_fat = sum(food.get('fat_g', 0) for food in meal_foods)
        total_fiber = sum(food.get('fiber_g', 0) for food in meal_foods)
        
        # Calculate macro percentages
        protein_calories = total_protein * 4
        carbs_calories = total_carbs * 4
        fat_calories = total_fat * 9
        
        total_macro_calories = protein_calories + carbs_calories + fat_calories
        
        analysis = {
            'total_calories': total_calories,
            'total_protein': total_protein,
            'total_carbs': total_carbs,
            'total_fat': total_fat,
            'total_fiber': total_fiber,
            'macro_ratios': {
                'protein_percentage': (protein_calories / total_macro_calories * 100) if total_macro_calories > 0 else 0,
                'carbs_percentage': (carbs_calories / total_macro_calories * 100) if total_macro_calories > 0 else 0,
                'fat_percentage': (fat_calories / total_macro_calories * 100) if total_macro_calories > 0 else 0
            },
            'balance_assessment': self._assess_macro_balance(protein_calories, carbs_calories, fat_calories),
            'recommendations': self._get_meal_recommendations(total_protein, total_carbs, total_fat, total_fiber)
        }
        
        return analysis
    
    def _assess_macro_balance(self, protein_cal: float, carbs_cal: float, fat_cal: float) -> str:
        """Assess the balance of macronutrients"""
        total = protein_cal + carbs_cal + fat_cal
        if total == 0:
            return "Unable to assess - no macronutrient data"
        
        protein_pct = protein_cal / total
        carbs_pct = carbs_cal / total
        fat_pct = fat_cal / total
        
        if 0.15 <= protein_pct <= 0.35 and 0.45 <= carbs_pct <= 0.65 and 0.20 <= fat_pct <= 0.35:
            return "Well-balanced macronutrient distribution"
        elif protein_pct < 0.15:
            return "Low in protein - consider adding more protein sources"
        elif carbs_pct > 0.70:
            return "High in carbohydrates - consider adding more protein or healthy fats"
        elif fat_pct > 0.40:
            return "High in fats - consider balancing with more carbohydrates or protein"
        else:
            return "Moderately balanced - minor adjustments could improve balance"
    
    def _get_meal_recommendations(self, protein: float, carbs: float, fat: float, fiber: float) -> List[str]:
        """Get recommendations to improve meal balance"""
        recommendations = []
        
        if protein < 20:
            recommendations.append("Add more protein sources like lean meat, fish, or legumes")
        if fiber < 10:
            recommendations.append("Include more high-fiber foods like vegetables or whole grains")
        if fat < 10:
            recommendations.append("Add healthy fats from sources like nuts, seeds, or avocado")
        
        if not recommendations:
            recommendations.append("This meal appears well-balanced nutritionally")
        
        return recommendations
    
    def process_complex_query(self, query: str) -> Dict[str, Any]:
        """
        Process complex nutrition queries and provide comprehensive responses
        
        Args:
            query: Complex user query
            
        Returns:
            Comprehensive response with recommendations and explanations
        """
        # Parse the query
        parsed_query = self.parse_user_query(query)
        
        # Find relevant foods
        similar_foods = self.find_similar_foods(query, top_k=5)
        
        # Generate meal recommendations
        meal_recommendations = self.generate_meal_recommendations(parsed_query)
        
        # Create comprehensive response
        response = {
            'parsed_query': parsed_query,
            'similar_foods': similar_foods,
            'meal_recommendations': meal_recommendations,
            'explanations': [],
            'meal_plan_suggestions': self._generate_meal_plan_suggestions(parsed_query)
        }
        
        # Generate explanations for top recommendations
        for food in meal_recommendations[:3]:
            explanation = self.generate_nutrition_explanation(food, parsed_query.get('health_goals'))
            response['explanations'].append({
                'food_name': food.get('food_name'),
                'explanation': explanation
            })
        
        return response
    
    def _generate_meal_plan_suggestions(self, parsed_query: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate daily meal plan suggestions"""
        suggestions = {
            'breakfast': [],
            'lunch': [],
            'dinner': [],
            'snacks': []
        }
        
        # Basic suggestions based on goals
        if 'weight_loss' in parsed_query.get('health_goals', []):
            suggestions['breakfast'] = ['Greek yogurt with berries', 'Vegetable omelet', 'Oatmeal with nuts']
            suggestions['lunch'] = ['Grilled chicken salad', 'Quinoa bowl with vegetables', 'Lentil soup']
            suggestions['dinner'] = ['Baked fish with broccoli', 'Lean turkey with sweet potato', 'Vegetable stir-fry']
            suggestions['snacks'] = ['Apple with almond butter', 'Carrot sticks with hummus', 'Greek yogurt']
        elif 'muscle_gain' in parsed_query.get('health_goals', []):
            suggestions['breakfast'] = ['Protein smoothie', 'Scrambled eggs with toast', 'Greek yogurt with granola']
            suggestions['lunch'] = ['Grilled chicken with rice', 'Salmon with quinoa', 'Turkey sandwich']
            suggestions['dinner'] = ['Lean beef with potatoes', 'Fish with pasta', 'Chicken stir-fry']
            suggestions['snacks'] = ['Protein shake', 'Nuts and dried fruit', 'Cottage cheese']
        else:
            # General healthy suggestions
            suggestions['breakfast'] = ['Whole grain cereal', 'Fresh fruit salad', 'Whole wheat toast with avocado']
            suggestions['lunch'] = ['Mixed green salad', 'Vegetable soup', 'Grilled sandwich']
            suggestions['dinner'] = ['Balanced plate with protein and vegetables', 'Pasta with marinara', 'Rice bowl']
            suggestions['snacks'] = ['Fresh fruit', 'Nuts', 'Yogurt']
        
        return suggestions