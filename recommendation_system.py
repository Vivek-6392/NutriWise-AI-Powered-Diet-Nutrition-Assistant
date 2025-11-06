"""
AI-Powered Recommendation System for Personalized Nutrition
Uses machine learning models and LLM integration for intelligent food suggestions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import openai
from datetime import datetime, timedelta
import requests

class PersonalizedNutritionRecommendationSystem:
    """
    Advanced recommendation system that combines ML models with LLM capabilities
    """
    
    def __init__(self, nutrition_data: pd.DataFrame, api_key: Optional[str] = None):
        """
        Initialize the recommendation system
        
        Args:
            nutrition_data: DataFrame containing nutrition information
            api_key: OpenAI API key for LLM integration (optional)
        """
        self.nutrition_data = nutrition_data
        self.api_key = api_key
        
        # ML Models
        self.preference_model = None
        self.nutrition_predictor = None
        self.clustering_model = None
        
        # Scalers and encoders
        self.feature_scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        
        # User profiles and history
        self.user_profiles = {}
        self.recommendation_history = {}
        
        # Initialize models
        self._initialize_models()
        
        # Nutrition scoring weights
        self.nutrition_weights = {
            'protein_quality': 0.25,
            'fiber_content': 0.20,
            'micronutrient_density': 0.20,
            'calorie_efficiency': 0.15,
            'ingredient_quality': 0.20
        }
        
        # Health condition mappings
        self.health_conditions = {
            'diabetes': {
                'avoid': ['high_sugar', 'refined_carbs'],
                'prefer': ['low_glycemic', 'high_fiber'],
                'nutrients': ['chromium', 'magnesium', 'fiber']
            },
            'hypertension': {
                'avoid': ['high_sodium', 'processed'],
                'prefer': ['potassium_rich', 'magnesium_rich'],
                'nutrients': ['potassium', 'magnesium', 'calcium']
            },
            'heart_disease': {
                'avoid': ['saturated_fat', 'trans_fat'],
                'prefer': ['omega3', 'antioxidants'],
                'nutrients': ['omega3', 'vitamin_e', 'folate']
            },
            'obesity': {
                'avoid': ['high_calorie', 'processed'],
                'prefer': ['low_calorie_dense', 'high_protein'],
                'nutrients': ['protein', 'fiber', 'chromium']
            }
        }
    
    def _initialize_models(self):
        """Initialize and train ML models"""
        try:
            self._prepare_training_data()
            self._train_preference_model()
            self._train_nutrition_predictor()
            self._create_food_clusters()
            logging.info("ML models initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
    
    def _prepare_training_data(self):
        """Prepare data for training ML models"""
        # Create synthetic user preference data for training
        np.random.seed(42)
        n_samples = len(self.nutrition_data) * 5
        
        # Generate synthetic user preferences
        self.training_data = []
        
        for _ in range(n_samples):
            # Random user characteristics
            age = np.random.randint(18, 80)
            weight = np.random.normal(70, 15)
            height = np.random.normal(170, 10)
            activity_level = np.random.choice(['sedentary', 'light', 'moderate', 'active', 'very_active'])
            health_goal = np.random.choice(['weight_loss', 'weight_gain', 'muscle_gain', 'maintenance'])
            
            # Random food item
            food_idx = np.random.randint(0, len(self.nutrition_data))
            food = self.nutrition_data.iloc[food_idx]
            
            # Generate preference score (0-1) based on user characteristics and food properties
            preference_score = self._generate_synthetic_preference(
                age, weight, height, activity_level, health_goal, food
            )
            
            self.training_data.append({
                'age': age,
                'weight': weight,
                'height': height,
                'activity_level': activity_level,
                'health_goal': health_goal,
                'food_calories': food.get('calories_per_100g', 0),
                'food_protein': food.get('protein_g', 0),
                'food_carbs': food.get('carbs_g', 0),
                'food_fat': food.get('fat_g', 0),
                'food_fiber': food.get('fiber_g', 0),
                'food_category': food.get('category', 'unknown'),
                'preference_score': preference_score
            })
        
        self.training_df = pd.DataFrame(self.training_data)
    
    def _generate_synthetic_preference(self, age: int, weight: float, height: float, 
                                     activity_level: str, health_goal: str, food: pd.Series) -> float:
        """Generate synthetic preference scores for training"""
        score = 0.5  # Base score
        
        # Adjust based on health goal
        if health_goal == 'weight_loss':
            if food.get('calories_per_100g', 0) < 200:
                score += 0.2
            if food.get('protein_g', 0) > 15:
                score += 0.15
        elif health_goal == 'muscle_gain':
            if food.get('protein_g', 0) > 20:
                score += 0.25
            if food.get('calories_per_100g', 0) > 250:
                score += 0.1
        
        # Adjust based on age
        if age > 50 and food.get('fiber_g', 0) > 5:
            score += 0.1
        
        # Add some randomness
        score += np.random.normal(0, 0.1)
        
        return np.clip(score, 0, 1)
    
    def _train_preference_model(self):
        """Train the user preference prediction model"""
        if hasattr(self, 'training_df') and not self.training_df.empty:
            # Prepare features
            feature_columns = ['age', 'weight', 'height', 'food_calories', 
                             'food_protein', 'food_carbs', 'food_fat', 'food_fiber']
            
            # Encode categorical variables
            activity_encoder = LabelEncoder()
            goal_encoder = LabelEncoder()
            category_encoder = LabelEncoder()
            
            X_categorical = pd.DataFrame({
                'activity_level': activity_encoder.fit_transform(self.training_df['activity_level']),
                'health_goal': goal_encoder.fit_transform(self.training_df['health_goal']),
                'food_category': category_encoder.fit_transform(self.training_df['food_category'])
            })
            
            X_numerical = self.training_df[feature_columns]
            X = pd.concat([X_numerical, X_categorical], axis=1)
            y = self.training_df['preference_score']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train model
            self.preference_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Convert regression to classification for this example
            y_train_class = (y_train > 0.6).astype(int)
            y_test_class = (y_test > 0.6).astype(int)
            
            self.preference_model.fit(X_train_scaled, y_train_class)
            
            # Evaluate
            predictions = self.preference_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test_class, predictions)
            logging.info(f"Preference model accuracy: {accuracy:.3f}")
            
            # Store encoders
            self.encoders = {
                'activity': activity_encoder,
                'goal': goal_encoder,
                'category': category_encoder
            }
    
    def _train_nutrition_predictor(self):
        """Train nutrition quality prediction model"""
        if len(self.nutrition_data) > 0:
            # Create nutrition quality score
            nutrition_scores = []
            for _, food in self.nutrition_data.iterrows():
                score = self._calculate_nutrition_quality_score(food)
                nutrition_scores.append(score)
            
            # Prepare features
            feature_columns = ['calories_per_100g', 'protein_g', 'carbs_g', 'fat_g', 'fiber_g']
            available_columns = [col for col in feature_columns if col in self.nutrition_data.columns]
            
            if available_columns:
                X = self.nutrition_data[available_columns].fillna(0)
                y = np.array(nutrition_scores)
                
                # Train model
                self.nutrition_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
                self.nutrition_predictor.fit(X, y)
                
                # Evaluate
                predictions = self.nutrition_predictor.predict(X)
                mse = mean_squared_error(y, predictions)
                logging.info(f"Nutrition predictor MSE: {mse:.3f}")
    
    def _calculate_nutrition_quality_score(self, food: pd.Series) -> float:
        """Calculate nutrition quality score for a food item"""
        score = 0
        
        # Protein quality
        protein = food.get('protein_g', 0)
        if protein > 20:
            score += 0.3
        elif protein > 10:
            score += 0.2
        elif protein > 5:
            score += 0.1
        
        # Fiber content
        fiber = food.get('fiber_g', 0)
        if fiber > 10:
            score += 0.2
        elif fiber > 5:
            score += 0.15
        elif fiber > 2:
            score += 0.1
        
        # Calorie efficiency (nutrients per calorie)
        calories = food.get('calories_per_100g', 1)
        nutrient_density = (protein + fiber) / calories if calories > 0 else 0
        score += min(nutrient_density * 0.5, 0.3)
        
        # Category bonus
        category = food.get('category', '')
        if category in ['vegetable', 'fruit', 'fish']:
            score += 0.2
        elif category in ['nuts', 'grain']:
            score += 0.1
        
        return min(score, 1.0)
    
    def _create_food_clusters(self):
        """Create food clusters for recommendation diversity"""
        if len(self.nutrition_data) > 5:
            feature_columns = ['calories_per_100g', 'protein_g', 'carbs_g', 'fat_g', 'fiber_g']
            available_columns = [col for col in feature_columns if col in self.nutrition_data.columns]
            
            if available_columns:
                X = self.nutrition_data[available_columns].fillna(0)
                
                # Determine optimal number of clusters
                n_clusters = min(8, max(3, len(self.nutrition_data) // 10))
                
                self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = self.clustering_model.fit_predict(X)
                
                self.nutrition_data['cluster'] = clusters
                logging.info(f"Created {n_clusters} food clusters")
    
    def create_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update user profile
        
        Args:
            user_id: Unique user identifier
            profile_data: User profile information
            
        Returns:
            Created/updated user profile
        """
        required_fields = ['age', 'weight', 'height', 'activity_level', 'health_goals']
        
        # Validate required fields
        for field in required_fields:
            if field not in profile_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Calculate additional metrics
        bmi = self._calculate_bmi(profile_data['weight'], profile_data['height'])
        bmr = self._calculate_bmr(profile_data['weight'], profile_data['height'], 
                                profile_data['age'], profile_data.get('gender', 'unknown'))
        tdee = self._calculate_tdee(bmr, profile_data['activity_level'])
        
        user_profile = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'basic_info': profile_data,
            'calculated_metrics': {
                'bmi': bmi,
                'bmr': bmr,
                'tdee': tdee,
                'bmi_category': self._get_bmi_category(bmi)
            },
            'dietary_restrictions': profile_data.get('dietary_restrictions', []),
            'allergies': profile_data.get('allergies', []),
            'health_conditions': profile_data.get('health_conditions', []),
            'preferences': profile_data.get('food_preferences', {}),
            'recommendation_history': []
        }
        
        self.user_profiles[user_id] = user_profile
        return user_profile
    
    def _calculate_bmi(self, weight: float, height: float) -> float:
        """Calculate BMI"""
        height_m = height / 100  # Convert cm to m
        return weight / (height_m ** 2)
    
    def _calculate_bmr(self, weight: float, height: float, age: int, gender: str) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation"""
        if gender.lower() == 'male':
            return 10 * weight + 6.25 * height - 5 * age + 5
        else:  # female or unknown
            return 10 * weight + 6.25 * height - 5 * age - 161
    
    def _calculate_tdee(self, bmr: float, activity_level: str) -> float:
        """Calculate Total Daily Energy Expenditure"""
        activity_multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        
        return bmr * activity_multipliers.get(activity_level, 1.2)
    
    def _get_bmi_category(self, bmi: float) -> str:
        """Get BMI category"""
        if bmi < 18.5:
            return 'underweight'
        elif bmi < 25:
            return 'normal'
        elif bmi < 30:
            return 'overweight'
        else:
            return 'obese'
    
    def generate_personalized_recommendations(self, user_id: str, 
                                            meal_type: str = 'general',
                                            num_recommendations: int = 10) -> Dict[str, Any]:
        """
        Generate personalized food recommendations for a user
        
        Args:
            user_id: User identifier
            meal_type: Type of meal (breakfast, lunch, dinner, snack)
            num_recommendations: Number of recommendations to return
            
        Returns:
            Dictionary with recommendations and explanations
        """
        if user_id not in self.user_profiles:
            raise ValueError(f"User profile not found: {user_id}")
        
        user_profile = self.user_profiles[user_id]
        
        # Get base recommendations
        base_recommendations = self._get_base_recommendations(user_profile, meal_type)
        
        # Apply ML-based scoring
        ml_scored_recommendations = self._apply_ml_scoring(base_recommendations, user_profile)
        
        # Apply rule-based filtering
        filtered_recommendations = self._apply_rule_based_filtering(ml_scored_recommendations, user_profile)
        
        # Diversify recommendations
        diverse_recommendations = self._diversify_recommendations(filtered_recommendations, num_recommendations)
        
        # Generate explanations
        explanations = self._generate_recommendation_explanations(diverse_recommendations, user_profile)
        
        # Create meal plan suggestions
        meal_plan = self._create_personalized_meal_plan(user_profile)
        
        # Update recommendation history
        self._update_recommendation_history(user_id, diverse_recommendations)
        
        return {
            'user_id': user_id,
            'meal_type': meal_type,
            'recommendations': diverse_recommendations,
            'explanations': explanations,
            'meal_plan_suggestions': meal_plan,
            'nutritional_targets': self._calculate_nutritional_targets(user_profile),
            'generated_at': datetime.now().isoformat()
        }
    
    def _get_base_recommendations(self, user_profile: Dict[str, Any], meal_type: str) -> List[Dict[str, Any]]:
        """Get base food recommendations based on user profile"""
        recommendations = []
        
        # Filter foods based on dietary restrictions
        filtered_foods = self.nutrition_data.copy()
        
        # Apply dietary restrictions
        restrictions = user_profile.get('dietary_restrictions', [])
        for restriction in restrictions:
            filtered_foods = self._apply_dietary_restriction(filtered_foods, restriction)
        
        # Apply allergy filters
        allergies = user_profile.get('allergies', [])
        for allergy in allergies:
            filtered_foods = self._apply_allergy_filter(filtered_foods, allergy)
        
        # Get meal-type specific foods
        meal_specific_foods = self._filter_by_meal_type(filtered_foods, meal_type)
        
        # Convert to list of dictionaries
        for _, food in meal_specific_foods.iterrows():
            food_dict = food.to_dict()
            food_dict['base_score'] = self._calculate_base_score(food_dict, user_profile)
            recommendations.append(food_dict)
        
        return recommendations
    
    def _apply_dietary_restriction(self, foods: pd.DataFrame, restriction: str) -> pd.DataFrame:
        """Apply dietary restriction filters"""
        restriction_filters = {
            'vegetarian': lambda df: df[~df['category'].isin(['meat', 'fish'])],
            'vegan': lambda df: df[~df['category'].isin(['meat', 'fish', 'dairy', 'eggs'])],
            'pescatarian': lambda df: df[~df['category'].isin(['meat'])],
            'keto': lambda df: df[df['carbs_g'] <= 10],
            'low_carb': lambda df: df[df['carbs_g'] <= 20],
            'paleo': lambda df: df[~df['food_name'].str.contains('grain|dairy|legume', case=False, na=False)],
            'gluten_free': lambda df: df[~df['food_name'].str.contains('wheat|gluten|bread', case=False, na=False)]
        }
        
        if restriction in restriction_filters:
            return restriction_filters[restriction](foods)
        return foods
    
    def _apply_allergy_filter(self, foods: pd.DataFrame, allergy: str) -> pd.DataFrame:
        """Apply allergy filters - same as in NLP engine"""
        allergy_filters = {
            'nuts': lambda df: df[~df['food_name'].str.contains('nut|almond|walnut|cashew|peanut', case=False, na=False)],
            'dairy': lambda df: df[df['category'] != 'dairy'],
            'eggs': lambda df: df[df['category'] != 'eggs'],
            'soy': lambda df: df[~df['food_name'].str.contains('soy|tofu', case=False, na=False)],
            'wheat': lambda df: df[~df['food_name'].str.contains('wheat|bread|pasta', case=False, na=False)],
            'shellfish': lambda df: df[~df['food_name'].str.contains('shrimp|crab|lobster', case=False, na=False)],
            'fish': lambda df: df[df['category'] != 'fish']
        }
        
        if allergy in allergy_filters:
            return allergy_filters[allergy](foods)
        return foods
    
    def _filter_by_meal_type(self, foods: pd.DataFrame, meal_type: str) -> pd.DataFrame:
        """Filter foods appropriate for meal type"""
        meal_categories = {
            'breakfast': ['grain', 'fruit', 'dairy', 'eggs'],
            'lunch': ['vegetable', 'grain', 'protein', 'nuts'],
            'dinner': ['meat', 'fish', 'vegetable', 'grain'],
            'snack': ['fruit', 'nuts', 'dairy']
        }
        
        if meal_type in meal_categories:
            preferred_categories = meal_categories[meal_type]
            # Include foods from preferred categories plus some others
            filtered = foods[foods['category'].isin(preferred_categories)]
            if len(filtered) < 5:  # If too few, include all
                return foods
            return filtered
        
        return foods
    
    def _calculate_base_score(self, food: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """Calculate base recommendation score"""
        score = 0.5  # Base score
        
        # Health goal alignment
        health_goals = user_profile['basic_info'].get('health_goals', [])
        for goal in health_goals:
            if goal == 'weight_loss':
                if food.get('calories_per_100g', 0) < 200:
                    score += 0.15
                if food.get('protein_g', 0) > 15:
                    score += 0.1
                if food.get('fiber_g', 0) > 5:
                    score += 0.1
            elif goal == 'muscle_gain':
                if food.get('protein_g', 0) > 20:
                    score += 0.2
                if food.get('calories_per_100g', 0) > 250:
                    score += 0.1
            elif goal == 'heart_health':
                if food.get('category') in ['fish', 'nuts', 'vegetable']:
                    score += 0.15
                if food.get('fiber_g', 0) > 3:
                    score += 0.1
        
        # BMI-based adjustments
        bmi_category = user_profile['calculated_metrics']['bmi_category']
        if bmi_category == 'overweight' or bmi_category == 'obese':
            if food.get('calories_per_100g', 0) < 150:
                score += 0.1
        elif bmi_category == 'underweight':
            if food.get('calories_per_100g', 0) > 300:
                score += 0.1
        
        # Health condition adjustments
        health_conditions = user_profile.get('health_conditions', [])
        for condition in health_conditions:
            if condition in self.health_conditions:
                condition_info = self.health_conditions[condition]
                # Apply condition-specific scoring logic
                if food.get('category') in condition_info.get('prefer', []):
                    score += 0.1
        
        return np.clip(score, 0, 1)
    
    def _apply_ml_scoring(self, recommendations: List[Dict[str, Any]], 
                         user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply ML model scoring to recommendations"""
        if self.preference_model is None:
            return recommendations
        
        try:
            for rec in recommendations:
                # Prepare features for ML model
                features = self._prepare_ml_features(rec, user_profile)
                
                # Get ML prediction
                ml_score = self.preference_model.predict_proba([features])[0][1]  # Probability of positive class
                
                # Combine with base score
                rec['ml_score'] = ml_score
                rec['combined_score'] = 0.6 * rec.get('base_score', 0.5) + 0.4 * ml_score
        
        except Exception as e:
            logging.warning(f"Error applying ML scoring: {e}")
            # Fall back to base scores
            for rec in recommendations:
                rec['ml_score'] = rec.get('base_score', 0.5)
                rec['combined_score'] = rec.get('base_score', 0.5)
        
        return recommendations
    
    def _prepare_ml_features(self, food: Dict[str, Any], user_profile: Dict[str, Any]) -> List[float]:
        """Prepare features for ML model prediction"""
        basic_info = user_profile['basic_info']
        
        # Map categorical variables
        activity_mapping = {'sedentary': 0, 'light': 1, 'moderate': 2, 'active': 3, 'very_active': 4}
        goal_mapping = {'weight_loss': 0, 'weight_gain': 1, 'muscle_gain': 2, 'maintenance': 3}
        category_mapping = {'meat': 0, 'fish': 1, 'vegetable': 2, 'fruit': 3, 'grain': 4, 'dairy': 5, 'nuts': 6, 'eggs': 7}
        
        features = [
            basic_info.get('age', 30),
            basic_info.get('weight', 70),
            basic_info.get('height', 170),
            food.get('calories_per_100g', 0),
            food.get('protein_g', 0),
            food.get('carbs_g', 0),
            food.get('fat_g', 0),
            food.get('fiber_g', 0),
            activity_mapping.get(basic_info.get('activity_level', 'moderate'), 2),
            goal_mapping.get(basic_info.get('health_goals', [{}])[0] if basic_info.get('health_goals') else 'maintenance', 3),
            category_mapping.get(food.get('category', 'unknown'), 0)
        ]
        
        return features
    
    def _apply_rule_based_filtering(self, recommendations: List[Dict[str, Any]], 
                                   user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply rule-based filtering and adjustments"""
        filtered_recommendations = []
        
        for rec in recommendations:
            # Skip foods that don't meet minimum thresholds
            if rec.get('combined_score', 0) < 0.3:
                continue
            
            # Apply health condition rules
            health_conditions = user_profile.get('health_conditions', [])
            skip_food = False
            
            for condition in health_conditions:
                if condition in self.health_conditions:
                    condition_info = self.health_conditions[condition]
                    
                    # Check avoid list
                    for avoid_item in condition_info.get('avoid', []):
                        if self._food_matches_criteria(rec, avoid_item):
                            skip_food = True
                            break
                    
                    if skip_food:
                        break
            
            if not skip_food:
                # Boost score for preferred foods
                for condition in health_conditions:
                    if condition in self.health_conditions:
                        condition_info = self.health_conditions[condition]
                        
                        for prefer_item in condition_info.get('prefer', []):
                            if self._food_matches_criteria(rec, prefer_item):
                                rec['combined_score'] += 0.1
                                rec['health_boost'] = True
                                break
                
                filtered_recommendations.append(rec)
        
        return filtered_recommendations
    
    def _food_matches_criteria(self, food: Dict[str, Any], criteria: str) -> bool:
        """Check if food matches specific criteria"""
        criteria_checks = {
            'high_sugar': lambda f: f.get('carbs_g', 0) > 15,  # Simplified check
            'high_sodium': lambda f: 'sodium' in f.get('food_name', '').lower(),
            'high_calorie': lambda f: f.get('calories_per_100g', 0) > 300,
            'low_calorie_dense': lambda f: f.get('calories_per_100g', 0) < 150,
            'high_protein': lambda f: f.get('protein_g', 0) > 15,
            'low_glycemic': lambda f: f.get('fiber_g', 0) > 3,
            'omega3': lambda f: f.get('category') == 'fish',
            'potassium_rich': lambda f: f.get('category') in ['vegetable', 'fruit']
        }
        
        return criteria_checks.get(criteria, lambda f: False)(food)
    
    def _diversify_recommendations(self, recommendations: List[Dict[str, Any]], 
                                 num_recommendations: int) -> List[Dict[str, Any]]:
        """Ensure diversity in recommendations"""
        # Sort by combined score
        sorted_recs = sorted(recommendations, key=lambda x: x.get('combined_score', 0), reverse=True)
        
        # Select diverse recommendations
        diverse_recs = []
        categories_used = set()
        
        # First pass: select best from each category
        for rec in sorted_recs:
            if len(diverse_recs) >= num_recommendations:
                break
            
            category = rec.get('category', 'unknown')
            if category not in categories_used:
                diverse_recs.append(rec)
                categories_used.add(category)
        
        # Second pass: fill remaining slots with highest scores
        for rec in sorted_recs:
            if len(diverse_recs) >= num_recommendations:
                break
            
            if rec not in diverse_recs:
                diverse_recs.append(rec)
        
        return diverse_recs[:num_recommendations]
    
    def _generate_recommendation_explanations(self, recommendations: List[Dict[str, Any]], 
                                            user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate explanations for recommendations"""
        explanations = []
        
        for rec in recommendations:
            explanation = {
                'food_name': rec.get('food_name', 'Unknown'),
                'reasons': [],
                'nutritional_highlights': [],
                'goal_alignment': []
            }
            
            # Nutritional highlights
            if rec.get('protein_g', 0) > 15:
                explanation['nutritional_highlights'].append(f"High in protein ({rec.get('protein_g', 0)}g per 100g)")
            
            if rec.get('fiber_g', 0) > 5:
                explanation['nutritional_highlights'].append(f"Rich in fiber ({rec.get('fiber_g', 0)}g per 100g)")
            
            if rec.get('calories_per_100g', 0) < 150:
                explanation['nutritional_highlights'].append("Low in calories")
            
            # Goal alignment
            health_goals = user_profile['basic_info'].get('health_goals', [])
            for goal in health_goals:
                if goal == 'weight_loss' and rec.get('calories_per_100g', 0) < 200:
                    explanation['goal_alignment'].append("Supports weight loss goals with lower calorie content")
                elif goal == 'muscle_gain' and rec.get('protein_g', 0) > 20:
                    explanation['goal_alignment'].append("Excellent for muscle gain with high protein content")
            
            # Health condition benefits
            if rec.get('health_boost'):
                explanation['reasons'].append("Specifically beneficial for your health conditions")
            
            # General reasons
            if rec.get('combined_score', 0) > 0.8:
                explanation['reasons'].append("Highly recommended based on your profile")
            elif rec.get('combined_score', 0) > 0.6:
                explanation['reasons'].append("Good match for your preferences")
            
            explanations.append(explanation)
        
        return explanations
    
    def _create_personalized_meal_plan(self, user_profile: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create a personalized daily meal plan"""
        meal_plan = {
            'breakfast': [],
            'lunch': [],
            'dinner': [],
            'snacks': []
        }
        
        # Get recommendations for each meal type
        for meal_type in meal_plan.keys():
            try:
                meal_recs = self.generate_personalized_recommendations(
                    user_profile['user_id'], meal_type, 3
                )
                meal_plan[meal_type] = [rec['food_name'] for rec in meal_recs.get('recommendations', [])]
            except:
                # Fallback to general suggestions
                meal_plan[meal_type] = ['Balanced meal options', 'Consult nutrition guidelines']
        
        return meal_plan
    
    def _calculate_nutritional_targets(self, user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Calculate daily nutritional targets for user"""
        calculated_metrics = user_profile.get('calculated_metrics', {})
        basic_info = user_profile.get('basic_info', {})
        
        tdee = calculated_metrics.get('tdee', 2000)
        
        # Adjust calories based on health goals
        health_goals = basic_info.get('health_goals', [])
        calorie_target = tdee
        
        if 'weight_loss' in health_goals:
            calorie_target = tdee * 0.8  # 20% deficit
        elif 'weight_gain' in health_goals:
            calorie_target = tdee * 1.2  # 20% surplus
        
        # Macro targets (as percentages of calories)
        protein_pct = 0.25 if 'muscle_gain' in health_goals else 0.20
        carb_pct = 0.45
        fat_pct = 1.0 - protein_pct - carb_pct
        
        targets = {
            'calories': calorie_target,
            'protein_g': (calorie_target * protein_pct) / 4,  # 4 cal/g
            'carbohydrates_g': (calorie_target * carb_pct) / 4,
            'fat_g': (calorie_target * fat_pct) / 9,  # 9 cal/g
            'fiber_g': 25 if basic_info.get('gender') == 'female' else 38,
            'water_l': 2.2 if basic_info.get('gender') == 'female' else 2.9
        }
        
        return targets
    
    def _update_recommendation_history(self, user_id: str, recommendations: List[Dict[str, Any]]):
        """Update user's recommendation history"""
        if user_id in self.user_profiles:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'recommendations': [rec.get('food_name') for rec in recommendations],
                'scores': [rec.get('combined_score') for rec in recommendations]
            }
            
            self.user_profiles[user_id]['recommendation_history'].append(history_entry)
            
            # Keep only last 50 entries
            if len(self.user_profiles[user_id]['recommendation_history']) > 50:
                self.user_profiles[user_id]['recommendation_history'] = \
                    self.user_profiles[user_id]['recommendation_history'][-50:]
    
    def get_llm_enhanced_recommendations(self, user_id: str, query: str) -> Dict[str, Any]:
        """
        Get LLM-enhanced recommendations with detailed explanations
        
        Args:
            user_id: User identifier
            query: Natural language query from user
            
        Returns:
            Enhanced recommendations with LLM-generated explanations
        """
        if not self.api_key:
            # Return standard recommendations if no API key
            return self.generate_personalized_recommendations(user_id)
        
        # Get base recommendations
        base_recommendations = self.generate_personalized_recommendations(user_id)
        
        # Prepare context for LLM
        user_profile = self.user_profiles[user_id]
        context = self._prepare_llm_context(user_profile, base_recommendations, query)
        
        try:
            # Call LLM for enhanced explanations
            enhanced_response = self._call_llm_api(context)
            
            # Parse and integrate LLM response
            base_recommendations['llm_explanation'] = enhanced_response
            base_recommendations['enhanced'] = True
            
        except Exception as e:
            logging.error(f"Error calling LLM API: {e}")
            base_recommendations['llm_explanation'] = "Standard recommendations provided"
            base_recommendations['enhanced'] = False
        
        return base_recommendations
    
    def _prepare_llm_context(self, user_profile: Dict[str, Any], 
                           recommendations: Dict[str, Any], query: str) -> str:
        """Prepare context for LLM API call"""
        context = f"""
        User Profile:
        - Age: {user_profile['basic_info'].get('age')}
        - Health Goals: {', '.join(user_profile['basic_info'].get('health_goals', []))}
        - BMI Category: {user_profile['calculated_metrics'].get('bmi_category')}
        - Dietary Restrictions: {', '.join(user_profile.get('dietary_restrictions', []))}
        - Allergies: {', '.join(user_profile.get('allergies', []))}
        
        User Query: {query}
        
        Top Recommended Foods:
        """
        
        for i, rec in enumerate(recommendations.get('recommendations', [])[:5]):
            context += f"""
        {i+1}. {rec.get('food_name')} 
           - Calories: {rec.get('calories_per_100g')}
           - Protein: {rec.get('protein_g')}g
           - Score: {rec.get('combined_score', 0):.2f}
        """
        
        context += """
        
        Please provide personalized nutrition advice explaining why these foods are recommended for this user, 
        considering their profile and query. Include specific nutritional benefits and how they align with their goals.
        """
        
        return context
    
    def _call_llm_api(self, context: str) -> str:
        """Call LLM API for enhanced recommendations"""
        # This is a placeholder for LLM API integration
        # In a real implementation, you would call OpenAI API or another LLM service
        
        try:
            # Example OpenAI API call (commented out as it requires actual API key)
            # response = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=[
            #         {"role": "system", "content": "You are a professional nutritionist providing personalized advice."},
            #         {"role": "user", "content": context}
            #     ],
            #     max_tokens=500
            # )
            # return response.choices[0].message.content
            
            # Fallback response for demo
            return """Based on your profile and goals, these recommendations are tailored to support your health journey. 
            The selected foods provide balanced nutrition with appropriate calorie content and macronutrient ratios 
            that align with your specific objectives."""
            
        except Exception as e:
            logging.error(f"LLM API call failed: {e}")
            return "Personalized recommendations based on your nutritional needs and preferences."
    
    def save_model(self, filepath: str):
        """Save trained models and user data"""
        model_data = {
            'preference_model': self.preference_model,
            'nutrition_predictor': self.nutrition_predictor,
            'clustering_model': self.clustering_model,
            'feature_scaler': self.feature_scaler,
            'encoders': getattr(self, 'encoders', {}),
            'user_profiles': self.user_profiles,
            'nutrition_weights': self.nutrition_weights
        }
        
        try:
            joblib.dump(model_data, filepath)
            logging.info(f"Models saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving models: {e}")
    
    def load_model(self, filepath: str):
        """Load trained models and user data"""
        try:
            model_data = joblib.load(filepath)
            
            self.preference_model = model_data.get('preference_model')
            self.nutrition_predictor = model_data.get('nutrition_predictor')
            self.clustering_model = model_data.get('clustering_model')
            self.feature_scaler = model_data.get('feature_scaler', StandardScaler())
            self.encoders = model_data.get('encoders', {})
            self.user_profiles = model_data.get('user_profiles', {})
            self.nutrition_weights = model_data.get('nutrition_weights', self.nutrition_weights)
            
            logging.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            # Initialize fresh models if loading fails
            self._initialize_models()