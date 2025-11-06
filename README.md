# 🧠 NLP for Diet and Nutrition Analysis

**Team Members:**  
Vivek Yadav [BT23CSA035] · Ojaswa Awasthi [BT23CSA060]  
**Guide:** Dr. Amol Bhopale  
**Institute:** Indian Institute of Information Technology, Nagpur  

---

## 🩺 Introduction

### 🎯 Objective
To develop a **personalized AI diet and nutrition assistant** that analyzes user dietary habits and health data, providing **tailored recommendations** for improved well-being.

### 💡 Motivation
With the rise of health consciousness and the need for personalized diet plans, there’s a major gap in accessible, data-driven nutrition guidance.  
This project leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** to bridge that gap with intelligent automation and education.

### 🔍 Core Idea
Use NLP for **semantic understanding of food items, recipes, and user health profiles** to dynamically generate:
- Custom meal plans  
- Nutritional insights  
- Context-aware recommendations  

---

## 🚨 The Nutrition Problem We’re Solving

| Challenge | Explanation |
|------------|--------------|
| **Rising Health Crisis** | Global increase in obesity, diabetes, and lifestyle diseases. |
| **Technology Gap** | Traditional diet apps rely on static rule-based logic. |
| **User Confusion** | Non-personalized, complex data leads to poor engagement. |
| **Overwhelming Data** | Nutrition labels and datasets are difficult to interpret. |
| **Dietary Restrictions** | Users with allergies or medical conditions lack tailored advice. |

---

## 🎯 Project Objectives

1. **Automated Food Analysis** – Analyze food and meal nutrients using NLP and nutrition databases.  
2. **Personalized Meal Planning** – Generate custom meal plans for user goals and health conditions.  
3. **Nutritional Education** – Explain nutrition concepts clearly to empower user awareness.  
4. **Health-Based Recommendations** – Provide targeted advice for diabetes, obesity, or heart conditions.  

---

## 🧠 Literature Review

### Part 1: Variational Autoencoder (VAE) for Meal Generation
1. **User Input:** Profile data (weight, height, health conditions) → input vector  
2. **VAE Encoder:** Maps input to latent parameters (μ, σ)  
3. **Latent Sampling:** Reparameterization produces latent vector *z*  
4. **GRU/LSTM Decoder:** Generates 6 sequential meals (breakfast → supper)  
5. **Multi-Head Output:** Predicts meal type, nutrients, and calories  

---

### Part 2: Dataset and Techniques

**Datasets**
- **University of Toronto FLIP 2017 & 2020**  
  - 19,720 to 74,445 food products  
- **Health Canada TRA (24 categories, 172 subcategories)**  
- **FSANZ Nutrient Profiling System**  
- **Indian Nutrition Database (Anuvaad_INDB_2024.11)**  

**Techniques & Models**
| Technique | Purpose |
|------------|----------|
| **Sentence-BERT (SBERT)** | Encodes food label text into semantic vectors |
| **t-SNE** | Visualizes product clusters in 2D |
| **Elastic Net** | Regression using Lasso + Ridge regularization |
| **K-Nearest Neighbors (KNN)** | Classifies food based on similarity |
| **XGBoost** | High-accuracy food categorization and nutrition prediction |

---

### Model Performance

| Task | Model | Metric | Result |
|------|--------|---------|--------|
| Major Food Category | XGBoost + SBERT | Accuracy | **0.98** |
| Subcategory | XGBoost + SBERT | Accuracy | **0.96** |
| FSANZ Nutrition Score (Structured) | Regression | R² = **0.98**, MSE = **2.5** |
| FSANZ Nutrition Score (SBERT) | Regression | R² = **0.87**, MSE = **14.4** |

**Key Findings**
- SBERT outperforms Bag-of-Words for food classification.  
- t-SNE effectively clusters food categories.  
- Structured nutrition data yields the highest predictive accuracy.  
- The pipeline generalizes globally for food categorization tasks.  

---

## 🧩 System Architecture

### 🔹 Dataset for Nutrition Analysis
- **Source:** Indian Nutrition Database *(Anuvaad_INDB_2024.11)*  
- **Data Points:** Calories, proteins, fats, carbs, vitamins, minerals  
- **Examples:**
  | Food | Serving | Energy | Protein | Fat | Carbs |
  |-------|----------|---------|----------|------|-------|
  | Hot Tea | 100ml | 1 kcal | 0.1g | 0g | 0.2g |
  | Mango Drink | 200ml | 100 kcal | 0.5g | 0.1g | 25g |
  | Espresso Coffee | 30ml | 2 kcal | 0.2g | 0g | 0.3g |

---

## ⚙️ Core Components

### 🧠 NLP Engine
- **Query Understanding:** Extracts dietary goals, allergies, and preferences.  
- **Semantic Food Search:** Finds relevant food matches via knowledge graphs.  
- **Recommendation Generation:** Provides context-aware, nutritionally balanced suggestions.  

### 🤖 ML Recommendation System
- **Algorithms:** RandomForest, GradientBoosting, Clustering, PCA  
- **Adaptive Learning:** Learns from user feedback to refine future suggestions.  
- **Health Condition Mapping:**  
  - Diabetes → High-fiber, low-sugar  
  - Heart Disease → Omega-3 rich, low-saturated fat  
  - Weight Management → Balanced satiety and caloric density  

### 🧬 LLM Integration
- **Nutritionist Role:** Provides scientific analysis and evidence-based recommendations.  
- **Meal Planner Role:** Builds balanced meal schedules considering portion synergy.  
- **Educator Role:** Explains complex nutrition ideas in simple language.  
- **Recommender Role:** Suggests modifications and explains their health benefits.  

---

## 🌍 Future Scope

- **Expanded Global Food Database** for wider personalization.  
- **AI-Powered Health Insights** predicting disease risk from eating patterns.  
- **Voice & Chat Interfaces** for hands-free interaction.  
- **Mobile Integration** for real-time diet tracking and feedback.  

---

## 🧑‍🏫 Guide

**Dr. Amol Bhopale**  
Indian Institute of Information Technology, Nagpur  

---

## 👨‍💻 Team Members

| Name | Roll No. |
|------|-----------|
| Vivek Yadav | BT23CSA035 |
| Ojaswa Awasthi | BT23CSA060 |

---

## 🪪 License

This project is created for academic and research use under the **IIIT Nagpur AI Project 2025** guidelines.  
Datasets are derived from open research and public nutrition databases.

---

> 💡 *“AI can transform nutrition guidance — from static calorie tracking to intelligent, adaptive wellness.”*
