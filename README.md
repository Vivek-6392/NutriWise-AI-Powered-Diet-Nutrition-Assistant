# 🧠 NLP for Diet and Nutrition Analysis

**Team Members:**  
Vivek Yadav [BT23CSA035] · Jayant Dhaka [BT23CSA045] · Ojaswa Awasthi [BT23CSA060]  
Sandesh Charhate [BT23CSA062] · Yashwant Chauhan [BT23CSA042]  
**Guide:** Dr. Amol Bhopale  
**Institute:** Indian Institute of Information Technology, Nagpur  

---

## 🩺 Introduction

### 🎯 Objective
To develop a **personalized AI diet and nutrition assistant** that analyzes user dietary habits and health data, providing **tailored recommendations** for improved well-being.

### 💡 Motivation
With the rise of health consciousness and diverse diet needs, there’s a gap in accessible, personalized nutrition guidance.  
This project bridges that gap using **Natural Language Processing (NLP)** and **Machine Learning (ML)** to deliver intelligent, adaptive diet insights.

### 🔍 Core Idea
Leverage NLP and Large Language Models to interpret user input, analyze foods, and dynamically generate:
- Personalized meal plans  
- Nutritional breakdowns  
- Educational explanations in natural language  

---

## 🚨 The Nutrition Problem We’re Solving

| Problem | Description |
|----------|--------------|
| Rising Lifestyle Diseases | Obesity and diabetes are increasing globally due to poor nutrition awareness. |
| Rigid, Rule-Based Apps | Most diet apps offer static, one-size-fits-all recommendations. |
| Complex Nutrition Data | Users struggle to understand food labels and nutrient information. |
| Limited Personalization | Existing tools rarely adapt to health conditions or cultural diets. |

---

## 🎯 Project Objectives

1. **Automated Food Analysis** – Extract and compute nutrient profiles using NLP and nutrition databases.  
2. **Personalized Meal Planning** – Generate adaptive meal plans based on health goals, allergies, and preferences.  
3. **Nutritional Education** – Explain nutrition principles clearly using LLMs.  
4. **Health-Based Recommendations** – Suggest suitable foods for diabetes, obesity, and heart conditions.  

---

## 🧠 Literature Review

### Part 1: Variational Autoencoder (VAE) Model
1. **User Input:** Health profile → normalized input vector.  
2. **Encoder:** Maps data to latent parameters (μ, σ).  
3. **Sampling:** Latent vector *z* represents personalized dietary space.  
4. **Decoder:** GRU/LSTM generates meal sequences.  
5. **Output:** Predicts nutrients, calories, and meal categories.  

---

### Part 2: Dataset and Models

**Datasets Used**
- **University of Toronto FLIP 2017 / 2020**
- **Health Canada TRA & FSANZ systems**
- **Indian Nutrition Database (Anuvaad_INDB_2024.11)**

**Techniques & Models**
| Method | Description |
|---------|-------------|
| Sentence-BERT (SBERT) | Encodes text labels into dense semantic vectors |
| t-SNE | Visualizes clustering of food categories |
| XGBoost | Tree-boosted regression/classification for high accuracy |
| KNN / Elastic Net | For similarity-based and mixed regularized learning |
| LM Studio + Local LLMs | Powers natural language reasoning and contextual explanations |

**Performance Highlights**
| Task | Model | Metric | Score |
|------|--------|---------|--------|
| Food Category Classification | XGBoost + SBERT | Accuracy | **0.98** |
| Subcategory Prediction | XGBoost + SBERT | Accuracy | **0.96** |
| FSANZ Nutrition Score (Structured) | Regression | R² = **0.98**, MSE = **2.5** |

---

## 🧩 System Architecture

### 🔹 Core Components

1. **NLP Engine**
   - Performs intent recognition and entity extraction (food, quantity, preference).
   - Uses SBERT for semantic similarity and text encoding.

2. **Nutrition Database**
   - Source: *Anuvaad_INDB_2024.11*  
   - Contains macronutrient and micronutrient data for Indian foods.  

   Example Entries:
   | Food | Serving | Energy | Protein | Fat | Carbs |
   |-------|----------|---------|----------|------|-------|
   | Hot Tea | 100ml | 1 kcal | 0.1g | 0g | 0.2g |
   | Mango Drink | 200ml | 100 kcal | 0.5g | 0.1g | 25g |

3. **Machine Learning Recommender**
   - Algorithms: RandomForest, GradientBoosting, KNN, PCA Clustering.  
   - Learns user patterns and refines meal suggestions dynamically.

4. **LLM Integration via LM Studio**
   - Local LLMs (e.g., Llama, Mistral, Phi) are run through **LM Studio**, enabling **offline AI reasoning**.  
   - Used for:
     - **Nutritional Q&A**
     - **Meal Plan Explanation**
     - **User Intent Parsing**
     - **Chat-style Recommendations**
   - Integrates seamlessly with the Streamlit front end through REST API or local inference endpoints.

5. **Streamlit Interface**
   - Provides an interactive dashboard for:
     - Uploading diet logs  
     - Viewing personalized insights  
     - Conversing with the AI nutrition assistant  

---

## ⚙️ Tech Stack

| Layer | Tools & Frameworks |
|-------|--------------------|
| **Frontend** | Streamlit |
| **Backend / Logic** | Python |
| **NLP** | spaCy, Sentence-BERT |
| **ML Models** | TensorFlow, PyTorch, XGBoost |
| **Visualization** | Matplotlib, Plotly |
| **LLM Integration** | **LM Studio (Local Model Hosting)** |
| **Database** | Anuvaad_INDB_2024.11 |
| **Deployment** | Streamlit / Localhost Server |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Vivek-6392/NutriWise-AI-Powered-Diet-Nutrition-Assistant.git
cd NutriWise-AI-Powered-Diet-Nutrition-Assistant
```
