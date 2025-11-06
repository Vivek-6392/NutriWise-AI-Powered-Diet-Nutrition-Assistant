import streamlit as st
import pandas as pd
import re
from datetime import datetime

# Import your advanced processor and LLM engine classes
from llm_integration import NutritionLLMIntegration, NutritionQuery, NutritionDataProcessor  # Adjust import if needed

# =====================
# App Configuration
# =====================
st.set_page_config(
    page_title="AI Nutrition Assistant",
    page_icon="🥗",
    layout="wide"
)

st.title("AI-Powered Diet & Nutrition Assistant")
st.markdown("Get **personalized nutrition analysis, meal plans, and educational insights** powered by NLP & ML.")

# =====================
# Load Dataset using Processor (CRITICAL FIX)
# =====================
@st.cache_data
def load_data_with_processor():
    excel_path = "Anuvaad_INDB_2024.11.xlsx"
    processor = NutritionDataProcessor(use_spacy=True)
    df = processor.load_nutrition_data([excel_path])
    return processor, df

processor, nutrition_data = load_data_with_processor()

# =====================
# Initialize LLM (CRITICAL FIX)
# =====================
st.sidebar.header("⚙️ LLM Settings")
use_lmstudio = st.sidebar.checkbox("Use LM Studio (local)", value=True)
openai_api_key = st.sidebar.text_input("OpenAI API Key (if not using LM Studio)", type="password")

# LM Studio config
lmstudio_url = "http://localhost:1234/v1/chat/completions"
lmstudio_model = "openchat-3.6-8b-20240522"

llm_engine = NutritionLLMIntegration(
    nutrition_data,
    processor,
    api_key=openai_api_key if not use_lmstudio else None,
    use_lmstudio=use_lmstudio,
    lmstudio_url=lmstudio_url,
    lmstudio_model=lmstudio_model
)

# =====================
# Sidebar
# =====================
st.sidebar.subheader("User Settings")
user_id = st.sidebar.text_input("Enter User ID:", "guest_123")
query_type = st.sidebar.selectbox(
    "Select Query Type:",
    ["recommendation", "analysis", "education", "meal_planning"]
)
st.sidebar.markdown("---")
st.sidebar.info("💡 If no API key is set, and LM Studio is off, the app falls back to **rule-based answers**.")

# =====================
# Tabs for Features
# =====================
tabs = st.tabs(["🔍 Food Analysis", "🍴 Meal Planning", "📚 Nutrition Education", "💡 Recommendations"])

# ---- Food Analysis ----
with tabs[0]:
    st.subheader("🔍 Food & Nutrition Analysis")
    query_text = st.text_area("Enter a food item or meal description:")

    if st.button("Analyze Food", key="analyze"):
        if query_text.strip():
            query = NutritionQuery(
                user_id=user_id,
                query=query_text,
                context={"goal": "analyze food nutrients"},
                timestamp=datetime.now(),
                query_type="analysis"
            )
            response = llm_engine.handle_query(query)
            st.success("✅ Nutrition Analysis:")
            st.write(response)

            if "food_name" in nutrition_data.columns:
                def normalize(text):
                    if not isinstance(text, str):
                        text = str(text)
                    text = text.strip().lower()
                    text = re.sub(r'[^\w\s]', '', text)
                    text = re.sub(r'\s+', ' ', text)
                    return text

                normalized_query = normalize(query_text)
                if "food_name_norm" not in nutrition_data.columns:
                    nutrition_data['food_name_norm'] = nutrition_data['food_name'].apply(normalize)

                # Show all foods whose normalized name is found inside normalized query
                mask = nutrition_data['food_name_norm'].apply(lambda fn: fn in normalized_query)
                matched = nutrition_data[mask]

                if not matched.empty:
                    st.dataframe(matched.drop(columns='food_name_norm'))
                else:
                    st.info("No relevant match found in dataset.")
            else:
                st.error("No food_name column found in dataset. Please check if the loaded file is correct.")

# ---- Meal Planning ----
with tabs[1]:
    st.subheader("🍴 Personalized Meal Planning")
    goal = st.selectbox("Choose your goal:", ["Weight Loss", "Muscle Gain", "Balanced Diet", "Diabetic Friendly"])
    restrictions = st.text_input("Any dietary restrictions? (e.g. vegan, gluten-free)")
    if st.button("Generate Meal Plan", key="mealplan"):
        query = NutritionQuery(
            user_id=user_id,
            query=f"Create a meal plan for {goal}, restrictions: {restrictions}",
            context={"goal": goal, "restrictions": restrictions},
            timestamp=datetime.now(),
            query_type="meal_planning"
        )
        response = llm_engine.handle_query(query)
        st.success("✅ Meal Plan Suggestion:")
        st.write(response)

# ---- Education ----
with tabs[2]:
    st.subheader("📚 Nutrition Education")
    edu_question = st.text_area("Ask a nutrition-related question (e.g. What is protein’s role?)")
    if st.button("Ask Educator", key="educator"):
        query = NutritionQuery(
            user_id=user_id,
            query=edu_question,
            context={},
            timestamp=datetime.now(),
            query_type="education"
        )
        response = llm_engine.handle_query(query)
        st.success("📘 Nutrition Education:")
        st.write(response)

# ---- Recommendations ----
with tabs[3]:
    st.subheader("💡 Personalized Recommendations")
    health_goal = st.text_input("What’s your health goal? (e.g. more energy, better skin, fat loss)")
    if st.button("Get Recommendations", key="recommend"):
        query = NutritionQuery(
            user_id=user_id,
            query=f"Recommend foods for {health_goal}",
            context={"goal": health_goal},
            timestamp=datetime.now(),
            query_type="recommendation"
        )
        response = llm_engine.handle_query(query)
        st.success("✅ Personalized Recommendation:")
        st.write(response)

# =====================
# Footer
# =====================
st.markdown("---")
st.markdown("🔬 Built with **Streamlit, NLP, ML/DL, and LLMs** (LM Studio / OpenAI) for personalized nutrition insights.")
