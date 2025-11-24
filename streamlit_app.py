"""Streamlit demo for restaurant recommendation system."""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from typing import List, Dict, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.data_loader import RestaurantDataLoader
from models.content_based import ContentBasedRecommender
from models.collaborative_filtering import (
    CollaborativeFilteringRecommender, 
    MatrixFactorizationRecommender, 
    ALSRecommender
)
from models.hybrid import HybridRecommender


@st.cache_data
def load_data():
    """Load restaurant data and models."""
    # Load data
    data_loader = RestaurantDataLoader("data/raw")
    restaurants_df, interactions_df, users_df = data_loader.load_data()
    data_loader.preprocess_data()
    
    # Load encoders
    data_loader.load_encoders("models")
    
    return restaurants_df, interactions_df, users_df, data_loader


@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    models_dir = "models"
    
    # Load content-based model
    try:
        content_model = ContentBasedRecommender()
        content_model.load_model(os.path.join(models_dir, "content_based_model.pkl"))
        models["content_based"] = content_model
    except FileNotFoundError:
        st.warning("Content-based model not found. Please train models first.")
    
    # Load collaborative filtering model
    try:
        collab_model = CollaborativeFilteringRecommender()
        # Note: This would need to be implemented in the actual model
        models["collaborative_filtering"] = collab_model
    except FileNotFoundError:
        st.warning("Collaborative filtering model not found.")
    
    # Load matrix factorization model
    try:
        mf_model = MatrixFactorizationRecommender()
        # Note: This would need to be implemented in the actual model
        models["matrix_factorization"] = mf_model
    except FileNotFoundError:
        st.warning("Matrix factorization model not found.")
    
    # Load ALS model
    try:
        als_model = ALSRecommender()
        # Note: This would need to be implemented in the actual model
        models["als"] = als_model
    except FileNotFoundError:
        st.warning("ALS model not found.")
    
    # Load hybrid model
    try:
        hybrid_model = HybridRecommender()
        # Note: This would need to be implemented in the actual model
        models["hybrid"] = hybrid_model
    except FileNotFoundError:
        st.warning("Hybrid model not found.")
    
    return models


def get_user_preferences_from_form() -> Dict[str, str]:
    """Get user preferences from the form."""
    return {
        "preferred_cuisine": st.session_state.get("preferred_cuisine", "Italian"),
        "preferred_price_range": st.session_state.get("preferred_price_range", "$$"),
        "preferred_neighborhood": st.session_state.get("preferred_neighborhood", "Downtown"),
        "preferred_ambiance": st.session_state.get("preferred_ambiance", "Casual")
    }


def display_recommendations(recommendations: List[Dict], model_name: str):
    """Display recommendations in a nice format."""
    st.subheader(f"Recommendations from {model_name}")
    
    if not recommendations:
        st.warning("No recommendations available.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{i}. {rec['name']}**")
                st.write(f"📍 {rec['neighborhood']} | 🍽️ {rec['cuisine']} | 💰 {rec['price_range']} | ⭐ {rec['rating']}")
                
                if 'similarity_score' in rec:
                    st.write(f"Similarity Score: {rec['similarity_score']:.3f}")
                elif 'score' in rec:
                    st.write(f"Recommendation Score: {rec['score']:.3f}")
                elif 'combined_score' in rec:
                    st.write(f"Combined Score: {rec['combined_score']:.3f}")
            
            with col2:
                if st.button(f"View Details", key=f"details_{model_name}_{i}"):
                    st.write("Restaurant Details:")
                    st.write(f"- Restaurant ID: {rec['restaurant_id']}")
                    st.write(f"- Name: {rec['name']}")
                    st.write(f"- Cuisine: {rec['cuisine']}")
                    st.write(f"- Price Range: {rec['price_range']}")
                    st.write(f"- Neighborhood: {rec['neighborhood']}")
                    st.write(f"- Rating: {rec['rating']}")
            
            st.divider()


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Restaurant Recommendation System",
        page_icon="🍽️",
        layout="wide"
    )
    
    st.title("🍽️ Restaurant Recommendation System")
    st.markdown("Discover your next favorite restaurant with AI-powered recommendations!")
    
    # Load data
    try:
        restaurants_df, interactions_df, users_df, data_loader = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please make sure you have run the data generation script first.")
        return
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("No trained models found. Please train models first using the training script.")
        return
    
    # Sidebar for user preferences
    st.sidebar.header("Your Preferences")
    
    # Get unique values for dropdowns
    cuisines = sorted(restaurants_df["cuisine"].unique())
    price_ranges = sorted(restaurants_df["price_range"].unique())
    neighborhoods = sorted(restaurants_df["neighborhood"].unique())
    ambiances = sorted(restaurants_df["ambiance"].unique())
    
    # User preference form
    with st.sidebar.form("user_preferences"):
        st.write("Tell us about your preferences:")
        
        preferred_cuisine = st.selectbox("Preferred Cuisine", cuisines, key="preferred_cuisine")
        preferred_price_range = st.selectbox("Preferred Price Range", price_ranges, key="preferred_price_range")
        preferred_neighborhood = st.selectbox("Preferred Neighborhood", neighborhoods, key="preferred_neighborhood")
        preferred_ambiance = st.selectbox("Preferred Ambiance", ambiances, key="preferred_ambiance")
        
        submitted = st.form_submit_button("Get Recommendations")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Recommendations", "Restaurant Search", "Model Comparison", "About"])
    
    with tab1:
        st.header("Personalized Recommendations")
        
        if submitted:
            user_preferences = {
                "preferred_cuisine": preferred_cuisine,
                "preferred_price_range": preferred_price_range,
                "preferred_neighborhood": preferred_neighborhood,
                "preferred_ambiance": preferred_ambiance
            }
            
            # Get recommendations from different models
            for model_name, model in models.items():
                try:
                    if model_name == "content_based":
                        recommendations = model.recommend_for_user(user_preferences, top_k=5)
                    else:
                        # For collaborative filtering models, we'd need a user_id
                        # For demo purposes, we'll use a random user
                        random_user = np.random.choice(users_df["user_id"].tolist())
                        recommendations = model.recommend(random_user, top_k=5)
                    
                    # Add restaurant details
                    detailed_recs = []
                    for rec in recommendations:
                        restaurant_id = rec["restaurant_id"]
                        restaurant_info = restaurants_df[
                            restaurants_df["restaurant_id"] == restaurant_id
                        ].iloc[0]
                        
                        detailed_rec = {
                            "restaurant_id": restaurant_id,
                            "name": restaurant_info["name"],
                            "cuisine": restaurant_info["cuisine"],
                            "price_range": restaurant_info["price_range"],
                            "neighborhood": restaurant_info["neighborhood"],
                            "rating": restaurant_info["rating"],
                            **rec
                        }
                        detailed_recs.append(detailed_rec)
                    
                    display_recommendations(detailed_recs, model_name.replace("_", " ").title())
                    
                except Exception as e:
                    st.error(f"Error getting recommendations from {model_name}: {e}")
    
    with tab2:
        st.header("Restaurant Search")
        
        # Search by restaurant name
        search_term = st.text_input("Search for a restaurant:", placeholder="Enter restaurant name...")
        
        if search_term:
            # Filter restaurants by name
            filtered_restaurants = restaurants_df[
                restaurants_df["name"].str.contains(search_term, case=False, na=False)
            ]
            
            if len(filtered_restaurants) > 0:
                st.write(f"Found {len(filtered_restaurants)} restaurants matching '{search_term}':")
                
                for _, restaurant in filtered_restaurants.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**{restaurant['name']}**")
                            st.write(f"📍 {restaurant['neighborhood']} | 🍽️ {restaurant['cuisine']} | 💰 {restaurant['price_range']} | ⭐ {restaurant['rating']}")
                        
                        with col2:
                            if st.button(f"Find Similar", key=f"similar_{restaurant['restaurant_id']}"):
                                # Get similar restaurants using content-based model
                                if "content_based" in models:
                                    similar_recs = models["content_based"].recommend(
                                        restaurant["restaurant_id"], top_k=5
                                    )
                                    
                                    st.write("Similar restaurants:")
                                    for rec in similar_recs:
                                        st.write(f"- {rec['name']} (similarity: {rec['similarity_score']:.3f})")
            
            else:
                st.warning(f"No restaurants found matching '{search_term}'")
    
    with tab3:
        st.header("Model Comparison")
        
        # Load evaluation results if available
        try:
            results_df = pd.read_csv("models/evaluation_results.csv")
            st.subheader("Model Performance Comparison")
            st.dataframe(results_df, use_container_width=True)
            
            # Plot metrics
            if st.button("Show Performance Charts"):
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                metrics = ["Precision@10", "Recall@10", "NDCG@10", "MAP@10"]
                
                for i, metric in enumerate(metrics):
                    ax = axes[i//2, i%2]
                    sns.barplot(data=results_df, x="Model", y=metric, ax=ax)
                    ax.set_title(f"{metric} Comparison")
                    ax.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        except FileNotFoundError:
            st.info("No evaluation results found. Please run the training script to generate model comparisons.")
    
    with tab4:
        st.header("About This System")
        
        st.markdown("""
        ## Restaurant Recommendation System
        
        This system uses multiple recommendation approaches to suggest restaurants:
        
        ### Models Used:
        - **Content-Based Filtering**: Recommends restaurants based on features like cuisine, price range, and ambiance
        - **Collaborative Filtering**: Uses user interaction patterns to find similar users and recommend restaurants
        - **Matrix Factorization**: Decomposes the user-restaurant interaction matrix into latent factors
        - **ALS (Alternating Least Squares)**: Advanced matrix factorization technique
        - **Hybrid**: Combines content-based and collaborative filtering approaches
        
        ### Features:
        - Personalized recommendations based on user preferences
        - Restaurant search and similarity finding
        - Model performance comparison
        - Interactive web interface
        
        ### Data:
        The system uses synthetic restaurant data including:
        - Restaurant information (name, cuisine, price range, location, rating)
        - User interactions (ratings, views, likes)
        - User profiles (age group, dietary restrictions)
        
        ### Evaluation Metrics:
        - Precision@K: Accuracy of top-K recommendations
        - Recall@K: Coverage of relevant items in top-K
        - NDCG@K: Normalized Discounted Cumulative Gain
        - MAP@K: Mean Average Precision
        - Hit Rate@K: Whether at least one relevant item is in top-K
        """)


if __name__ == "__main__":
    main()
