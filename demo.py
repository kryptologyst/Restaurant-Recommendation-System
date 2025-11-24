#!/usr/bin/env python3
"""Simple demo script to showcase the restaurant recommendation system."""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data.data_loader import RestaurantDataLoader
from models.content_based import ContentBasedRecommender


def main():
    """Run a simple demo of the recommendation system."""
    print("🍽️ Restaurant Recommendation System Demo")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading data...")
    data_loader = RestaurantDataLoader("data/raw")
    restaurants_df, interactions_df, users_df = data_loader.load_data()
    data_loader.preprocess_data()
    
    print(f"   ✓ Loaded {len(restaurants_df)} restaurants")
    print(f"   ✓ Loaded {len(interactions_df)} interactions")
    print(f"   ✓ Loaded {len(users_df)} users")
    
    # Train content-based model
    print("\n2. Training content-based recommendation model...")
    model = ContentBasedRecommender(random_state=42)
    model.fit(restaurants_df)
    print("   ✓ Model trained successfully")
    
    # Show some restaurant examples
    print("\n3. Sample restaurants in our database:")
    sample_restaurants = restaurants_df.sample(5)
    for _, restaurant in sample_restaurants.iterrows():
        print(f"   • {restaurant['name']} ({restaurant['cuisine']}) - {restaurant['price_range']} - {restaurant['neighborhood']}")
    
    # Demo recommendations
    print("\n4. Getting recommendations for different user preferences:")
    
    # Demo 1: Italian food lover
    print("\n   👤 User who loves Italian food:")
    italian_prefs = {
        "preferred_cuisine": "Italian",
        "preferred_price_range": "$$",
        "preferred_neighborhood": "Downtown",
        "preferred_ambiance": "Casual"
    }
    
    italian_recs = model.recommend_for_user(italian_prefs, top_k=3)
    for i, rec in enumerate(italian_recs, 1):
        print(f"      {i}. {rec['name']} - {rec['cuisine']} - Similarity: {rec['similarity_score']:.3f}")
    
    # Demo 2: Fine dining enthusiast
    print("\n   👤 User who prefers fine dining:")
    fine_dining_prefs = {
        "preferred_cuisine": "French",
        "preferred_price_range": "$$$$",
        "preferred_neighborhood": "Manhattan",
        "preferred_ambiance": "Fine Dining"
    }
    
    fine_dining_recs = model.recommend_for_user(fine_dining_prefs, top_k=3)
    for i, rec in enumerate(fine_dining_recs, 1):
        print(f"      {i}. {rec['name']} - {rec['cuisine']} - Similarity: {rec['similarity_score']:.3f}")
    
    # Demo 3: Similar restaurant recommendations
    print("\n5. Finding similar restaurants:")
    sample_restaurant = restaurants_df.iloc[0]
    print(f"   🔍 Finding restaurants similar to '{sample_restaurant['name']}'")
    
    similar_recs = model.recommend(sample_restaurant["restaurant_id"], top_k=3)
    for i, rec in enumerate(similar_recs, 1):
        print(f"      {i}. {rec['name']} - {rec['cuisine']} - Similarity: {rec['similarity_score']:.3f}")
    
    # Show data statistics
    print("\n6. Dataset Statistics:")
    print(f"   • Most popular cuisine: {restaurants_df['cuisine'].value_counts().index[0]}")
    print(f"   • Average rating: {restaurants_df['rating'].mean():.2f}")
    print(f"   • Price range distribution:")
    price_counts = restaurants_df['price_range'].value_counts()
    for price, count in price_counts.items():
        print(f"     - {price}: {count} restaurants")
    
    print(f"   • Neighborhood distribution:")
    neighborhood_counts = restaurants_df['neighborhood'].value_counts().head(5)
    for neighborhood, count in neighborhood_counts.items():
        print(f"     - {neighborhood}: {count} restaurants")
    
    print("\n✅ Demo completed successfully!")
    print("\nTo explore more features:")
    print("   • Run 'streamlit run streamlit_app.py' for the interactive web demo")
    print("   • Check 'models/evaluation_results.csv' for model performance metrics")
    print("   • Run 'python scripts/train_models.py' to retrain all models")


if __name__ == "__main__":
    main()
