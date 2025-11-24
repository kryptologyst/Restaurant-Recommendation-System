"""Content-based recommendation models for restaurant recommendations."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import os


class ContentBasedRecommender:
    """Content-based recommendation system using restaurant features."""
    
    def __init__(self, random_state: int = 42):
        """Initialize content-based recommender.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()
        self.restaurant_features = None
        self.restaurant_similarity_matrix = None
        self.restaurants_df = None
        
    def fit(self, restaurants_df: pd.DataFrame) -> None:
        """Fit the content-based recommender.
        
        Args:
            restaurants_df: DataFrame with restaurant information
        """
        self.restaurants_df = restaurants_df.copy()
        
        # Create text features from restaurant descriptions
        text_features = self._create_text_features(restaurants_df)
        
        # Create categorical features
        categorical_features = self._create_categorical_features(restaurants_df)
        
        # Combine features
        self.restaurant_features = np.hstack([text_features, categorical_features])
        
        # Compute similarity matrix
        self.restaurant_similarity_matrix = cosine_similarity(self.restaurant_features)
        
    def _create_text_features(self, restaurants_df: pd.DataFrame) -> np.ndarray:
        """Create TF-IDF features from restaurant text.
        
        Args:
            restaurants_df: DataFrame with restaurant information
            
        Returns:
            TF-IDF feature matrix
        """
        # Combine text fields
        text_data = (
            restaurants_df["name"] + " " +
            restaurants_df["description"] + " " +
            restaurants_df["cuisine"] + " " +
            restaurants_df["ambiance"]
        )
        
        return self.tfidf_vectorizer.fit_transform(text_data).toarray()
    
    def _create_categorical_features(self, restaurants_df: pd.DataFrame) -> np.ndarray:
        """Create categorical features.
        
        Args:
            restaurants_df: DataFrame with restaurant information
            
        Returns:
            Categorical feature matrix
        """
        # One-hot encode categorical variables
        cuisine_dummies = pd.get_dummies(restaurants_df["cuisine"], prefix="cuisine")
        price_dummies = pd.get_dummies(restaurants_df["price_range"], prefix="price")
        neighborhood_dummies = pd.get_dummies(restaurants_df["neighborhood"], prefix="neighborhood")
        ambiance_dummies = pd.get_dummies(restaurants_df["ambiance"], prefix="ambiance")
        
        categorical_features = pd.concat([
            cuisine_dummies,
            price_dummies,
            neighborhood_dummies,
            ambiance_dummies
        ], axis=1)
        
        # Add numerical features
        numerical_features = restaurants_df[["rating", "latitude", "longitude"]].values
        
        return np.hstack([categorical_features.values, numerical_features])
    
    def recommend(
        self, 
        restaurant_id: str, 
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Dict]:
        """Recommend similar restaurants.
        
        Args:
            restaurant_id: ID of the reference restaurant
            top_k: Number of recommendations
            exclude_self: Whether to exclude the reference restaurant
            
        Returns:
            List of recommended restaurants with similarity scores
        """
        if self.restaurant_similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Find restaurant index
        restaurant_idx = self.restaurants_df[
            self.restaurants_df["restaurant_id"] == restaurant_id
        ].index[0]
        
        # Get similarity scores
        similarity_scores = self.restaurant_similarity_matrix[restaurant_idx]
        
        # Get top similar restaurants
        if exclude_self:
            similarity_scores[restaurant_idx] = -1
        
        top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if similarity_scores[idx] > 0:  # Only include positive similarities
                restaurant = self.restaurants_df.iloc[idx]
                recommendations.append({
                    "restaurant_id": restaurant["restaurant_id"],
                    "name": restaurant["name"],
                    "cuisine": restaurant["cuisine"],
                    "price_range": restaurant["price_range"],
                    "neighborhood": restaurant["neighborhood"],
                    "rating": restaurant["rating"],
                    "similarity_score": similarity_scores[idx]
                })
        
        return recommendations
    
    def recommend_for_user(
        self, 
        user_preferences: Dict[str, str], 
        top_k: int = 10
    ) -> List[Dict]:
        """Recommend restaurants based on user preferences.
        
        Args:
            user_preferences: Dictionary with user preferences
            top_k: Number of recommendations
            
        Returns:
            List of recommended restaurants
        """
        if self.restaurant_similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create a virtual restaurant based on user preferences
        virtual_restaurant = self._create_virtual_restaurant(user_preferences)
        
        # Find most similar restaurants
        similarities = cosine_similarity(
            virtual_restaurant.reshape(1, -1),
            self.restaurant_features
        )[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        recommendations = []
        for idx in top_indices:
            restaurant = self.restaurants_df.iloc[idx]
            recommendations.append({
                "restaurant_id": restaurant["restaurant_id"],
                "name": restaurant["name"],
                "cuisine": restaurant["cuisine"],
                "price_range": restaurant["price_range"],
                "neighborhood": restaurant["neighborhood"],
                "rating": restaurant["rating"],
                "similarity_score": similarities[idx]
            })
        
        return recommendations
    
    def _create_virtual_restaurant(self, user_preferences: Dict[str, str]) -> np.ndarray:
        """Create a virtual restaurant based on user preferences.
        
        Args:
            user_preferences: Dictionary with user preferences
            
        Returns:
            Feature vector for virtual restaurant
        """
        # Create text description
        text_desc = f"{user_preferences.get('preferred_cuisine', '')} restaurant in {user_preferences.get('preferred_neighborhood', '')} with {user_preferences.get('preferred_ambiance', 'casual')} atmosphere"
        
        # Transform text
        text_features = self.tfidf_vectorizer.transform([text_desc]).toarray()
        
        # Create categorical features
        cuisine_dummies = pd.get_dummies([user_preferences.get('preferred_cuisine', '')], prefix="cuisine")
        price_dummies = pd.get_dummies([user_preferences.get('preferred_price_range', '$$')], prefix="price")
        neighborhood_dummies = pd.get_dummies([user_preferences.get('preferred_neighborhood', 'Downtown')], prefix="neighborhood")
        ambiance_dummies = pd.get_dummies([user_preferences.get('preferred_ambiance', 'Casual')], prefix="ambiance")
        
        # Ensure all columns match training data
        all_cuisines = [f"cuisine_{c}" for c in self.restaurants_df["cuisine"].unique()]
        all_prices = [f"price_{p}" for p in self.restaurants_df["price_range"].unique()]
        all_neighborhoods = [f"neighborhood_{n}" for n in self.restaurants_df["neighborhood"].unique()]
        all_ambiances = [f"ambiance_{a}" for a in self.restaurants_df["ambiance"].unique()]
        
        # Create full categorical feature vector
        categorical_features = np.zeros(len(all_cuisines) + len(all_prices) + len(all_neighborhoods) + len(all_ambiances) + 3)
        
        # Set appropriate features
        cuisine_idx = all_cuisines.index(f"cuisine_{user_preferences.get('preferred_cuisine', '')}")
        categorical_features[cuisine_idx] = 1
        
        price_idx = all_prices.index(f"price_{user_preferences.get('preferred_price_range', '$$')}")
        categorical_features[len(all_cuisines) + price_idx] = 1
        
        neighborhood_idx = all_neighborhoods.index(f"neighborhood_{user_preferences.get('preferred_neighborhood', 'Downtown')}")
        categorical_features[len(all_cuisines) + len(all_prices) + neighborhood_idx] = 1
        
        ambiance_idx = all_ambiances.index(f"ambiance_{user_preferences.get('preferred_ambiance', 'Casual')}")
        categorical_features[len(all_cuisines) + len(all_prices) + len(all_neighborhoods) + ambiance_idx] = 1
        
        # Add numerical features (default values)
        categorical_features[-3:] = [4.0, 40.75, -73.95]  # rating, lat, lng
        
        return np.hstack([text_features[0], categorical_features])
    
    def save_model(self, save_path: str) -> None:
        """Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        model_data = {
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "scaler": self.scaler,
            "restaurant_features": self.restaurant_features,
            "restaurant_similarity_matrix": self.restaurant_similarity_matrix,
            "restaurants_df": self.restaurants_df,
            "random_state": self.random_state
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)
    
    def load_model(self, load_path: str) -> None:
        """Load a trained model.
        
        Args:
            load_path: Path to load the model from
        """
        with open(load_path, "rb") as f:
            model_data = pickle.load(f)
        
        self.tfidf_vectorizer = model_data["tfidf_vectorizer"]
        self.scaler = model_data["scaler"]
        self.restaurant_features = model_data["restaurant_features"]
        self.restaurant_similarity_matrix = model_data["restaurant_similarity_matrix"]
        self.restaurants_df = model_data["restaurants_df"]
        self.random_state = model_data["random_state"]
