"""Hybrid recommendation models combining multiple approaches."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle
import os

from .content_based import ContentBasedRecommender
from .collaborative_filtering import CollaborativeFilteringRecommender, MatrixFactorizationRecommender, ALSRecommender


class HybridRecommender:
    """Hybrid recommendation system combining content-based and collaborative filtering."""
    
    def __init__(
        self, 
        content_weight: float = 0.3,
        collaborative_weight: float = 0.7,
        random_state: int = 42
    ):
        """Initialize hybrid recommender.
        
        Args:
            content_weight: Weight for content-based recommendations
            collaborative_weight: Weight for collaborative filtering recommendations
            random_state: Random seed for reproducibility
        """
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.random_state = random_state
        
        self.content_model = ContentBasedRecommender(random_state=random_state)
        self.collaborative_model = None
        self.restaurants_df = None
        self.user_encoder = None
        self.restaurant_encoder = None
        
    def fit(
        self, 
        restaurants_df: pd.DataFrame,
        interaction_matrix: np.ndarray,
        user_encoder,
        restaurant_encoder,
        collaborative_type: str = "matrix_factorization"
    ) -> None:
        """Fit the hybrid recommender.
        
        Args:
            restaurants_df: DataFrame with restaurant information
            interaction_matrix: User-restaurant interaction matrix
            user_encoder: Label encoder for users
            restaurant_encoder: Label encoder for restaurants
            collaborative_type: Type of collaborative filtering model
        """
        self.restaurants_df = restaurants_df
        self.user_encoder = user_encoder
        self.restaurant_encoder = restaurant_encoder
        
        # Fit content-based model
        self.content_model.fit(restaurants_df)
        
        # Fit collaborative filtering model
        if collaborative_type == "matrix_factorization":
            self.collaborative_model = MatrixFactorizationRecommender(random_state=self.random_state)
        elif collaborative_type == "als":
            self.collaborative_model = ALSRecommender(random_state=self.random_state)
        else:
            self.collaborative_model = CollaborativeFilteringRecommender(random_state=self.random_state)
        
        self.collaborative_model.fit(interaction_matrix, user_encoder, restaurant_encoder)
        
    def recommend(
        self, 
        user_id: str, 
        top_k: int = 10,
        user_preferences: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """Recommend restaurants using hybrid approach.
        
        Args:
            user_id: ID of the user
            top_k: Number of recommendations
            user_preferences: Optional user preferences for content-based filtering
            
        Returns:
            List of recommended restaurants
        """
        # Get collaborative filtering recommendations
        if hasattr(self.collaborative_model, 'recommend'):
            collab_recs = self.collaborative_model.recommend(user_id, top_k)
        else:
            collab_recs = self.collaborative_model.recommend_user_based(user_id, top_k)
        
        # Get content-based recommendations
        if user_preferences:
            content_recs = self.content_model.recommend_for_user(user_preferences, top_k)
        else:
            # Use user's interaction history to infer preferences
            user_preferences = self._infer_user_preferences(user_id)
            content_recs = self.content_model.recommend_for_user(user_preferences, top_k)
        
        # Combine recommendations
        combined_recs = self._combine_recommendations(
            collab_recs, 
            content_recs, 
            top_k
        )
        
        # Add restaurant details
        return self._add_restaurant_details(combined_recs)
    
    def _infer_user_preferences(self, user_id: str) -> Dict[str, str]:
        """Infer user preferences from interaction history.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with inferred preferences
        """
        # This is a simplified version - in practice, you'd analyze the user's interaction history
        return {
            "preferred_cuisine": "Italian",
            "preferred_price_range": "$$",
            "preferred_neighborhood": "Downtown",
            "preferred_ambiance": "Casual"
        }
    
    def _combine_recommendations(
        self, 
        collab_recs: List[Dict], 
        content_recs: List[Dict], 
        top_k: int
    ) -> List[Dict]:
        """Combine collaborative and content-based recommendations.
        
        Args:
            collab_recs: Collaborative filtering recommendations
            content_recs: Content-based recommendations
            top_k: Number of final recommendations
            
        Returns:
            Combined recommendations
        """
        # Create score dictionaries
        collab_scores = {rec["restaurant_id"]: rec["score"] for rec in collab_recs}
        content_scores = {rec["restaurant_id"]: rec["similarity_score"] for rec in content_recs}
        
        # Get all unique restaurant IDs
        all_restaurant_ids = set(collab_scores.keys()) | set(content_scores.keys())
        
        # Combine scores
        combined_scores = {}
        for restaurant_id in all_restaurant_ids:
            collab_score = collab_scores.get(restaurant_id, 0)
            content_score = content_scores.get(restaurant_id, 0)
            
            # Normalize scores
            collab_score_norm = collab_score / max(collab_scores.values()) if collab_scores.values() else 0
            content_score_norm = content_score / max(content_scores.values()) if content_scores.values() else 0
            
            # Weighted combination
            combined_score = (
                self.collaborative_weight * collab_score_norm +
                self.content_weight * content_score_norm
            )
            
            combined_scores[restaurant_id] = combined_score
        
        # Sort by combined score
        sorted_restaurants = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top-k recommendations
        return [
            {
                "restaurant_id": restaurant_id,
                "combined_score": score,
                "collaborative_score": collab_scores.get(restaurant_id, 0),
                "content_score": content_scores.get(restaurant_id, 0)
            }
            for restaurant_id, score in sorted_restaurants[:top_k]
        ]
    
    def _add_restaurant_details(self, recommendations: List[Dict]) -> List[Dict]:
        """Add restaurant details to recommendations.
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            Recommendations with restaurant details
        """
        detailed_recs = []
        for rec in recommendations:
            restaurant_id = rec["restaurant_id"]
            restaurant_info = self.restaurants_df[
                self.restaurants_df["restaurant_id"] == restaurant_id
            ].iloc[0]
            
            detailed_rec = {
                "restaurant_id": restaurant_id,
                "name": restaurant_info["name"],
                "cuisine": restaurant_info["cuisine"],
                "price_range": restaurant_info["price_range"],
                "neighborhood": restaurant_info["neighborhood"],
                "rating": restaurant_info["rating"],
                "combined_score": rec["combined_score"],
                "collaborative_score": rec["collaborative_score"],
                "content_score": rec["content_score"]
            }
            
            detailed_recs.append(detailed_rec)
        
        return detailed_recs


class EnsembleRecommender:
    """Ensemble recommendation system using multiple models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize ensemble recommender.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.weights = {}
        self.restaurants_df = None
        
    def add_model(self, name: str, model: object, weight: float = 1.0) -> None:
        """Add a model to the ensemble.
        
        Args:
            name: Name of the model
            model: Trained recommendation model
            weight: Weight for this model in the ensemble
        """
        self.models[name] = model
        self.weights[name] = weight
        
    def fit(
        self, 
        restaurants_df: pd.DataFrame,
        interaction_matrix: np.ndarray,
        user_encoder,
        restaurant_encoder
    ) -> None:
        """Fit all models in the ensemble.
        
        Args:
            restaurants_df: DataFrame with restaurant information
            interaction_matrix: User-restaurant interaction matrix
            user_encoder: Label encoder for users
            restaurant_encoder: Label encoder for restaurants
        """
        self.restaurants_df = restaurants_df
        
        # Fit each model
        for name, model in self.models.items():
            if hasattr(model, 'fit'):
                if name == "content_based":
                    model.fit(restaurants_df)
                else:
                    model.fit(interaction_matrix, user_encoder, restaurant_encoder)
    
    def recommend(
        self, 
        user_id: str, 
        top_k: int = 10,
        user_preferences: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """Recommend restaurants using ensemble approach.
        
        Args:
            user_id: ID of the user
            top_k: Number of recommendations
            user_preferences: Optional user preferences
            
        Returns:
            List of recommended restaurants
        """
        all_recommendations = {}
        
        # Get recommendations from each model
        for name, model in self.models.items():
            try:
                if name == "content_based" and user_preferences:
                    recs = model.recommend_for_user(user_preferences, top_k)
                    # Convert to standard format
                    recs = [
                        {
                            "restaurant_id": rec["restaurant_id"],
                            "score": rec["similarity_score"]
                        }
                        for rec in recs
                    ]
                else:
                    recs = model.recommend(user_id, top_k)
                
                all_recommendations[name] = recs
            except Exception as e:
                print(f"Error getting recommendations from {name}: {e}")
                all_recommendations[name] = []
        
        # Combine recommendations
        combined_scores = {}
        for model_name, recs in all_recommendations.items():
            weight = self.weights[model_name]
            for rec in recs:
                restaurant_id = rec["restaurant_id"]
                score = rec["score"]
                
                if restaurant_id not in combined_scores:
                    combined_scores[restaurant_id] = 0
                
                combined_scores[restaurant_id] += weight * score
        
        # Sort by combined score
        sorted_restaurants = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top-k recommendations
        return [
            {
                "restaurant_id": restaurant_id,
                "ensemble_score": score
            }
            for restaurant_id, score in sorted_restaurants[:top_k]
        ]
    
    def save_ensemble(self, save_path: str) -> None:
        """Save the ensemble model.
        
        Args:
            save_path: Path to save the model
        """
        ensemble_data = {
            "models": self.models,
            "weights": self.weights,
            "restaurants_df": self.restaurants_df,
            "random_state": self.random_state
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(ensemble_data, f)
    
    def load_ensemble(self, load_path: str) -> None:
        """Load the ensemble model.
        
        Args:
            load_path: Path to load the model from
        """
        with open(load_path, "rb") as f:
            ensemble_data = pickle.load(f)
        
        self.models = ensemble_data["models"]
        self.weights = ensemble_data["weights"]
        self.restaurants_df = ensemble_data["restaurants_df"]
        self.random_state = ensemble_data["random_state"]
