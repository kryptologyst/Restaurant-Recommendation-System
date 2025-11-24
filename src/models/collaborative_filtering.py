"""Collaborative filtering recommendation models."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import implicit
import pickle
import os


class CollaborativeFilteringRecommender:
    """Collaborative filtering recommendation system."""
    
    def __init__(self, random_state: int = 42):
        """Initialize collaborative filtering recommender.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.interaction_matrix = None
        self.user_encoder = None
        self.restaurant_encoder = None
        
    def fit(self, interaction_matrix: np.ndarray, user_encoder, restaurant_encoder) -> None:
        """Fit the collaborative filtering recommender.
        
        Args:
            interaction_matrix: User-restaurant interaction matrix
            user_encoder: Label encoder for users
            restaurant_encoder: Label encoder for restaurants
        """
        self.interaction_matrix = interaction_matrix
        self.user_encoder = user_encoder
        self.restaurant_encoder = restaurant_encoder
        
        # Compute user similarity matrix
        self.user_similarity_matrix = cosine_similarity(interaction_matrix)
        
        # Compute item similarity matrix
        self.item_similarity_matrix = cosine_similarity(interaction_matrix.T)
        
    def recommend_user_based(
        self, 
        user_id: str, 
        top_k: int = 10,
        n_neighbors: int = 50
    ) -> List[Dict]:
        """Recommend restaurants using user-based collaborative filtering.
        
        Args:
            user_id: ID of the user
            top_k: Number of recommendations
            n_neighbors: Number of similar users to consider
            
        Returns:
            List of recommended restaurants
        """
        if self.user_similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user index
        user_idx = self.user_encoder.transform([user_id])[0]
        
        # Get user's interactions
        user_interactions = self.interaction_matrix[user_idx]
        
        # Find similar users
        user_similarities = self.user_similarity_matrix[user_idx]
        similar_users = np.argsort(user_similarities)[-n_neighbors-1:-1][::-1]
        
        # Calculate recommendation scores
        scores = np.zeros(self.interaction_matrix.shape[1])
        
        for similar_user_idx in similar_users:
            similarity = user_similarities[similar_user_idx]
            similar_user_interactions = self.interaction_matrix[similar_user_idx]
            
            # Add weighted interactions
            scores += similarity * similar_user_interactions
        
        # Remove already interacted restaurants
        scores[user_interactions > 0] = 0
        
        # Get top recommendations
        top_restaurant_indices = np.argsort(scores)[-top_k:][::-1]
        
        recommendations = []
        for restaurant_idx in top_restaurant_indices:
            if scores[restaurant_idx] > 0:
                restaurant_id = self.restaurant_encoder.inverse_transform([restaurant_idx])[0]
                recommendations.append({
                    "restaurant_id": restaurant_id,
                    "score": scores[restaurant_idx]
                })
        
        return recommendations
    
    def recommend_item_based(
        self, 
        user_id: str, 
        top_k: int = 10,
        n_neighbors: int = 50
    ) -> List[Dict]:
        """Recommend restaurants using item-based collaborative filtering.
        
        Args:
            user_id: ID of the user
            top_k: Number of recommendations
            n_neighbors: Number of similar restaurants to consider
            
        Returns:
            List of recommended restaurants
        """
        if self.item_similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user index
        user_idx = self.user_encoder.transform([user_id])[0]
        
        # Get user's interactions
        user_interactions = self.interaction_matrix[user_idx]
        
        # Calculate recommendation scores
        scores = np.zeros(self.interaction_matrix.shape[1])
        
        for restaurant_idx in range(self.interaction_matrix.shape[1]):
            if user_interactions[restaurant_idx] > 0:  # User has interacted with this restaurant
                # Find similar restaurants
                restaurant_similarities = self.item_similarity_matrix[restaurant_idx]
                similar_restaurants = np.argsort(restaurant_similarities)[-n_neighbors-1:-1][::-1]
                
                # Add weighted interactions
                for similar_restaurant_idx in similar_restaurants:
                    similarity = restaurant_similarities[similar_restaurant_idx]
                    scores[similar_restaurant_idx] += similarity * user_interactions[restaurant_idx]
        
        # Remove already interacted restaurants
        scores[user_interactions > 0] = 0
        
        # Get top recommendations
        top_restaurant_indices = np.argsort(scores)[-top_k:][::-1]
        
        recommendations = []
        for restaurant_idx in top_restaurant_indices:
            if scores[restaurant_idx] > 0:
                restaurant_id = self.restaurant_encoder.inverse_transform([restaurant_idx])[0]
                recommendations.append({
                    "restaurant_id": restaurant_id,
                    "score": scores[restaurant_idx]
                })
        
        return recommendations


class MatrixFactorizationRecommender:
    """Matrix factorization recommendation system using SVD."""
    
    def __init__(self, n_components: int = 50, random_state: int = 42):
        """Initialize matrix factorization recommender.
        
        Args:
            n_components: Number of latent factors
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.user_factors = None
        self.item_factors = None
        self.user_encoder = None
        self.restaurant_encoder = None
        
    def fit(self, interaction_matrix: np.ndarray, user_encoder, restaurant_encoder) -> None:
        """Fit the matrix factorization model.
        
        Args:
            interaction_matrix: User-restaurant interaction matrix
            user_encoder: Label encoder for users
            restaurant_encoder: Label encoder for restaurants
        """
        self.user_encoder = user_encoder
        self.restaurant_encoder = restaurant_encoder
        
        # Fit SVD
        self.svd.fit(interaction_matrix)
        
        # Get factor matrices
        self.user_factors = self.svd.transform(interaction_matrix)
        self.item_factors = self.svd.components_.T
        
    def recommend(
        self, 
        user_id: str, 
        top_k: int = 10
    ) -> List[Dict]:
        """Recommend restaurants using matrix factorization.
        
        Args:
            user_id: ID of the user
            top_k: Number of recommendations
            
        Returns:
            List of recommended restaurants
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user index
        user_idx = self.user_encoder.transform([user_id])[0]
        
        # Get user's interactions
        user_interactions = self.interaction_matrix[user_idx]
        
        # Calculate recommendation scores
        user_vector = self.user_factors[user_idx]
        scores = np.dot(self.item_factors, user_vector)
        
        # Remove already interacted restaurants
        scores[user_interactions > 0] = 0
        
        # Get top recommendations
        top_restaurant_indices = np.argsort(scores)[-top_k:][::-1]
        
        recommendations = []
        for restaurant_idx in top_restaurant_indices:
            if scores[restaurant_idx] > 0:
                restaurant_id = self.restaurant_encoder.inverse_transform([restaurant_idx])[0]
                recommendations.append({
                    "restaurant_id": restaurant_id,
                    "score": scores[restaurant_idx]
                })
        
        return recommendations


class ALSRecommender:
    """Alternating Least Squares recommendation system using implicit library."""
    
    def __init__(self, factors: int = 50, iterations: int = 15, random_state: int = 42):
        """Initialize ALS recommender.
        
        Args:
            factors: Number of latent factors
            iterations: Number of iterations
            random_state: Random seed for reproducibility
        """
        self.factors = factors
        self.iterations = iterations
        self.random_state = random_state
        
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            random_state=random_state
        )
        self.user_encoder = None
        self.restaurant_encoder = None
        
    def fit(self, interaction_matrix: np.ndarray, user_encoder, restaurant_encoder) -> None:
        """Fit the ALS model.
        
        Args:
            interaction_matrix: User-restaurant interaction matrix
            user_encoder: Label encoder for users
            restaurant_encoder: Label encoder for restaurants
        """
        self.user_encoder = user_encoder
        self.restaurant_encoder = restaurant_encoder
        
        # Convert to CSR format for implicit library
        from scipy.sparse import csr_matrix
        sparse_matrix = csr_matrix(interaction_matrix)
        
        # Fit the model
        self.model.fit(sparse_matrix)
        
    def recommend(
        self, 
        user_id: str, 
        top_k: int = 10
    ) -> List[Dict]:
        """Recommend restaurants using ALS.
        
        Args:
            user_id: ID of the user
            top_k: Number of recommendations
            
        Returns:
            List of recommended restaurants
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user index
        user_idx = self.user_encoder.transform([user_id])[0]
        
        # Get recommendations
        recommendations = self.model.recommend(
            user_idx, 
            self.model.user_factors, 
            N=top_k
        )
        
        result = []
        for restaurant_idx, score in recommendations:
            restaurant_id = self.restaurant_encoder.inverse_transform([restaurant_idx])[0]
            result.append({
                "restaurant_id": restaurant_id,
                "score": score
            })
        
        return result
