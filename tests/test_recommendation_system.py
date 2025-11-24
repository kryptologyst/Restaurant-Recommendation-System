"""Unit tests for restaurant recommendation system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data.generate_data import generate_restaurant_data, generate_user_interactions, set_seed
from data.data_loader import RestaurantDataLoader
from models.content_based import ContentBasedRecommender
from models.collaborative_filtering import CollaborativeFilteringRecommender, MatrixFactorizationRecommender
from models.hybrid import HybridRecommender
from evaluation.metrics import RecommendationEvaluator


class TestDataGeneration:
    """Test data generation functions."""
    
    def test_set_seed(self):
        """Test seed setting function."""
        set_seed(42)
        # Test that random numbers are reproducible
        np.random.seed(42)
        first_random = np.random.random()
        
        set_seed(42)
        np.random.seed(42)
        second_random = np.random.random()
        
        assert first_random == second_random
    
    def test_generate_restaurant_data(self):
        """Test restaurant data generation."""
        restaurants_df = generate_restaurant_data(n_restaurants=10, seed=42)
        
        assert len(restaurants_df) == 10
        assert "restaurant_id" in restaurants_df.columns
        assert "name" in restaurants_df.columns
        assert "cuisine" in restaurants_df.columns
        assert "price_range" in restaurants_df.columns
        assert "neighborhood" in restaurants_df.columns
        assert "rating" in restaurants_df.columns
        
        # Check that all restaurant IDs are unique
        assert len(restaurants_df["restaurant_id"].unique()) == 10
        
        # Check rating range
        assert restaurants_df["rating"].min() >= 1.0
        assert restaurants_df["rating"].max() <= 5.0
    
    def test_generate_user_interactions(self):
        """Test user interaction generation."""
        restaurants_df = generate_restaurant_data(n_restaurants=10, seed=42)
        interactions_df = generate_user_interactions(
            restaurants_df, n_users=5, n_interactions=20, seed=42
        )
        
        assert len(interactions_df) <= 20  # May be less due to probability filtering
        assert "user_id" in interactions_df.columns
        assert "restaurant_id" in interactions_df.columns
        assert "rating" in interactions_df.columns
        assert "timestamp" in interactions_df.columns
        
        # Check rating range
        assert interactions_df["rating"].min() >= 1
        assert interactions_df["rating"].max() <= 5


class TestDataLoader:
    """Test data loading and preprocessing."""
    
    def test_restaurant_data_loader_init(self):
        """Test data loader initialization."""
        loader = RestaurantDataLoader("data/raw")
        assert loader.data_dir == "data/raw"
        assert loader.restaurants_df is None
        assert loader.interactions_df is None
        assert loader.users_df is None
    
    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        """Test data loading."""
        # Mock CSV files
        mock_restaurants = pd.DataFrame({
            "restaurant_id": ["r1", "r2"],
            "name": ["Restaurant 1", "Restaurant 2"],
            "cuisine": ["Italian", "Chinese"],
            "price_range": ["$$", "$$$"],
            "neighborhood": ["Downtown", "Midtown"],
            "ambiance": ["Casual", "Fine Dining"],
            "description": ["Good food", "Great food"],
            "rating": [4.0, 4.5],
            "latitude": [40.7, 40.8],
            "longitude": [-74.0, -73.9]
        })
        
        mock_interactions = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "restaurant_id": ["r1", "r2"],
            "rating": [4, 5],
            "timestamp": ["2023-01-01", "2023-01-02"],
            "interaction_type": ["review", "like"]
        })
        
        mock_users = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "age_group": ["18-25", "26-35"],
            "dietary_restrictions": ["None", "Vegetarian"],
            "signup_date": ["2023-01-01", "2023-01-02"]
        })
        
        mock_read_csv.side_effect = [mock_restaurants, mock_interactions, mock_users]
        
        loader = RestaurantDataLoader("data/raw")
        restaurants_df, interactions_df, users_df = loader.load_data()
        
        assert len(restaurants_df) == 2
        assert len(interactions_df) == 2
        assert len(users_df) == 2
        
        # Check timestamp conversion
        assert pd.api.types.is_datetime64_any_dtype(interactions_df["timestamp"])


class TestContentBasedRecommender:
    """Test content-based recommendation model."""
    
    def test_content_based_recommender_init(self):
        """Test content-based recommender initialization."""
        model = ContentBasedRecommender(random_state=42)
        assert model.random_state == 42
        assert model.restaurant_features is None
        assert model.restaurant_similarity_matrix is None
    
    def test_fit(self):
        """Test model fitting."""
        restaurants_df = pd.DataFrame({
            "restaurant_id": ["r1", "r2", "r3"],
            "name": ["Italian Place", "Chinese Restaurant", "Japanese Sushi"],
            "cuisine": ["Italian", "Chinese", "Japanese"],
            "price_range": ["$$", "$$", "$$$"],
            "neighborhood": ["Downtown", "Midtown", "Uptown"],
            "ambiance": ["Casual", "Family-friendly", "Fine Dining"],
            "description": ["Good Italian food", "Authentic Chinese", "Fresh sushi"],
            "rating": [4.0, 4.2, 4.5],
            "latitude": [40.7, 40.8, 40.9],
            "longitude": [-74.0, -73.9, -73.8]
        })
        
        model = ContentBasedRecommender(random_state=42)
        model.fit(restaurants_df)
        
        assert model.restaurant_features is not None
        assert model.restaurant_similarity_matrix is not None
        assert model.restaurants_df is not None
        
        # Check similarity matrix shape
        assert model.restaurant_similarity_matrix.shape == (3, 3)
    
    def test_recommend(self):
        """Test restaurant recommendation."""
        restaurants_df = pd.DataFrame({
            "restaurant_id": ["r1", "r2", "r3"],
            "name": ["Italian Place", "Chinese Restaurant", "Japanese Sushi"],
            "cuisine": ["Italian", "Chinese", "Japanese"],
            "price_range": ["$$", "$$", "$$$"],
            "neighborhood": ["Downtown", "Midtown", "Uptown"],
            "ambiance": ["Casual", "Family-friendly", "Fine Dining"],
            "description": ["Good Italian food", "Authentic Chinese", "Fresh sushi"],
            "rating": [4.0, 4.2, 4.5],
            "latitude": [40.7, 40.8, 40.9],
            "longitude": [-74.0, -73.9, -73.8]
        })
        
        model = ContentBasedRecommender(random_state=42)
        model.fit(restaurants_df)
        
        recommendations = model.recommend("r1", top_k=2)
        
        assert len(recommendations) <= 2
        assert all("restaurant_id" in rec for rec in recommendations)
        assert all("similarity_score" in rec for rec in recommendations)
    
    def test_recommend_for_user(self):
        """Test user-based recommendations."""
        restaurants_df = pd.DataFrame({
            "restaurant_id": ["r1", "r2", "r3"],
            "name": ["Italian Place", "Chinese Restaurant", "Japanese Sushi"],
            "cuisine": ["Italian", "Chinese", "Japanese"],
            "price_range": ["$$", "$$", "$$$"],
            "neighborhood": ["Downtown", "Midtown", "Uptown"],
            "ambiance": ["Casual", "Family-friendly", "Fine Dining"],
            "description": ["Good Italian food", "Authentic Chinese", "Fresh sushi"],
            "rating": [4.0, 4.2, 4.5],
            "latitude": [40.7, 40.8, 40.9],
            "longitude": [-74.0, -73.9, -73.8]
        })
        
        model = ContentBasedRecommender(random_state=42)
        model.fit(restaurants_df)
        
        user_preferences = {
            "preferred_cuisine": "Italian",
            "preferred_price_range": "$$",
            "preferred_neighborhood": "Downtown",
            "preferred_ambiance": "Casual"
        }
        
        recommendations = model.recommend_for_user(user_preferences, top_k=2)
        
        assert len(recommendations) <= 2
        assert all("restaurant_id" in rec for rec in recommendations)
        assert all("similarity_score" in rec for rec in recommendations)


class TestCollaborativeFilteringRecommender:
    """Test collaborative filtering recommendation model."""
    
    def test_collaborative_filtering_recommender_init(self):
        """Test collaborative filtering recommender initialization."""
        model = CollaborativeFilteringRecommender(random_state=42)
        assert model.random_state == 42
        assert model.user_similarity_matrix is None
        assert model.item_similarity_matrix is None
    
    def test_fit(self):
        """Test model fitting."""
        # Create mock interaction matrix
        interaction_matrix = np.array([
            [4, 0, 5, 0],
            [0, 3, 0, 4],
            [5, 0, 4, 0]
        ])
        
        # Create mock encoders
        user_encoder = Mock()
        user_encoder.classes_ = ["u1", "u2", "u3"]
        restaurant_encoder = Mock()
        restaurant_encoder.classes_ = ["r1", "r2", "r3", "r4"]
        
        model = CollaborativeFilteringRecommender(random_state=42)
        model.fit(interaction_matrix, user_encoder, restaurant_encoder)
        
        assert model.user_similarity_matrix is not None
        assert model.item_similarity_matrix is not None
        assert model.interaction_matrix is not None
        
        # Check similarity matrix shapes
        assert model.user_similarity_matrix.shape == (3, 3)
        assert model.item_similarity_matrix.shape == (4, 4)


class TestMatrixFactorizationRecommender:
    """Test matrix factorization recommendation model."""
    
    def test_matrix_factorization_recommender_init(self):
        """Test matrix factorization recommender initialization."""
        model = MatrixFactorizationRecommender(n_components=10, random_state=42)
        assert model.n_components == 10
        assert model.random_state == 42
        assert model.user_factors is None
        assert model.item_factors is None
    
    def test_fit(self):
        """Test model fitting."""
        # Create mock interaction matrix
        interaction_matrix = np.array([
            [4, 0, 5, 0],
            [0, 3, 0, 4],
            [5, 0, 4, 0]
        ])
        
        # Create mock encoders
        user_encoder = Mock()
        user_encoder.classes_ = ["u1", "u2", "u3"]
        restaurant_encoder = Mock()
        restaurant_encoder.classes_ = ["r1", "r2", "r3", "r4"]
        
        model = MatrixFactorizationRecommender(n_components=2, random_state=42)
        model.fit(interaction_matrix, user_encoder, restaurant_encoder)
        
        assert model.user_factors is not None
        assert model.item_factors is not None
        
        # Check factor matrix shapes
        assert model.user_factors.shape == (3, 2)
        assert model.item_factors.shape == (4, 2)


class TestRecommendationEvaluator:
    """Test recommendation evaluation metrics."""
    
    def test_evaluator_init(self):
        """Test evaluator initialization."""
        evaluator = RecommendationEvaluator()
        assert evaluator.metrics_history == {}
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        evaluator = RecommendationEvaluator()
        
        recommendations = ["r1", "r2", "r3", "r4", "r5"]
        relevant_items = ["r1", "r3", "r5"]
        
        precision_3 = evaluator.precision_at_k(recommendations, relevant_items, 3)
        precision_5 = evaluator.precision_at_k(recommendations, relevant_items, 5)
        
        assert precision_3 == 2/3  # r1, r3 are in top 3
        assert precision_5 == 3/5  # r1, r3, r5 are in top 5
    
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        evaluator = RecommendationEvaluator()
        
        recommendations = ["r1", "r2", "r3", "r4", "r5"]
        relevant_items = ["r1", "r3", "r5"]
        
        recall_3 = evaluator.recall_at_k(recommendations, relevant_items, 3)
        recall_5 = evaluator.recall_at_k(recommendations, relevant_items, 5)
        
        assert recall_3 == 2/3  # 2 out of 3 relevant items found
        assert recall_5 == 3/3  # All 3 relevant items found
    
    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        evaluator = RecommendationEvaluator()
        
        recommendations = ["r1", "r2", "r3", "r4", "r5"]
        relevant_items = ["r1", "r3", "r5"]
        
        ndcg_3 = evaluator.ndcg_at_k(recommendations, relevant_items, 3)
        ndcg_5 = evaluator.ndcg_at_k(recommendations, relevant_items, 5)
        
        assert 0 <= ndcg_3 <= 1
        assert 0 <= ndcg_5 <= 1
        assert ndcg_5 >= ndcg_3  # NDCG should be higher for larger k
    
    def test_hit_rate_at_k(self):
        """Test Hit Rate@K calculation."""
        evaluator = RecommendationEvaluator()
        
        recommendations = ["r1", "r2", "r3", "r4", "r5"]
        relevant_items = ["r1", "r3", "r5"]
        
        hit_rate_2 = evaluator.hit_rate_at_k(recommendations, relevant_items, 2)
        hit_rate_5 = evaluator.hit_rate_at_k(recommendations, relevant_items, 5)
        
        assert hit_rate_2 == 1.0  # r1 is in top 2
        assert hit_rate_5 == 1.0  # r1, r3, r5 are in top 5
        
        # Test case with no hits
        no_hit_recommendations = ["r6", "r7", "r8"]
        hit_rate_no_hit = evaluator.hit_rate_at_k(no_hit_recommendations, relevant_items, 3)
        assert hit_rate_no_hit == 0.0


class TestHybridRecommender:
    """Test hybrid recommendation model."""
    
    def test_hybrid_recommender_init(self):
        """Test hybrid recommender initialization."""
        model = HybridRecommender(content_weight=0.3, collaborative_weight=0.7, random_state=42)
        assert model.content_weight == 0.3
        assert model.collaborative_weight == 0.7
        assert model.random_state == 42
        assert model.content_model is not None
        assert model.collaborative_model is None


if __name__ == "__main__":
    pytest.main([__file__])
