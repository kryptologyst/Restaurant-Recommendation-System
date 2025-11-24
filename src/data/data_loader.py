"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os


class RestaurantDataLoader:
    """Data loader for restaurant recommendation system."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = data_dir
        self.restaurants_df: Optional[pd.DataFrame] = None
        self.interactions_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        
        # Encoders for categorical variables
        self.user_encoder = LabelEncoder()
        self.restaurant_encoder = LabelEncoder()
        self.cuisine_encoder = LabelEncoder()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all data files.
        
        Returns:
            Tuple of (restaurants_df, interactions_df, users_df)
        """
        restaurants_path = os.path.join(self.data_dir, "restaurants.csv")
        interactions_path = os.path.join(self.data_dir, "interactions.csv")
        users_path = os.path.join(self.data_dir, "users.csv")
        
        if not all(os.path.exists(p) for p in [restaurants_path, interactions_path, users_path]):
            raise FileNotFoundError("Data files not found. Please run generate_data.py first.")
        
        self.restaurants_df = pd.read_csv(restaurants_path)
        self.interactions_df = pd.read_csv(interactions_path)
        self.users_df = pd.read_csv(users_path)
        
        # Convert timestamp to datetime
        self.interactions_df["timestamp"] = pd.to_datetime(self.interactions_df["timestamp"])
        
        return self.restaurants_df, self.interactions_df, self.users_df
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Preprocess the loaded data.
        
        Returns:
            Tuple of preprocessed dataframes
        """
        if self.restaurants_df is None or self.interactions_df is None or self.users_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Encode categorical variables
        self.interactions_df["user_id_encoded"] = self.user_encoder.fit_transform(self.interactions_df["user_id"])
        self.interactions_df["restaurant_id_encoded"] = self.restaurant_encoder.fit_transform(self.interactions_df["restaurant_id"])
        
        # Add cuisine encoding to restaurants
        self.restaurants_df["cuisine_encoded"] = self.cuisine_encoder.fit_transform(self.restaurants_df["cuisine"])
        
        # Create user-restaurant interaction matrix
        self.interaction_matrix = self._create_interaction_matrix()
        
        return self.restaurants_df, self.interactions_df, self.users_df
    
    def _create_interaction_matrix(self) -> np.ndarray:
        """Create user-restaurant interaction matrix.
        
        Returns:
            Interaction matrix of shape (n_users, n_restaurants)
        """
        n_users = len(self.user_encoder.classes_)
        n_restaurants = len(self.restaurant_encoder.classes_)
        
        matrix = np.zeros((n_users, n_restaurants))
        
        for _, row in self.interactions_df.iterrows():
            user_idx = row["user_id_encoded"]
            restaurant_idx = row["restaurant_id_encoded"]
            rating = row["rating"]
            matrix[user_idx, restaurant_idx] = rating
        
        return matrix
    
    def get_train_test_split(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split interactions into train and test sets.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_matrix, test_matrix, train_interactions, test_interactions)
        """
        if self.interactions_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Split interactions
        # Remove stratification to avoid issues with users having only one interaction
        train_interactions, test_interactions = train_test_split(
            self.interactions_df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Create matrices
        n_users = len(self.user_encoder.classes_)
        n_restaurants = len(self.restaurant_encoder.classes_)
        
        train_matrix = np.zeros((n_users, n_restaurants))
        test_matrix = np.zeros((n_users, n_restaurants))
        
        for _, row in train_interactions.iterrows():
            user_idx = row["user_id_encoded"]
            restaurant_idx = row["restaurant_id_encoded"]
            rating = row["rating"]
            train_matrix[user_idx, restaurant_idx] = rating
        
        for _, row in test_interactions.iterrows():
            user_idx = row["user_id_encoded"]
            restaurant_idx = row["restaurant_id_encoded"]
            rating = row["rating"]
            test_matrix[user_idx, restaurant_idx] = rating
        
        return train_matrix, test_matrix, train_interactions, test_interactions
    
    def get_user_features(self) -> pd.DataFrame:
        """Get user features for content-based filtering.
        
        Returns:
            DataFrame with user features
        """
        if self.users_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create user preference features based on interaction history
        user_preferences = []
        
        for user_id in self.users_df["user_id"]:
            user_interactions = self.interactions_df[self.interactions_df["user_id"] == user_id]
            
            if len(user_interactions) > 0:
                # Get preferred cuisines
                restaurant_ids = user_interactions["restaurant_id"].tolist()
                cuisines = self.restaurants_df[self.restaurants_df["restaurant_id"].isin(restaurant_ids)]["cuisine"].tolist()
                cuisine_counts = pd.Series(cuisines).value_counts()
                
                # Get preferred price ranges
                price_ranges = self.restaurants_df[self.restaurants_df["restaurant_id"].isin(restaurant_ids)]["price_range"].tolist()
                price_counts = pd.Series(price_ranges).value_counts()
                
                # Get preferred neighborhoods
                neighborhoods = self.restaurants_df[self.restaurants_df["restaurant_id"].isin(restaurant_ids)]["neighborhood"].tolist()
                neighborhood_counts = pd.Series(neighborhoods).value_counts()
                
                user_preferences.append({
                    "user_id": user_id,
                    "preferred_cuisine": cuisine_counts.index[0] if len(cuisine_counts) > 0 else "Unknown",
                    "preferred_price_range": price_counts.index[0] if len(price_counts) > 0 else "$$",
                    "preferred_neighborhood": neighborhood_counts.index[0] if len(neighborhood_counts) > 0 else "Downtown",
                    "avg_rating_given": user_interactions["rating"].mean(),
                    "total_interactions": len(user_interactions)
                })
            else:
                user_preferences.append({
                    "user_id": user_id,
                    "preferred_cuisine": "Unknown",
                    "preferred_price_range": "$$",
                    "preferred_neighborhood": "Downtown",
                    "avg_rating_given": 3.0,
                    "total_interactions": 0
                })
        
        return pd.DataFrame(user_preferences)
    
    def save_encoders(self, save_dir: str = "models") -> None:
        """Save encoders for later use.
        
        Args:
            save_dir: Directory to save encoders
        """
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, "user_encoder.pkl"), "wb") as f:
            pickle.dump(self.user_encoder, f)
        
        with open(os.path.join(save_dir, "restaurant_encoder.pkl"), "wb") as f:
            pickle.dump(self.restaurant_encoder, f)
        
        with open(os.path.join(save_dir, "cuisine_encoder.pkl"), "wb") as f:
            pickle.dump(self.cuisine_encoder, f)
    
    def load_encoders(self, load_dir: str = "models") -> None:
        """Load saved encoders.
        
        Args:
            load_dir: Directory containing saved encoders
        """
        with open(os.path.join(load_dir, "user_encoder.pkl"), "rb") as f:
            self.user_encoder = pickle.load(f)
        
        with open(os.path.join(load_dir, "restaurant_encoder.pkl"), "rb") as f:
            self.restaurant_encoder = pickle.load(f)
        
        with open(os.path.join(load_dir, "cuisine_encoder.pkl"), "rb") as f:
            self.cuisine_encoder = pickle.load(f)
