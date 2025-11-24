"""Data generation and loading utilities for restaurant recommendation system."""

import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def generate_restaurant_data(n_restaurants: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic restaurant data.
    
    Args:
        n_restaurants: Number of restaurants to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with restaurant information
    """
    set_seed(seed)
    
    cuisines = [
        "Italian", "Chinese", "Japanese", "Mexican", "Indian", "French", 
        "Thai", "Korean", "Mediterranean", "American", "Vietnamese", "Greek",
        "Spanish", "German", "Brazilian", "Lebanese", "Turkish", "Ethiopian"
    ]
    
    price_ranges = ["$", "$$", "$$$", "$$$$"]
    
    neighborhoods = [
        "Downtown", "Midtown", "Brooklyn", "Queens", "Manhattan", "Bronx",
        "Williamsburg", "Astoria", "Park Slope", "Chelsea", "SoHo", "Tribeca"
    ]
    
    ambiances = [
        "Casual", "Fine Dining", "Family-friendly", "Romantic", "Trendy", 
        "Cozy", "Upscale", "Fast-casual", "Outdoor seating", "Bar atmosphere"
    ]
    
    restaurants = []
    for i in range(n_restaurants):
        cuisine = random.choice(cuisines)
        price = random.choice(price_ranges)
        neighborhood = random.choice(neighborhoods)
        ambiance = random.choice(ambiances)
        
        # Generate realistic restaurant name
        if cuisine == "Italian":
            name = f"{random.choice(['Bella', 'Mario', 'Luigi', 'Nonna'])} {random.choice(['Trattoria', 'Ristorante', 'Pizzeria', 'Bistro'])}"
        elif cuisine == "Chinese":
            name = f"{random.choice(['Golden', 'Dragon', 'Lucky', 'Imperial'])} {random.choice(['Garden', 'Palace', 'House', 'Kitchen'])}"
        elif cuisine == "Japanese":
            name = f"{random.choice(['Sakura', 'Tokyo', 'Zen', 'Bamboo'])} {random.choice(['Sushi', 'Ramen', 'Garden', 'House'])}"
        else:
            name = f"{random.choice(['The', 'Cafe', 'Bistro', 'Kitchen'])} {cuisine} {random.choice(['House', 'Corner', 'Spot', 'Place'])}"
        
        # Generate description
        description = f"{cuisine} restaurant in {neighborhood} with {ambiance.lower()} atmosphere. {price} price range."
        
        restaurants.append({
            "restaurant_id": f"rest_{i:04d}",
            "name": name,
            "cuisine": cuisine,
            "price_range": price,
            "neighborhood": neighborhood,
            "ambiance": ambiance,
            "description": description,
            "rating": round(np.random.normal(4.0, 0.8), 1),
            "latitude": np.random.uniform(40.7, 40.8),
            "longitude": np.random.uniform(-74.0, -73.9)
        })
    
    return pd.DataFrame(restaurants)


def generate_user_interactions(
    restaurants_df: pd.DataFrame, 
    n_users: int = 500, 
    n_interactions: int = 5000,
    seed: int = 42
) -> pd.DataFrame:
    """Generate user interaction data.
    
    Args:
        restaurants_df: Restaurant data
        n_users: Number of users
        n_interactions: Number of interactions to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with user interactions
    """
    set_seed(seed)
    
    interactions = []
    start_date = datetime.now() - timedelta(days=365)
    
    # Generate user preferences
    user_preferences = {}
    for user_id in range(n_users):
        user_preferences[f"user_{user_id:04d}"] = {
            "preferred_cuisines": random.sample(restaurants_df["cuisine"].unique().tolist(), k=random.randint(2, 5)),
            "preferred_price": random.choice(restaurants_df["price_range"].unique().tolist()),
            "preferred_neighborhoods": random.sample(restaurants_df["neighborhood"].unique().tolist(), k=random.randint(1, 3))
        }
    
    # Generate interactions based on preferences
    for _ in range(n_interactions):
        user_id = f"user_{random.randint(0, n_users-1):04d}"
        restaurant = restaurants_df.sample(1).iloc[0]
        
        # Calculate interaction probability based on preferences
        prob = 0.1  # Base probability
        
        if restaurant["cuisine"] in user_preferences[user_id]["preferred_cuisines"]:
            prob += 0.3
        if restaurant["price_range"] == user_preferences[user_id]["preferred_price"]:
            prob += 0.2
        if restaurant["neighborhood"] in user_preferences[user_id]["preferred_neighborhoods"]:
            prob += 0.2
        
        # Add some randomness
        prob += np.random.normal(0, 0.1)
        prob = max(0.01, min(0.9, prob))
        
        if random.random() < prob:
            # Generate timestamp
            timestamp = start_date + timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Generate rating based on restaurant rating with some noise
            base_rating = restaurant["rating"]
            user_rating = max(1, min(5, int(round(base_rating + np.random.normal(0, 0.5)))))
            
            interactions.append({
                "user_id": user_id,
                "restaurant_id": restaurant["restaurant_id"],
                "rating": user_rating,
                "timestamp": timestamp,
                "interaction_type": random.choice(["view", "like", "bookmark", "review"])
            })
    
    return pd.DataFrame(interactions)


def generate_user_profiles(n_users: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate user profile data.
    
    Args:
        n_users: Number of users
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with user profiles
    """
    set_seed(seed)
    
    age_groups = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    dietary_restrictions = ["None", "Vegetarian", "Vegan", "Gluten-free", "Halal", "Kosher"]
    
    users = []
    for i in range(n_users):
        users.append({
            "user_id": f"user_{i:04d}",
            "age_group": random.choice(age_groups),
            "dietary_restrictions": random.choice(dietary_restrictions),
            "signup_date": datetime.now() - timedelta(days=random.randint(30, 1000))
        })
    
    return pd.DataFrame(users)


def create_sample_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a complete sample dataset for the restaurant recommendation system.
    
    Returns:
        Tuple of (restaurants_df, interactions_df, users_df)
    """
    print("Generating restaurant data...")
    restaurants_df = generate_restaurant_data(n_restaurants=1000)
    
    print("Generating user interactions...")
    interactions_df = generate_user_interactions(restaurants_df, n_users=500, n_interactions=5000)
    
    print("Generating user profiles...")
    users_df = generate_user_profiles(n_users=500)
    
    return restaurants_df, interactions_df, users_df


if __name__ == "__main__":
    # Generate and save sample data
    restaurants_df, interactions_df, users_df = create_sample_dataset()
    
    # Save to CSV files
    restaurants_df.to_csv("data/raw/restaurants.csv", index=False)
    interactions_df.to_csv("data/raw/interactions.csv", index=False)
    users_df.to_csv("data/raw/users.csv", index=False)
    
    print(f"Generated {len(restaurants_df)} restaurants")
    print(f"Generated {len(interactions_df)} interactions")
    print(f"Generated {len(users_df)} users")
    print("Data saved to data/raw/ directory")
