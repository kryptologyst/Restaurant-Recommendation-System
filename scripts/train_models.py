"""Main training script for restaurant recommendation system."""

import os
import sys
import argparse
import yaml
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.data_loader import RestaurantDataLoader
from src.data.generate_data import create_sample_dataset
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative_filtering import (
    CollaborativeFilteringRecommender, 
    MatrixFactorizationRecommender, 
    ALSRecommender
)
from src.models.hybrid import HybridRecommender, EnsembleRecommender
from src.evaluation.metrics import RecommendationEvaluator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_models(
    restaurants_df: pd.DataFrame,
    train_matrix: np.ndarray,
    user_encoder,
    restaurant_encoder,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Train all recommendation models.
    
    Args:
        restaurants_df: Restaurant data
        train_matrix: Training interaction matrix
        user_encoder: User label encoder
        restaurant_encoder: Restaurant label encoder
        config: Configuration dictionary
        
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    print("Training Content-Based Recommender...")
    content_model = ContentBasedRecommender(random_state=config["random_state"])
    content_model.fit(restaurants_df)
    models["content_based"] = content_model
    
    print("Training Collaborative Filtering Recommender...")
    collab_model = CollaborativeFilteringRecommender(random_state=config["random_state"])
    collab_model.fit(train_matrix, user_encoder, restaurant_encoder)
    models["collaborative_filtering"] = collab_model
    
    print("Training Matrix Factorization Recommender...")
    mf_model = MatrixFactorizationRecommender(
        n_components=config["mf_components"],
        random_state=config["random_state"]
    )
    mf_model.fit(train_matrix, user_encoder, restaurant_encoder)
    models["matrix_factorization"] = mf_model
    
    print("Training ALS Recommender...")
    als_model = ALSRecommender(
        factors=config["als_factors"],
        iterations=config["als_iterations"],
        random_state=config["random_state"]
    )
    als_model.fit(train_matrix, user_encoder, restaurant_encoder)
    models["als"] = als_model
    
    print("Training Hybrid Recommender...")
    hybrid_model = HybridRecommender(
        content_weight=config["hybrid_content_weight"],
        collaborative_weight=config["hybrid_collaborative_weight"],
        random_state=config["random_state"]
    )
    hybrid_model.fit(
        restaurants_df, train_matrix, user_encoder, restaurant_encoder,
        collaborative_type="matrix_factorization"
    )
    models["hybrid"] = hybrid_model
    
    print("Training Ensemble Recommender...")
    ensemble_model = EnsembleRecommender(random_state=config["random_state"])
    ensemble_model.add_model("content_based", content_model, weight=0.2)
    ensemble_model.add_model("matrix_factorization", mf_model, weight=0.4)
    ensemble_model.add_model("als", als_model, weight=0.4)
    ensemble_model.fit(restaurants_df, train_matrix, user_encoder, restaurant_encoder)
    models["ensemble"] = ensemble_model
    
    return models


def evaluate_models(
    models: Dict[str, Any],
    test_data: pd.DataFrame,
    restaurants_df: pd.DataFrame,
    user_encoder,
    restaurant_encoder,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """Evaluate all trained models.
    
    Args:
        models: Dictionary of trained models
        test_data: Test interaction data
        restaurants_df: Restaurant data
        user_encoder: User label encoder
        restaurant_encoder: Restaurant label encoder
        config: Configuration dictionary
        
    Returns:
        DataFrame with evaluation results
    """
    evaluator = RecommendationEvaluator()
    
    k_values = config["evaluation"]["k_values"]
    
    print("Evaluating models...")
    results_df = evaluator.compare_models(
        models, test_data, restaurants_df,
        user_encoder, restaurant_encoder, k_values
    )
    
    # Save results
    results_path = os.path.join("models", "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation results saved to {results_path}")
    
    # Plot results
    if config["evaluation"]["plot_results"]:
        plot_dir = "assets/plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        for metric in ["Precision@10", "Recall@10", "NDCG@10"]:
            evaluator.plot_metrics_comparison(
                results_df, metric, 
                save_path=os.path.join(plot_dir, f"{metric.lower()}_comparison.png")
            )
    
    return results_df


def save_models(models: Dict[str, Any], save_dir: str = "models") -> None:
    """Save trained models.
    
    Args:
        models: Dictionary of trained models
        save_dir: Directory to save models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, model in models.items():
        model_path = os.path.join(save_dir, f"{model_name}_model.pkl")
        
        if hasattr(model, 'save_model'):
            model.save_model(model_path)
        elif hasattr(model, 'save_ensemble'):
            model.save_ensemble(model_path)
        else:
            # Generic pickle save
            import pickle
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        
        print(f"Saved {model_name} model to {model_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train restaurant recommendation models")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--generate-data", action="store_true",
                       help="Generate sample data if it doesn't exist")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    np.random.seed(config["random_state"])
    
    # Generate data if requested or if data doesn't exist
    data_dir = "data/raw"
    if args.generate_data or not os.path.exists(data_dir):
        print("Generating sample data...")
        os.makedirs(data_dir, exist_ok=True)
        restaurants_df, interactions_df, users_df = create_sample_dataset()
    else:
        print("Loading existing data...")
        data_loader = RestaurantDataLoader(data_dir)
        restaurants_df, interactions_df, users_df = data_loader.load_data()
        data_loader.preprocess_data()
    
    # Create data loader and get train/test split
    data_loader = RestaurantDataLoader(data_dir)
    restaurants_df, interactions_df, users_df = data_loader.load_data()
    data_loader.preprocess_data()
    
    train_matrix, test_matrix, train_interactions, test_interactions = data_loader.get_train_test_split(
        test_size=config["test_size"],
        random_state=config["random_state"]
    )
    
    print(f"Training data shape: {train_matrix.shape}")
    print(f"Test data shape: {test_matrix.shape}")
    print(f"Number of training interactions: {len(train_interactions)}")
    print(f"Number of test interactions: {len(test_interactions)}")
    
    # Train models
    models = train_models(
        restaurants_df, train_matrix,
        data_loader.user_encoder, data_loader.restaurant_encoder,
        config
    )
    
    # Evaluate models
    results_df = evaluate_models(
        models, test_interactions, restaurants_df,
        data_loader.user_encoder, data_loader.restaurant_encoder,
        config
    )
    
    # Print results
    print("\nModel Comparison Results:")
    print("=" * 50)
    print(results_df.to_string(index=False))
    
    # Save models
    save_models(models)
    
    # Save encoders
    data_loader.save_encoders()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
