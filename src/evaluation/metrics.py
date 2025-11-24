"""Evaluation metrics and model comparison utilities."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns


class RecommendationEvaluator:
    """Evaluator for recommendation system metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.metrics_history = {}
        
    def precision_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate Precision@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
        
        return relevant_in_top_k / k
    
    def recall_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate Recall@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
        
        return relevant_in_top_k / len(relevant_items)
    
    def ndcg_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate NDCG@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        if k == 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommendations[:k]):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def map_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate MAP@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant_items)
    
    def hit_rate_at_k(
        self, 
        recommendations: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Hit Rate@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        return 1.0 if len(set(top_k_recs) & set(relevant_items)) > 0 else 0.0
    
    def coverage(
        self, 
        all_recommendations: List[List[str]], 
        total_items: int
    ) -> float:
        """Calculate coverage of recommendations.
        
        Args:
            all_recommendations: List of recommendation lists for all users
            total_items: Total number of items in the catalog
            
        Returns:
            Coverage score
        """
        all_recommended_items = set()
        for recs in all_recommendations:
            all_recommended_items.update(recs)
        
        return len(all_recommended_items) / total_items
    
    def diversity(
        self, 
        recommendations: List[str], 
        item_features: pd.DataFrame,
        feature_cols: List[str]
    ) -> float:
        """Calculate diversity of recommendations.
        
        Args:
            recommendations: List of recommended item IDs
            item_features: DataFrame with item features
            feature_cols: List of feature columns to consider
            
        Returns:
            Diversity score
        """
        if len(recommendations) <= 1:
            return 0.0
        
        # Get features for recommended items
        rec_features = item_features[
            item_features.index.isin(recommendations)
        ][feature_cols]
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(rec_features)):
            for j in range(i + 1, len(rec_features)):
                # Simple Hamming distance for categorical features
                distance = (rec_features.iloc[i] != rec_features.iloc[j]).sum()
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def popularity_bias(
        self, 
        recommendations: List[str], 
        item_popularity: Dict[str, int]
    ) -> float:
        """Calculate popularity bias of recommendations.
        
        Args:
            recommendations: List of recommended item IDs
            item_popularity: Dictionary mapping item IDs to popularity counts
            
        Returns:
            Average popularity of recommended items
        """
        if not recommendations:
            return 0.0
        
        popularities = [item_popularity.get(item_id, 0) for item_id in recommendations]
        return np.mean(popularities)
    
    def evaluate_model(
        self, 
        model, 
        test_data: pd.DataFrame,
        restaurants_df: pd.DataFrame,
        user_encoder,
        restaurant_encoder,
        k_values: List[int] = [5, 10, 20],
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """Evaluate a recommendation model.
        
        Args:
            model: Trained recommendation model
            test_data: Test interaction data
            restaurants_df: Restaurant information
            user_encoder: User label encoder
            restaurant_encoder: Restaurant label encoder
            k_values: List of K values to evaluate
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Group test data by user
        user_groups = test_data.groupby("user_id")
        
        all_precisions = {k: [] for k in k_values}
        all_recalls = {k: [] for k in k_values}
        all_ndcgs = {k: [] for k in k_values}
        all_maps = {k: [] for k in k_values}
        all_hit_rates = {k: [] for k in k_values}
        
        for user_id, user_data in user_groups:
            # Get relevant items for this user
            relevant_items = user_data["restaurant_id"].tolist()
            
            try:
                # Get recommendations
                if hasattr(model, 'recommend'):
                    recs = model.recommend(user_id, max(k_values))
                else:
                    recs = model.recommend_user_based(user_id, max(k_values))
                
                recommendations = [rec["restaurant_id"] for rec in recs]
                
                # Calculate metrics for each k
                for k in k_values:
                    all_precisions[k].append(
                        self.precision_at_k(recommendations, relevant_items, k)
                    )
                    all_recalls[k].append(
                        self.recall_at_k(recommendations, relevant_items, k)
                    )
                    all_ndcgs[k].append(
                        self.ndcg_at_k(recommendations, relevant_items, k)
                    )
                    all_maps[k].append(
                        self.map_at_k(recommendations, relevant_items, k)
                    )
                    all_hit_rates[k].append(
                        self.hit_rate_at_k(recommendations, relevant_items, k)
                    )
                    
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue
        
        # Calculate average metrics
        for k in k_values:
            metrics[f"Precision@{k}"] = np.mean(all_precisions[k])
            metrics[f"Recall@{k}"] = np.mean(all_recalls[k])
            metrics[f"NDCG@{k}"] = np.mean(all_ndcgs[k])
            metrics[f"MAP@{k}"] = np.mean(all_maps[k])
            metrics[f"HitRate@{k}"] = np.mean(all_hit_rates[k])
        
        # Store metrics history
        self.metrics_history[model_name] = metrics
        
        return metrics
    
    def compare_models(
        self, 
        models: Dict[str, object],
        test_data: pd.DataFrame,
        restaurants_df: pd.DataFrame,
        user_encoder,
        restaurant_encoder,
        k_values: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """Compare multiple models.
        
        Args:
            models: Dictionary mapping model names to model objects
            test_data: Test interaction data
            restaurants_df: Restaurant information
            user_encoder: User label encoder
            restaurant_encoder: Restaurant label encoder
            k_values: List of K values to evaluate
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(
                model, test_data, restaurants_df, 
                user_encoder, restaurant_encoder, k_values, model_name
            )
            
            metrics["Model"] = model_name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_metrics_comparison(
        self, 
        metrics_df: pd.DataFrame,
        metric: str = "Precision@10",
        save_path: Optional[str] = None
    ) -> None:
        """Plot metrics comparison.
        
        Args:
            metrics_df: DataFrame with metrics comparison
            metric: Metric to plot
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics_df, x="Model", y=metric)
        plt.title(f"{metric} Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_metrics_by_k(
        self, 
        model_name: str,
        k_values: List[int] = [5, 10, 20],
        save_path: Optional[str] = None
    ) -> None:
        """Plot metrics vs K for a specific model.
        
        Args:
            model_name: Name of the model
            k_values: List of K values
            save_path: Optional path to save the plot
        """
        if model_name not in self.metrics_history:
            print(f"Model {model_name} not found in metrics history")
            return
        
        metrics = self.metrics_history[model_name]
        
        plt.figure(figsize=(12, 8))
        
        # Plot different metrics
        metrics_to_plot = ["Precision", "Recall", "NDCG", "MAP"]
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, 2, i + 1)
            values = [metrics[f"{metric}@{k}"] for k in k_values]
            plt.plot(k_values, values, marker="o")
            plt.title(f"{metric}@{k}")
            plt.xlabel("K")
            plt.ylabel(metric)
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
