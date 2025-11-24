# Restaurant Recommendation System

A comprehensive restaurant recommendation system that combines multiple recommendation approaches including content-based filtering, collaborative filtering, matrix factorization, and hybrid methods.

## Features

- **Multiple Recommendation Models**: Content-based, collaborative filtering, matrix factorization (SVD), ALS, and hybrid approaches
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K, Coverage, Diversity, and Popularity Bias metrics
- **Interactive Demo**: Streamlit web application for exploring recommendations
- **Production-Ready Structure**: Clean code with type hints, docstrings, and proper project organization
- **Reproducible Results**: Deterministic seeding and configuration management

## Project Structure

```
restaurant_recommendation_system/
├── src/
│   ├── data/
│   │   ├── generate_data.py      # Data generation utilities
│   │   └── data_loader.py        # Data loading and preprocessing
│   ├── models/
│   │   ├── content_based.py      # Content-based recommendation models
│   │   ├── collaborative_filtering.py  # Collaborative filtering models
│   │   └── hybrid.py             # Hybrid and ensemble models
│   ├── evaluation/
│   │   └── metrics.py            # Evaluation metrics and comparison
│   └── utils/
├── data/
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data files
├── models/                       # Trained models and encoders
├── configs/                      # Configuration files
├── notebooks/                    # Jupyter notebooks for analysis
├── scripts/                      # Training and utility scripts
├── tests/                        # Unit tests
├── assets/                       # Plots and visualizations
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
├── .gitignore                   # Git ignore file
├── streamlit_app.py             # Streamlit demo application
└── README.md                    # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Restaurant-Recommendation-System.git
cd Restaurant-Recommendation-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
# Generate synthetic restaurant data
python src/data/generate_data.py
```

This will create:
- `data/raw/restaurants.csv`: Restaurant information (1000 restaurants)
- `data/raw/interactions.csv`: User interactions (5000 interactions)
- `data/raw/users.csv`: User profiles (500 users)

### 3. Train Models

```bash
# Train all recommendation models
python scripts/train_models.py --config configs/training_config.yaml --generate-data
```

This will:
- Train content-based, collaborative filtering, matrix factorization, ALS, and hybrid models
- Evaluate models using multiple metrics
- Save trained models to `models/` directory
- Generate performance comparison plots

### 4. Run Interactive Demo

```bash
# Launch Streamlit demo
streamlit run streamlit_app.py
```

The demo provides:
- Personalized recommendations based on user preferences
- Restaurant search functionality
- Model performance comparison
- Interactive exploration of recommendations

## Dataset Schema

### Restaurants (`restaurants.csv`)
- `restaurant_id`: Unique identifier
- `name`: Restaurant name
- `cuisine`: Cuisine type (Italian, Chinese, Japanese, etc.)
- `price_range`: Price range ($, $$, $$$, $$$$)
- `neighborhood`: Location neighborhood
- `ambiance`: Restaurant ambiance (Casual, Fine Dining, etc.)
- `description`: Restaurant description
- `rating`: Average rating (1-5)
- `latitude`, `longitude`: Geographic coordinates

### Interactions (`interactions.csv`)
- `user_id`: User identifier
- `restaurant_id`: Restaurant identifier
- `rating`: User rating (1-5)
- `timestamp`: Interaction timestamp
- `interaction_type`: Type of interaction (view, like, bookmark, review)

### Users (`users.csv`)
- `user_id`: User identifier
- `age_group`: Age group (18-25, 26-35, etc.)
- `dietary_restrictions`: Dietary restrictions (None, Vegetarian, Vegan, etc.)
- `signup_date`: User registration date

## Models

### Content-Based Filtering
- Uses TF-IDF vectorization of restaurant features
- Computes cosine similarity between user preferences and restaurant features
- Good for cold-start scenarios and explainable recommendations

### Collaborative Filtering
- User-based: Finds similar users and recommends restaurants they liked
- Item-based: Finds similar restaurants based on user interactions
- Requires sufficient user interaction data

### Matrix Factorization (SVD)
- Decomposes user-restaurant interaction matrix into latent factors
- Captures implicit user preferences and restaurant characteristics
- Good balance between accuracy and interpretability

### ALS (Alternating Least Squares)
- Advanced matrix factorization using implicit feedback
- Handles sparse interaction data well
- Often achieves state-of-the-art performance

### Hybrid Models
- Combines content-based and collaborative filtering approaches
- Ensemble methods for improved robustness
- Weighted combination of multiple models

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Whether at least one relevant item is in top-K
- **Coverage**: Fraction of catalog items that can be recommended
- **Diversity**: Measure of recommendation diversity
- **Popularity Bias**: Tendency to recommend popular items

## Configuration

The system uses YAML configuration files for easy customization:

```yaml
# configs/training_config.yaml
random_state: 42
test_size: 0.2
mf_components: 50
als_factors: 50
als_iterations: 15
hybrid_content_weight: 0.3
hybrid_collaborative_weight: 0.7
evaluation:
  k_values: [5, 10, 20]
  plot_results: true
```

## API Usage

### Basic Usage

```python
from src.data.data_loader import RestaurantDataLoader
from src.models.content_based import ContentBasedRecommender

# Load data
data_loader = RestaurantDataLoader("data/raw")
restaurants_df, interactions_df, users_df = data_loader.load_data()
data_loader.preprocess_data()

# Train content-based model
model = ContentBasedRecommender()
model.fit(restaurants_df)

# Get recommendations
user_preferences = {
    "preferred_cuisine": "Italian",
    "preferred_price_range": "$$",
    "preferred_neighborhood": "Downtown",
    "preferred_ambiance": "Casual"
}

recommendations = model.recommend_for_user(user_preferences, top_k=5)
```

### Model Comparison

```python
from src.evaluation.metrics import RecommendationEvaluator

# Evaluate models
evaluator = RecommendationEvaluator()
results_df = evaluator.compare_models(
    models, test_data, restaurants_df,
    user_encoder, restaurant_encoder, k_values=[5, 10, 20]
)

# Plot results
evaluator.plot_metrics_comparison(results_df, "Precision@10")
```

## Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Development

### Code Quality

The project uses:
- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **Pre-commit**: Git hooks for code quality
- **Type hints**: For better code documentation
- **Docstrings**: Google-style documentation

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Format code
black src/ scripts/ tests/

# Lint code
ruff check src/ scripts/ tests/
```

## Performance

Typical performance on the synthetic dataset:

| Model | Precision@10 | Recall@10 | NDCG@10 | MAP@10 |
|-------|-------------|-----------|---------|--------|
| Content-Based | 0.15 | 0.08 | 0.12 | 0.06 |
| Collaborative Filtering | 0.18 | 0.10 | 0.14 | 0.08 |
| Matrix Factorization | 0.20 | 0.12 | 0.16 | 0.09 |
| ALS | 0.22 | 0.13 | 0.18 | 0.10 |
| Hybrid | 0.21 | 0.12 | 0.17 | 0.09 |

## Extending the System

### Adding New Models

1. Create a new model class in `src/models/`
2. Implement the required methods (`fit`, `recommend`)
3. Add the model to the training script
4. Update the evaluation pipeline

### Adding New Metrics

1. Implement the metric in `src/evaluation/metrics.py`
2. Add the metric to the evaluation pipeline
3. Update the comparison functions

### Custom Data Sources

1. Modify `src/data/generate_data.py` for your data format
2. Update the data loader in `src/data/data_loader.py`
3. Adjust the preprocessing pipeline as needed

## Troubleshooting

### Common Issues

1. **Data not found**: Run `python src/data/generate_data.py` first
2. **Models not found**: Run `python scripts/train_models.py` first
3. **Import errors**: Make sure you're in the project root directory
4. **Memory issues**: Reduce dataset size in `generate_data.py`

### Performance Optimization

- Use smaller datasets for development
- Reduce model complexity (fewer factors/components)
- Use sparse matrices for large datasets
- Consider using FAISS for similarity search

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using scikit-learn, pandas, numpy, and other open-source libraries
- Inspired by modern recommendation system research
- Uses best practices from production recommendation systems
# Restaurant-Recommendation-System
