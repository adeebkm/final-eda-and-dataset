# Movie Recommendation System

## Project Overview
This project implements a movie recommendation system using multiple filtering techniques: content-based filtering (KNN), collaborative filtering with sentiment analysis, and a hybrid approach. The system analyzes movie features like plot, genres, directors, ratings, and emotional tone to suggest similar movies that users might enjoy.

## Dataset

The dataset used in this project is `movie_dataset_enhanced.csv`, which contains information about 1290 movies with the following key features:
- Movie titles and basic information (year, duration)
- Plot descriptions
- Genre classifications
- Director and cast information
- Ratings from multiple sources
- Vote counts

### Cleaned Dataset Schema
After preprocessing, the cleaned dataset (`movie_dataset_clean.csv`) includes:

| Column | Description | Type |
|--------|-------------|------|
| title | Movie title | string |
| year | Release year | numeric |
| plot | Original plot description | string |
| genres | Original genres format | string |
| combined_rating | Aggregated rating | float |
| imdb_id | IMDB identifier | string |
| vote_count | Number of votes received | float |
| director | Original director format | string |
| stars | Original cast format | string |
| genres_list | Genres as Python list | list |
| genres_clean | Genres as comma-separated string | string |
| director_list | Directors as Python list | list |
| director_clean | Directors as comma-separated string | string |
| stars_list | Cast as Python list | list |
| stars_clean | Cast as comma-separated string | string |
| plot_clean | Cleaned plot text | string |
| content_features | Combined text features for content analysis | string |

## Implementation

### Data Cleaning (`clean_movie_dataset.py`)
- Selects essential columns for recommendations
- Handles missing values
- Processes text and list data for analysis
- Creates combined content features
- Ensures data quality and consistency

### Recommendation System (`movie_recommendation.py`)
The recommendation system implements three approaches:

1. **Content-Based Filtering with KNN**
   - Uses TF-IDF vectorization on movie content features
   - Applies K-Nearest Neighbors to find similar movies
   - Similarity based on plot, genres, directors, and cast
   - Returns movies with similar content characteristics

2. **Collaborative Filtering with Sentiment Analysis**
   - Analyzes genre and rating similarities between movies
   - Performs sentiment analysis on movie plots to find emotionally similar films
   - Calculates mood match percentage between films
   - Weighted combination: 50% genre similarity, 30% rating similarity, 20% sentiment similarity
   - Adaptive to available data (falls back gracefully if sentiment data is unavailable)

3. **Hybrid Recommendations**
   - Combines content-based and collaborative filtering scores
   - Weighted average prioritizes content similarity (70%) over collaborative metrics (30%)
   - Provides more balanced recommendations leveraging both approaches

### Web Interface (`app.py`)
A responsive web application that allows users to:
- Search for movies by title with autocomplete-like suggestions
- Select recommendation methods (content-based, collaborative, or hybrid)
- Choose the number of recommendations to display (3, 5, or 10)
- View recommended movies with detailed information
- See similarity scores presented as percentages
- View sentiment/mood matching when using collaborative filtering
- Learn about the different recommendation approaches

### Command-Line Interface (`movie_recommender_cli.py`)
A user-friendly command-line interface for interacting with the recommendation system:

```
python3 movie_recommender_cli.py [options]

Options:
  --movie MOVIE      Movie title to get recommendations for
  --method {content,collaborative,hybrid}
                     Recommendation method to use (default: hybrid)
  --num NUM          Number of recommendations to show (default: 5)
  --list-movies      List all available movies
  --search SEARCH    Search for a movie by partial title
```

### Frontend Implementation
The web interface is built with:
- HTML/CSS/JavaScript for the frontend
- Bootstrap for responsive design
- Flask for the backend server
- Features a clean, intuitive UI with:
  - Movie search functionality
  - Method selection options
  - Detailed movie cards showing recommendations
  - Educational information about recommendation algorithms

## Usage in Python

```python
# Import the recommender
from movie_recommendation import MovieRecommender

# Initialize the recommender
recommender = MovieRecommender()

# Build the KNN model (this also builds the content model)
recommender.build_knn_model()

# Get content-based recommendations
content_recs = recommender.recommend_similar_movies("The Godfather")

# Get collaborative filtering recommendations with sentiment analysis
cf_recs = recommender.collaborative_filtering("The Godfather")

# Get hybrid recommendations
hybrid_recs = recommender.hybrid_recommendations("The Godfather")
```

## Running the Web Application

1. Install the required dependencies:
   ```
   pip install flask pandas scikit-learn textblob
   ```

2. Download NLTK data for TextBlob:
   ```
   python -m textblob.download_corpora
   ```

3. Start the web server:
   ```
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

## Future Improvements

Potential enhancements for the recommendation system:
1. Incorporate user data for personalized recommendations
2. Implement matrix factorization techniques like SVD or ALS
3. Add more advanced NLP for deeper plot understanding
4. Implement recommendation diversity measures
5. Add evaluation metrics to measure recommendation quality
6. Improve the recommendations by fine-tuning the similarity metrics
7. Add support for user profiles and preferences
8. Add movie posters and visual content

## Dependencies
- pandas
- numpy
- scikit-learn
- flask
- textblob (for sentiment analysis)
- re (regular expressions)
- ast (for safe evaluation of string representations)
