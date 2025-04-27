# Movie Recommendation System - Code Structure

## Project Structure

```
movie-recommendation-system/
│
├── app.py                       # Flask web application
├── clean_movie_dataset.py       # Data cleaning and preprocessing
├── movie_recommendation.py      # Core recommendation algorithms
├── recommender.py               # Alternative implementation of recommendation algorithms
├── movie_recommender_cli.py     # Command-line interface
│
├── static/                      # Web application static files
│   ├── style.css                # Custom CSS styling
│   └── script.js                # Frontend JavaScript
│
├── templates/                   # Flask HTML templates
│   └── index.html               # Main web interface
│
├── data/                        # Data files
│   ├── movie_dataset_enhanced.csv   # Original movie dataset
│   ├── movie_dataset_enhanced.json  # JSON version of the dataset
│   └── movie_dataset_clean.csv      # Cleaned dataset (generated)
│
├── movie_dataset_eda.ipynb      # Exploratory Data Analysis notebook
├── movie_dataset_eda.py         # EDA script version
├── movie_dataset_eda.html       # EDA HTML report
│
└── docs/                        # Documentation
    ├── recommendation.md            # General project documentation
    ├── algorithm_explanation.md     # Technical explanation of algorithms
    ├── webapp_readme.md             # Web application documentation
    └── code_structure.md            # This file - code structure documentation
```

## Core Components and Their Functions

### 1. Data Processing (`clean_movie_dataset.py`)

This module handles data preparation and cleaning:

- **Main Class**: `MovieDataCleaner`
  - **Methods**:
    - `clean_data()`: Performs data cleaning and feature engineering
    - `parse_list_strings()`: Processes string lists into Python lists
    - `clean_text()`: Removes special characters and normalizes text
    - `create_content_features()`: Combines features for content analysis

- **Functions**:
  - `load_and_clean_data()`: Loads the dataset and runs cleaning operations
  - `save_cleaned_data()`: Stores the processed data as CSV

### 2. Recommendation Engine (`movie_recommendation.py`)

The core recommendation system with three filtering approaches:

- **Main Class**: `MovieRecommender`
  - **Initialization**:
    - Loads the cleaned dataset
    - Initializes model parameters
    - Sets up paths and configuration
  
  - **Core Methods**:
    - `build_knn_model()`: Creates TF-IDF vectors and KNN model for content-based filtering
      - Processes text data from content features
      - Applies TF-IDF vectorization with max_features=5000
      - Builds KNN model with cosine similarity metric
    
    - `recommend_similar_movies(movie_title, n_recommendations)`: 
      - Generates content-based recommendations
      - Uses movie features like plot, genres, directors, and cast
      - Returns DataFrame with similarity scores
    
    - `collaborative_filtering(movie_title, n_recommendations)`: 
      - Implements collaborative filtering with sentiment analysis
      - Analyzes genre and rating similarities (50% and 30% weights)
      - Uses TextBlob for sentiment analysis (20% weight)
      - Returns DataFrame with combined similarity scores
    
    - `hybrid_recommendations(movie_title, n_recommendations)`: 
      - Combines content-based and collaborative approaches
      - Gets larger sets from both methods
      - Applies weighted combination (70% content, 30% collaborative)
      - Returns optimally balanced recommendations
  
  - **Helper Methods**:
    - `find_movie_idx(movie_title)`: Locates movie index in dataset
    - `get_movie_details(movie_title)`: Retrieves movie information
    - `genre_similarity(movie_genres, ref_genres)`: Calculates Jaccard similarity of genres
    - `calculate_sentiment_similarity(ref_plot, movie_plot)`: Analyzes plot sentiment matching
    - `preprocess_data()`: Handles missing values and prepares features

### 3. Alternative Recommender (`recommender.py`)

An alternative implementation of the recommendation system with similar functionality:

- **Content-Based Filtering**:
  - Generates TF-IDF vectors from movie features
  - Implements KNN search for similar content
  - Returns movies with similar themes and characteristics

- **Collaborative Filtering**:
  - Uses a simpler implementation focused on genre and rating similarities
  - Incorporates sentiment analysis for emotional tone matching
  - Returns movies with similar audience appeal

- **Hybrid Recommendations**:
  - Blends content and collaborative scores with configurable weights
  - Ensures robust recommendations that balance both approaches
  - Prioritizes highly similar movies across multiple dimensions

### 4. Web Application (`app.py`)

Flask application for web interface interaction:

- **Setup and Initialization**:
  - Initializes Flask app with custom JSON encoder for NumPy types
  - Creates `MovieRecommender` instance
  - Loads dataset and builds recommendation models
  
- **API Endpoints**:
  - `/`: Serves main interface (index.html)
  - `/search`: 
    - Handles movie search requests with query parameter
    - Supports both exact and partial title matching
    - Returns JSON with movie details (title, year, genres, rating)
  
  - `/recommend`: 
    - Processes POST requests with movie title, method, and number
    - Handles three methods: content, collaborative, hybrid
    - Provides detailed logging and error handling
    - Returns formatted movie recommendations with similarity scores
  
  - `/movies`: 
    - Returns all movie titles as JSON array
    - Used for autocomplete and suggestion functionality
  
- **Request Processing**:
  - Detailed validation of input parameters
  - Robust error handling with meaningful messages
  - Extensive logging for troubleshooting
  - Careful handling of missing values and edge cases
  - Custom JSON encoder for handling NumPy and NaN values

### 5. Command Line Interface (`movie_recommender_cli.py`)

Provides a CLI for accessing recommendations:

- **Argument Parsing**:
  - `--movie`: Movie title to get recommendations for
  - `--method`: Recommendation method (content, collaborative, hybrid)
  - `--num`: Number of recommendations to show (default: 5)
  - `--list-movies`: List all available movies
  - `--search`: Search for a movie by partial title
  
- **Functions**:
  - `search_movie(title_fragment)`: 
    - Searches for movies by partial title
    - Returns exact and partial matches
    - Displays formatted results
  
  - `display_recommendations(recommendations, method)`: 
    - Formats and displays recommendation results
    - Shows title, year, genres, directors, and similarity scores
    - Presents information in a readable terminal format
  
  - `list_all_movies()`: 
    - Shows all available movie titles alphabetically
    - Paginates results for better readability

- **Error Handling**:
  - Validates user input before processing
  - Provides helpful error messages
  - Graceful handling of movie not found cases

### 6. Frontend (`static/` and `templates/`)

The user interface components:

- **HTML (`index.html`)**:
  - Responsive Bootstrap-based layout
  - Search form for movie selection
  - Method selection radio buttons 
  - Number selector for recommendation count
  - Results display with movie cards
  - Educational accordion sections about algorithms
  - Movie card template for dynamic results
  
- **CSS (`style.css`)**:
  - Responsive layout with media queries
  - Movie card styling with hover effects
  - Custom form control appearance
  - Loading indicator animations
  
- **JavaScript (`script.js`)**:
  - Event listeners for user interactions
  - Fetch API for AJAX requests
  - Dynamic rendering of search results
  - Movie selection and recommendation display
  - Error handling and user feedback
  - Similarity score formatting and display

## Data Flow

1. **Data Preparation**:
   ```
   Raw Dataset → clean_movie_dataset.py → Cleaned Dataset
   ```

2. **Model Building**:
   ```
   Cleaned Dataset → MovieRecommender.build_knn_model() → TF-IDF + KNN Model
   ```

3. **Recommendation Process**:
   ```
   User Input → app.py/CLI → MovieRecommender methods → Formatted Results
   ```

4. **Web Application Flow**:
   ```
   User → Web UI → Flask API → Recommender → JSON Response → Display Results
   ```

## Key Technical Implementations

### Content-Based Filtering
- `TfidfVectorizer` transforms text features into numerical vectors
- `NearestNeighbors` finds similar movies using cosine similarity
- Preprocessing combines plot, genres, directors, and stars

### Collaborative Filtering with Sentiment
- Calculates genre similarity using Jaccard index
- Computes rating similarity using normalized differences
- Uses `TextBlob` to analyze and compare plot sentiment polarity
- Combines similarities with weighted average (50/30/20)

### Hybrid Approach
- Obtains larger sets from both methods
- Maps collaborative scores to content recommendations
- Applies weighted combination (70/30)
- Sorts and returns top matches

### Error Handling
- Graceful degradation when data is missing
- Column presence checking before access
- NaN handling in calculations and JSON responses
- Try/except blocks for critical operations

## Algorithms and Models

1. **TF-IDF Vectorization**:
   - Converts text to sparse numerical matrix
   - Weights terms by frequency and uniqueness
   - Dimension: (1290 movies × 5000 features)

2. **K-Nearest Neighbors**:
   - Finds similar movies in TF-IDF vector space
   - Uses cosine similarity metric
   - Configurable number of neighbors (default: 10)

3. **Sentiment Analysis**:
   - Extracts emotional tone of movie plots
   - Provides polarity scores (-1 to +1)
   - Compares both direction and magnitude

4. **Similarity Calculations**:
   - Content: Cosine similarity in TF-IDF space
   - Genre: Jaccard similarity of genre sets
   - Rating: Normalized inverse difference
   - Sentiment: Polarity direction and magnitude match

## Dependencies and Requirements

- **Python Libraries**:
  - `pandas`: Data handling and manipulation
  - `numpy`: Numerical operations
  - `scikit-learn`: TF-IDF and KNN implementation
  - `flask`: Web server and API
  - `textblob`: Sentiment analysis
  - `re`: Regular expressions for text cleaning
  - `ast`: Safe evaluation of string representations
  
- **Frontend**:
  - Bootstrap 5 (CDN): Responsive layout
  - Vanilla JavaScript: No additional frameworks

## Testing and Debugging

- Flask debug mode enabled in development
- Detailed logging of recommendation processes
- Error feedback in API responses
- Caching of TF-IDF and KNN models for performance 