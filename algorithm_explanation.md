# Movie Recommendation Algorithms - Technical Explanation

This document provides a technical explanation of the recommendation algorithms used in our movie recommendation system.

## Overview

The system implements three main recommendation approaches:
1. Content-Based Filtering using KNN
2. Collaborative Filtering with Sentiment Analysis
3. Hybrid Recommendations

Each approach has distinct characteristics, strengths, and weaknesses, making them suitable for different recommendation scenarios.

## 1. Content-Based Filtering with KNN

### Algorithm Workflow

1. **Text Preprocessing**:
   - Convert all text features to lowercase
   - Remove special characters and extra whitespace
   - Combine multiple features (plot, genres, directors, stars) into a single text representation

2. **TF-IDF Vectorization**:
   - Transform the text data into a numerical format using Term Frequency-Inverse Document Frequency (TF-IDF)
   - This creates a sparse matrix where:
     - Rows represent movies
     - Columns represent unique terms (words) from all content features
     - Values represent the importance of each term to a movie relative to the entire collection

3. **K-Nearest Neighbors (KNN)**:
   - For each movie, find the k most similar movies based on cosine similarity in the TF-IDF space
   - Cosine similarity measures the cosine of the angle between two non-zero vectors, effectively comparing their orientation regardless of magnitude

### Technical Implementation

```python
# Create TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the content features
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_content_features)

# Initialize and fit KNN model
knn_model = NearestNeighbors(n_neighbors=n, algorithm='auto', metric='cosine')
knn_model.fit(tfidf_matrix)

# Get recommendations for a reference movie
distances, indices = knn_model.kneighbors(tfidf_matrix[movie_idx].reshape(1, -1))
```

### Mathematical Details

1. **TF-IDF Calculation**:
   - Term Frequency (TF) = (Number of times term t appears in document d) / (Total number of terms in document d)
   - Inverse Document Frequency (IDF) = log(Total number of documents / Number of documents containing term t)
   - TF-IDF = TF × IDF

2. **Cosine Similarity**:
   - For two document vectors A and B, the cosine similarity is:
   - cos(θ) = (A·B) / (||A|| × ||B||)
   - Where A·B is the dot product, and ||A|| and ||B|| are the magnitudes

### Feature Importance

The most influential factors in content-based recommendations (in order of impact):

1. **Plot** (highest weight): Contains rich semantic information about the movie's story, themes, setting, and characters
2. **Genres**: Categorical information that helps group similar movies
3. **Directors and Stars**: Capture stylistic elements and performance characteristics

## 2. Collaborative Filtering with Sentiment Analysis

Our implementation uses a simplified item-based collaborative filtering approach with sentiment analysis to match the emotional tone of movies.

### Algorithm Workflow

1. **Feature Extraction**:
   - Extract genre information and ratings for each movie
   - Extract plot text for sentiment analysis
   
2. **Similarity Calculation**:
   - **Genre Similarity**: Calculate Jaccard similarity between genre sets (50% weight)
   - **Rating Similarity**: Compute inverse normalized difference between ratings (30% weight)
   - **Sentiment Similarity**: Analyze emotional tone of movie plots (20% weight)

3. **Sentiment Analysis Process**:
   - Use TextBlob to calculate sentiment polarity of movie plots
   - Compare sentiment polarities between movies
   - Higher similarity for movies with matching emotional tones
   - Consider both sign and magnitude of sentiment

4. **Weighted Combination**:
   - Combine all similarities with configurable weights
   - Adaptive weighting based on available data

### Technical Implementation

```python
# Genre similarity (Jaccard index)
def genre_similarity(movie_genres, ref_genres):
    if not movie_genres_set or not ref_genres:
        return 0
    return len(ref_genres.intersection(movie_genres_set)) / len(ref_genres.union(movie_genres_set))

# Rating similarity
rating_diff = abs(movie_rating - ref_rating)
rating_similarity = 1 - (rating_diff / max_rating_diff)  # Normalized to [0,1]

# Sentiment analysis using TextBlob
ref_sentiment = TextBlob(ref_movie_plot).sentiment.polarity
movie_sentiment = TextBlob(movie_plot).sentiment.polarity

# Calculate sentiment similarity
if (ref_sentiment >= 0 and movie_sentiment >= 0) or (ref_sentiment < 0 and movie_sentiment < 0):
    # Same direction of sentiment (both positive or both negative)
    diff = abs(abs(ref_sentiment) - abs(movie_sentiment))
    sentiment_similarity = 1 - min(diff, 1)  # Ensure it's between 0 and 1
else:
    # Opposite sentiments, less similar
    diff = abs(ref_sentiment - movie_sentiment)
    sentiment_similarity = max(0, 1 - (diff / 2))

# Weighted combination
cf_score = (0.5 * genre_similarity) + (0.3 * rating_similarity) + (0.2 * sentiment_similarity)
```

### Mathematical Details

1. **Jaccard Similarity for Genres**:
   - For two sets A and B, the Jaccard similarity is:
   - J(A,B) = |A ∩ B| / |A ∪ B|
   - Where |A ∩ B| is the size of the intersection and |A ∪ B| is the size of the union

2. **Rating Similarity**:
   - Normalized inverse distance: 1 - (|rating_A - rating_B| / max_possible_difference)

3. **Sentiment Polarity**:
   - TextBlob returns a polarity score between -1 (very negative) and 1 (very positive)
   - 0 represents neutral sentiment
   
4. **Sentiment Similarity Calculation**:
   - For same polarity direction (both positive or both negative):
     - Focus on magnitude difference: 1 - |magnitude_A - magnitude_B|
   - For opposite polarity directions:
     - Calculate total distance in sentiment space: 1 - (|polarity_A - polarity_B| / 2)

## 3. Hybrid Recommendations

The hybrid approach combines the strengths of both content-based and collaborative filtering methods.

### Algorithm Workflow

1. **Obtain Recommendations from Both Methods**:
   - Generate a larger set of recommendations from content-based filtering
   - Generate a larger set of recommendations from collaborative filtering

2. **Score Combination**:
   - For each movie in the content-based recommendations, associate its collaborative filtering score if available
   - Calculate a weighted average of both scores (default: 0.7 for content, 0.3 for collaborative)

3. **Re-ranking**:
   - Sort movies by their hybrid scores and return the top N recommendations

### Technical Implementation

```python
# Get content-based recommendations
content_recs = recommend_similar_movies(movie_title, n_recommendations*2)

# Get collaborative filtering recommendations
cf_recs = collaborative_filtering(movie_title, n_recommendations*2)

# Convert CF results to a lookup dictionary
cf_scores = dict(zip(cf_recs['title'], cf_recs['cf_score']))

# Add CF score to content recommendations
content_recs['cf_score'] = content_recs['title'].apply(lambda x: cf_scores.get(x, 0))

# Calculate hybrid score
content_recs['hybrid_score'] = (0.7 * content_recs['similarity_score']) + (0.3 * content_recs['cf_score'])

# Sort and return top N
hybrid_recs = content_recs.sort_values('hybrid_score', ascending=False).head(n_recommendations)
```

## Web Application Implementation

The web application provides an intuitive interface for interacting with the recommendation system.

### Backend (Flask)

1. **Core Functionality**:
   - RESTful API endpoints for movie search and recommendations
   - JSON serialization with proper handling of NaN values and missing data
   - Robust error handling and graceful fallbacks

2. **API Endpoints**:
   - `GET /` - Serves the main page
   - `GET /search?query={movie_title}` - Searches for movies by title
   - `POST /recommend` - Gets recommendations based on selected movie and method
   - `GET /movies` - Returns all available movie titles

3. **Recommendation Processing**:
   - Processes recommendation requests
   - Handles different recommendation methods (content, collaborative, hybrid)
   - Returns detailed movie information and similarity scores

### Frontend (HTML/CSS/JavaScript)

1. **User Interface Components**:
   - Movie search with autocomplete-like results
   - Method selection (Content-based, Collaborative, Hybrid)
   - Number of recommendations selector
   - Results display with detailed movie cards
   - Informational sections explaining the algorithms

2. **Interactive Features**:
   - Real-time search results
   - Dynamic loading of recommendations
   - Similarity scores displayed as percentages
   - Sentiment/mood match display for collaborative filtering
   - Responsive design for mobile and desktop

3. **Data Flow**:
   - User inputs a movie title
   - System retrieves matching movies
   - User selects a movie and recommendation method
   - System processes the request and returns recommendations
   - Frontend displays the results with relevant details

## Performance Considerations

1. **TF-IDF Complexity**:
   - Time complexity: O(n_documents × avg_tokens_per_document)
   - Space complexity: O(n_documents × n_unique_terms) - but sparse matrix makes this efficient

2. **KNN Search**:
   - Naive implementation: O(n_documents × n_features)
   - With optimized algorithms and tree-based structures: O(log n_documents)

3. **Sentiment Analysis**:
   - Time complexity: O(text_length) for each movie
   - One-time computation that can be cached for better performance

4. **Collaborative Filtering**:
   - Time complexity: O(n_documents) for computing similarities
   - Space complexity: O(n_documents)

5. **Hybrid Approach**:
   - Slightly higher computational cost due to running both algorithms
   - Improved recommendation quality often justifies the additional computation

## Discussion: Strengths and Limitations

### Content-Based Filtering Strengths:
- No cold-start problem for new items
- Can recommend niche or unpopular items
- Explainable recommendations based on feature similarity

### Content-Based Filtering Limitations:
- Limited diversity (tendency to recommend very similar items)
- Cannot capture complex user preferences or context
- Quality depends on richness of features

### Collaborative Filtering Strengths:
- Can capture unexpected patterns and recommendations
- Sentiment analysis adds emotional context
- Often provides more diverse recommendations

### Collaborative Filtering Limitations:
- Cold-start problem for new users/items
- Limited by the available rating data
- Sentiment analysis may not capture complex emotions in movies

### Hybrid Approach Benefits:
- Mitigates limitations of individual approaches
- More balanced and robust recommendations
- Adaptive to different types of content and user preferences

## Future Algorithmic Improvements

1. **Advanced NLP for Content Processing**:
   - Word embeddings (Word2Vec, GloVe)
   - Contextual embeddings (BERT, GPT)
   - More sophisticated sentiment analysis models
   - Topic modeling (LDA) for thematic analysis

2. **True Collaborative Filtering with User Data**:
   - User-item matrix factorization (SVD, ALS)
   - Neural collaborative filtering

3. **Contextual and Sequential Recommendations**:
   - Session-based recommendations
   - Time-aware recommendations
   - Context-aware recommendations

4. **Evaluation and Optimization**:
   - A/B testing for comparing algorithm performance
   - Multi-objective optimization balancing accuracy, diversity, novelty 