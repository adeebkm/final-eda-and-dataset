# Movie Recommendation System by us

A comprehensive movie recommendation system that combines content-based filtering, collaborative filtering with sentiment analysis, and a hybrid approach to provide personalized movie recommendations.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Data Processing](#data-processing)
5. [Recommendation Algorithms](#recommendation-algorithms)
6. [Similarity Metrics](#similarity-metrics)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Web Application](#web-application)
9. [Installation & Setup](#installation--setup)
10. [Usage Guide](#usage-guide)
11. [Results & Performance](#results--performance)
12. [Future Enhancements](#future-enhancements)

## Project Overview

This project implements a sophisticated movie recommendation system that uses multiple approaches to provide accurate and diverse movie recommendations. The system combines content-based filtering, collaborative filtering with sentiment analysis, and a hybrid approach to deliver personalized recommendations.

## Features

- **Multiple Recommendation Methods**:
  - Content-Based Filtering
  - Collaborative Filtering with Sentiment Analysis
  - Hybrid Approach
- **Advanced Data Processing**:
  - Data cleaning and preprocessing
  - Feature extraction
  - Sentiment analysis of movie plots
- **Web Interface**:
  - User-friendly search functionality
  - Configurable recommendation settings
  - Detailed movie information display
- **Evaluation Framework**:
  - Comprehensive metrics calculation
  - Performance visualization
  - Results analysis

## System Architecture

The system consists of several key components:

1. **Data Processing Pipeline**:
   - Data cleaning and preprocessing (`clean_movie_dataset.py`)
   - Feature extraction and transformation
   - Sentiment analysis integration

2. **Recommendation Engine**:
   - Content-based filtering implementation
   - Collaborative filtering with sentiment analysis
   - Hybrid recommendation algorithm

3. **Web Application**:
   - Flask backend (`app.py`)
   - Responsive frontend interface
   - Real-time recommendation generation

4. **Evaluation Framework**:
   - Metrics calculation (`evaluation_metrics.py`)
   - Performance visualization
   - Results analysis

## Data Processing

The system processes movie data through several stages:

1. **Data Cleaning**:
   - Handling missing values
   - Standardizing formats
   - Removing duplicates

2. **Feature Extraction**:
   - Genre analysis
   - Director and cast information
   - Plot sentiment analysis

3. **Data Transformation**:
   - TF-IDF vectorization
   - Feature normalization
   - Similarity matrix computation

## Recommendation Algorithms

### 1. Content-Based Filtering
- Analyzes movie features (plot, genres, directors, cast)
- Uses TF-IDF vectorization for text features
- Implements KNN for similarity calculation

### 2. Collaborative Filtering with Sentiment Analysis
- Considers user ratings and preferences
- Analyzes emotional tone of movie plots
- Combines genre similarity, rating patterns, and sentiment matching

### 3. Hybrid Approach
- Weighted combination of content-based and collaborative filtering
- Dynamic adjustment based on available data
- Optimized for accuracy and diversity

## Similarity Metrics

### Cosine Similarity vs. Euclidean Distance

Our recommendation system uses cosine similarity rather than Euclidean distance for several important reasons:

#### 1. **Direction vs. Magnitude**
- **Cosine Similarity** measures the angle between vectors, focusing on the direction of preference rather than magnitude
- **Euclidean Distance** measures absolute distance, which can be misleading when users have different rating scales

#### 2. **High-Dimensional Spaces**
- Movie feature vectors are high-dimensional (genres, actors, plot terms, etc.)
- Euclidean distance suffers from the "curse of dimensionality" where distances become less meaningful
- Cosine similarity remains effective in high-dimensional spaces, capturing conceptual similarity

#### 3. **Sparse Data Handling**
- **Cosine Similarity** works well with sparse data (many zero values), which is common in movie features
- Two movies with a few matching features can have high cosine similarity even if they differ in many other aspects
- Euclidean distance would penalize these differences disproportionately

#### 4. **Scale Invariance**
- Cosine similarity is scale-invariant (normalizes vectors), so it's not affected by how enthusiastic users are with ratings
- Example: A user who rates movies 4-5 and another who rates 1-5 can be compared fairly

#### 5. **Practical Example**
Consider two sci-fi movies: One with high ratings and another with moderate ratings but similar content:
- Euclidean distance would suggest they're dissimilar due to rating differences
- Cosine similarity would recognize their content similarity despite rating differences

This focus on preference direction rather than magnitude leads to more intuitive recommendations, especially in content-based filtering where we're matching movie feature vectors.

### Jaccard Similarity vs. Sarwar/Karypis Methods

Our system employs Jaccard similarity for specific components where it offers advantages over traditional collaborative filtering methods (Sarwar/Karypis):

#### 1. **Optimal for Set-Based Comparisons**
- **Jaccard Similarity** excels at comparing sets of features (genres, actors, tags)
- It measures the size of intersection divided by size of union: |A∩B|/|A∪B|
- Perfect for our genre-matching components where categorical overlap matters most

#### 2. **Binary/Categorical Data Handling**
- Jaccard naturally handles binary or categorical data (a movie either has a genre or doesn't)
- Traditional Sarwar/Karypis methods are optimized for numerical ratings
- For genre comparison in our serendipity calculation, Jaccard provides more intuitive results

#### 3. **Sparsity Robustness**
- Jaccard only considers non-zero elements (features that exist)
- Less vulnerable to the extreme sparsity that affects collaborative filtering approaches
- Provides meaningful similarities even with limited overlapping features

#### 4. **No Rating Bias**
- Jaccard ignores rating magnitude issues entirely
- Sarwar/Karypis methods must account for different user rating scales and biases
- Appropriate when comparing movie attributes rather than user preferences

#### 5. **Hybrid Approach Benefits**
- Our system uses Jaccard for content features (especially genres)
- Pearson correlation (Sarwar/Karypis) for user rating patterns when available
- This hybrid approach leverages the strengths of each method in its most appropriate context

The choice of similarity measure significantly impacts recommendation quality. For our genre-based matching and serendipity calculation, Jaccard similarity provides more intuitive results that better reflect the conceptual similarity between movies.

## Evaluation Metrics

The system is evaluated using these key metrics:

1. **Diversity**: Measures the variety of genres in recommendations
   - Calculated as the ratio of unique genres to the number of recommendations
   - Higher values indicate more diverse recommendations across different genres
   - Ideal for users looking to explore variety in their movie watching

2. **Novelty**: Assesses how new the recommendations are to users
   - Calculated as the proportion of recommended movies not in the user's history
   - Higher values indicate more fresh content being recommended
   - Balances introducing new movies while respecting user preferences

3. **Serendipity**: Evaluates surprising but relevant recommendations
   - Identifies movies users wouldn't find on their own but would enjoy
   - Uses genre similarity to ensure recommendations are still relevant
   - Measures the "pleasant surprise" factor of recommendations

Results are visualized and saved for analysis:
- Bar charts showing metric scores in `evaluation_metrics.png`
- Standard deviation analysis for reliability assessment
- Detailed JSON results in `evaluation_results.json`

## Web Application

The web interface provides:

1. **Search Functionality**:
   - Real-time movie search
   - Auto-complete suggestions
   - Detailed movie information

2. **Recommendation Settings**:
   - Method selection
   - Number of recommendations
   - Advanced filtering options

3. **Results Display**:
   - Movie cards with detailed information
   - Similarity scores
   - Sentiment matching indicators

## Installation & Setup

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download NLTK data:
   ```bash
   python -m textblob.download_corpora
   ```

3. Start the web server:
   ```bash
   python app.py
   ```

4. Access the application at:
   ```
   http://localhost:8080
   ```

## Usage Guide

1. **Search for Movies**:
   - Enter a movie title in the search box
   - Select from the displayed results

2. **Configure Recommendations**:
   - Choose the recommendation method
   - Select the number of recommendations
   - Apply any desired filters

3. **View Results**:
   - Examine recommended movies
   - Review similarity scores
   - Check sentiment matching

## Results & Performance

The system has been evaluated using our comprehensive metrics framework on synthetic recommendation data generated for 20 users:

- **Diversity Score**: 0.78 ± 0.31
   - This high score indicates strong variety in genres across recommendations
   - The standard deviation (±0.31) shows flexibility in recommendations based on user preferences
   - Each recommendation set contains nearly as many unique genres as movies

- **Novelty Score**: 0.44 ± 0.16
   - This balanced score shows 44% of recommendations are new to users
   - Demonstrates a good balance between familiar content and new discoveries
   - Particularly important for expanding users' movie horizons

- **Serendipity Score**: 0.11 ± 0.16
   - This score reflects how many recommendations are both surprising and relevant
   - Currently an area for improvement in our algorithm
   - Even at this level, it ensures some "pleasant surprises" for users

Our evaluation methodology uses synthetic data generation to assess the recommendation system's performance across diverse user preferences. The visualization of these metrics is available in `evaluation_metrics.png`, and detailed results are stored in `evaluation_results.json`.

## Future Enhancements

1. **Advanced Features**:
   - User accounts and personalized profiles
   - Social features and sharing
   - Advanced filtering and sorting options

2. **Technical Improvements**:
   - Real-time recommendation updates
   - Machine learning model optimization
   - Enhanced sentiment analysis

3. **User Experience**:
   - Movie posters and trailers
   - User feedback integration
   - Mobile application development

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
