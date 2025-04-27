import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import re
import ast
from clean_movie_dataset import clean_dataset
from textblob import TextBlob

class MovieRecommender:
    def __init__(self, df=None):
        """Initialize the movie recommender system
        
        Args:
            df (pandas.DataFrame, optional): Cleaned movie dataset. If None,
                the system will attempt to load from file, and if that fails,
                it will clean the raw dataset.
        """
        if df is not None:
            self.df = df
        else:
            try:
                print("Attempting to load cleaned dataset...")
                self.df = pd.read_csv('movie_dataset_clean.csv')
                print("Cleaned dataset loaded successfully.")
            except FileNotFoundError:
                print("Cleaned dataset not found. Cleaning original dataset...")
                self.df = clean_dataset()
        
        # Initialize models
        self.content_model = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.knn_model = None
    
    def preprocess_text(self, text):
        """Clean and normalize text data
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_content_model(self):
        """Build the content-based recommendation model using TF-IDF"""
        print("Building content-based recommendation model...")
        
        # Create a TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        
        # Fit and transform the content features
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.df['content_features'].apply(self.preprocess_text)
        )
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        self.content_model = True
    
    def build_knn_model(self, n_neighbors=10):
        """Build the KNN model for recommendation
        
        Args:
            n_neighbors (int): Number of neighbors to consider
        """
        print("Building KNN recommendation model...")
        
        # Ensure content model is built first
        if not self.content_model:
            self.build_content_model()
        
        # Initialize and fit KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm='auto',
            metric='cosine'
        )
        self.knn_model.fit(self.tfidf_matrix)
        
        print(f"KNN model built with {n_neighbors} neighbors")
    
    def get_movie_index(self, movie_title):
        """Get the index of a movie by its title
        
        Args:
            movie_title (str): Title of the movie
            
        Returns:
            int: Index of the movie, or None if not found
        """
        matches = self.df[self.df['title'].str.lower() == movie_title.lower()]
        if not matches.empty:
            return matches.index[0]
        else:
            # Try partial matching
            matches = self.df[self.df['title'].str.lower().str.contains(movie_title.lower())]
            if not matches.empty:
                return matches.index[0]
            else:
                return None
    
    def recommend_similar_movies(self, movie_title, n_recommendations=5):
        """Recommend similar movies based on content similarity
        
        Args:
            movie_title (str): Title of the reference movie
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            pandas.DataFrame: Dataframe of recommended movies
        """
        # Ensure KNN model is built
        if self.knn_model is None:
            self.build_knn_model()
        
        # Get movie index
        movie_idx = self.get_movie_index(movie_title)
        if movie_idx is None:
            print(f"Movie '{movie_title}' not found in the dataset.")
            return pd.DataFrame()
        
        # Get KNN indices and distances
        distances, indices = self.knn_model.kneighbors(
            self.tfidf_matrix[movie_idx].reshape(1, -1),
            n_neighbors=n_recommendations+1  # +1 because the movie itself will be included
        )
        
        # Get recommended movie indices (skip the first one as it's the movie itself)
        recommended_indices = indices.flatten()[1:]
        
        # Create recommendations dataframe
        recommendations = self.df.iloc[recommended_indices][
            ['title', 'year', 'genres_clean', 'combined_rating', 'director_clean']
        ].copy()
        
        # Add similarity score
        similarity_scores = 1 - distances.flatten()[1:]  # Convert distance to similarity
        recommendations['similarity_score'] = similarity_scores
        
        return recommendations.sort_values('similarity_score', ascending=False)
    
    def collaborative_filtering(self, movie_title, n_recommendations=5):
        """Simple collaborative filtering based on rating similarity
        
        Args:
            movie_title (str): Title of the reference movie
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            pandas.DataFrame: Dataframe of recommended movies
        """
        # Get movie index
        movie_idx = self.get_movie_index(movie_title)
        if movie_idx is None:
            print(f"Movie '{movie_title}' not found in the dataset.")
            return pd.DataFrame()
        
        # Get reference movie's features
        ref_movie = self.df.iloc[movie_idx]
        
        # Handle genres safely
        ref_genres = set()
        if 'genres_list' in self.df.columns and isinstance(ref_movie.get('genres_list'), list):
            ref_genres = set(ref_movie['genres_list'])
        elif 'genres' in self.df.columns:
            # If genres_list is not available, try using genres string
            genres_str = ref_movie.get('genres', '')
            if isinstance(genres_str, str) and genres_str:
                ref_genres = set(genres_str.split('|'))
        
        print(f"Reference genres: {ref_genres}")
        
        # Get sentiment of reference movie plot if available
        ref_sentiment = 0
        plot_col = None
        
        # Determine which plot column to use
        for col in ['plot_clean', 'plot', 'overview', 'summary', 'description']:
            if col in self.df.columns:
                plot_col = col
                break
        
        if plot_col and pd.notna(ref_movie.get(plot_col)) and ref_movie.get(plot_col) != '':
            try:
                ref_sentiment = TextBlob(str(ref_movie.get(plot_col))).sentiment.polarity
                print(f"Reference movie sentiment: {ref_sentiment} (using {plot_col} column)")
            except Exception as e:
                print(f"Error calculating reference movie sentiment: {str(e)}")
                ref_sentiment = 0
        else:
            print("No plot data available for sentiment analysis")
        
        # Create a subset of movies excluding the reference movie
        other_movies = self.df[self.df.index != movie_idx].copy()
        
        # Determine which rating column to use
        rating_col = None
        for col in ['combined_rating', 'rating', 'vote_average']:
            if col in self.df.columns:
                rating_col = col
                break
        
        if rating_col:
            print(f"Using '{rating_col}' for rating similarity")
            # Filter for movies with ratings
            other_movies = other_movies.dropna(subset=[rating_col])
        else:
            print("No rating column found for similarity calculation")
        
        # Calculate genre similarity - proportion of genres that match
        def genre_similarity(row):
            movie_genres = set()
            
            # Try to get genres from different possible formats
            if 'genres_list' in self.df.columns and isinstance(row.get('genres_list'), list):
                movie_genres = set(row['genres_list'])
            elif 'genres' in self.df.columns and isinstance(row.get('genres'), str):
                movie_genres = set(row['genres'].split('|'))
            
            if not movie_genres or not ref_genres:
                return 0
            
            return len(ref_genres.intersection(movie_genres)) / len(ref_genres.union(movie_genres))
        
        # Calculate genre similarities
        other_movies['genre_similarity'] = other_movies.apply(genre_similarity, axis=1)
        
        # Calculate rating similarity (penalize large rating differences)
        if rating_col and pd.notna(ref_movie.get(rating_col)):
            other_movies['rating_diff'] = abs(other_movies[rating_col] - ref_movie[rating_col])
            max_rating_diff = other_movies['rating_diff'].max() if not other_movies.empty else 1
            if max_rating_diff > 0:
                other_movies['rating_similarity'] = 1 - (other_movies['rating_diff'] / max_rating_diff)
            else:
                other_movies['rating_similarity'] = 1.0  # All ratings are the same
        else:
            other_movies['rating_similarity'] = 0.5  # Neutral if reference rating is missing
        
        # Calculate sentiment similarity only if we have a plot column and reference sentiment
        if plot_col and ref_sentiment != 0:
            # Calculate sentiment similarity
            def calculate_sentiment_similarity(movie_plot):
                # Default to neutral similarity if no plot
                if pd.isna(movie_plot) or movie_plot == '':
                    return 0.5
                
                try:
                    # Calculate sentiment of movie plot
                    movie_sentiment = TextBlob(str(movie_plot)).sentiment.polarity
                    
                    # Calculate similarity (inverse of absolute difference, normalized to 0-1)
                    # When both sentiments have same sign, they're more similar
                    if (ref_sentiment >= 0 and movie_sentiment >= 0) or (ref_sentiment < 0 and movie_sentiment < 0):
                        # Same direction of sentiment (both positive or both negative)
                        # Calculate similarity based on difference in magnitude
                        diff = abs(abs(ref_sentiment) - abs(movie_sentiment))
                        # Map difference to similarity (closer to 0 difference = higher similarity)
                        return 1 - min(diff, 1)  # Ensure it's between 0 and 1
                    else:
                        # Opposite sentiments, less similar
                        # Calculate total distance in polarity space (range -2 to 2)
                        diff = abs(ref_sentiment - movie_sentiment)
                        # Map to similarity scale (0-1)
                        return max(0, 1 - (diff / 2))
                except Exception as e:
                    print(f"Error calculating sentiment similarity: {str(e)}")
                    return 0.5  # Default to neutral
            
            # Add sentiment similarity for each movie
            other_movies['sentiment_similarity'] = other_movies[plot_col].apply(calculate_sentiment_similarity)
            
            # Calculate overall similarity score with sentiment (weighted average)
            other_movies['cf_score'] = (
                (0.5 * other_movies['genre_similarity']) + 
                (0.3 * other_movies['rating_similarity']) +
                (0.2 * other_movies['sentiment_similarity'])
            )
        else:
            # Without sentiment, just use genre and rating
            print("Skipping sentiment analysis due to missing data")
            other_movies['sentiment_similarity'] = 0.5  # Neutral sentiment similarity
            other_movies['cf_score'] = (
                (0.7 * other_movies['genre_similarity']) + 
                (0.3 * other_movies['rating_similarity'])
            )
        
        # Prepare columns for results
        result_columns = ['title', 'year']
        
        # Add genres column (use appropriate one based on what's available)
        if 'genres_clean' in self.df.columns:
            result_columns.append('genres_clean')
        elif 'genres' in self.df.columns:
            result_columns.append('genres')
        
        # Add rating column
        if rating_col:
            result_columns.append(rating_col)
        
        # Add director column if available
        if 'director_clean' in self.df.columns:
            result_columns.append('director_clean')
        elif 'director' in self.df.columns:
            result_columns.append('director')
        
        # Add score columns
        result_columns.extend(['cf_score', 'sentiment_similarity'])
        
        # Get top recommendations - handle missing columns gracefully
        valid_columns = [col for col in result_columns if col in other_movies.columns]
        recommendations = other_movies.sort_values('cf_score', ascending=False).head(n_recommendations)[valid_columns]
        
        return recommendations
    
    def hybrid_recommendations(self, movie_title, n_recommendations=5):
        """Hybrid recommendation combining content-based and collaborative filtering
        
        Args:
            movie_title (str): Title of the reference movie
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            pandas.DataFrame: Dataframe of recommended movies
        """
        # Get content-based recommendations
        content_recs = self.recommend_similar_movies(movie_title, n_recommendations*2)
        if content_recs.empty:
            return pd.DataFrame()
        
        # Get collaborative filtering recommendations
        cf_recs = self.collaborative_filtering(movie_title, n_recommendations*2)
        
        # Merge recommendations
        # Convert to dictionaries for faster lookup
        cf_scores = dict(zip(cf_recs['title'], cf_recs['cf_score']))
        
        # Add CF score to content recommendations
        content_recs['cf_score'] = content_recs['title'].apply(
            lambda x: cf_scores.get(x, 0)
        )
        
        # Calculate hybrid score (weighted average)
        content_recs['hybrid_score'] = (
            (0.7 * content_recs['similarity_score']) + 
            (0.3 * content_recs['cf_score'])
        )
        
        # Get top hybrid recommendations
        hybrid_recs = content_recs.sort_values(
            'hybrid_score', ascending=False
        ).head(n_recommendations)[
            ['title', 'year', 'genres_clean', 'combined_rating', 'director_clean', 'hybrid_score']
        ]
        
        return hybrid_recs

if __name__ == "__main__":
    # Create recommender instance
    recommender = MovieRecommender()
    
    # Build models
    recommender.build_knn_model()
    
    # Example: Get recommendations for a movie
    test_movie = "The Shawshank Redemption"
    print(f"\nContent-based recommendations for '{test_movie}':")
    content_recs = recommender.recommend_similar_movies(test_movie)
    print(content_recs)
    
    print(f"\nCollaborative filtering recommendations for '{test_movie}':")
    cf_recs = recommender.collaborative_filtering(test_movie)
    print(cf_recs)
    
    print(f"\nHybrid recommendations for '{test_movie}':")
    hybrid_recs = recommender.hybrid_recommendations(test_movie)
    print(hybrid_recs) 