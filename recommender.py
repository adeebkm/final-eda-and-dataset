import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import re
import ast
from clean_movie_dataset import clean_dataset

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
    
    def get_content_based_recommendations(self, movie_title, n_recommendations=5):
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
            ['title', 'year', 'genres', 'rating', 'director']
        ].copy()
        
        # Add similarity score
        similarity_scores = 1 - distances.flatten()[1:]  # Convert distance to similarity
        recommendations['score'] = similarity_scores
        
        return recommendations.sort_values('score', ascending=False)
    
    def get_collaborative_recommendations(self, movie_title, n_recommendations=5):
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
        
        # Parse genres if stored as string representation of list
        if isinstance(ref_movie.get('genres_list'), str):
            try:
                ref_genres = set(ast.literal_eval(ref_movie['genres_list']))
            except:
                ref_genres = set(ref_movie.get('genres', '').split('|')) if isinstance(ref_movie.get('genres'), str) else set()
        else:
            ref_genres = set(ref_movie.get('genres_list', [])) if isinstance(ref_movie.get('genres_list'), list) else set()
            if not ref_genres and isinstance(ref_movie.get('genres'), str):
                ref_genres = set(ref_movie['genres'].split('|'))
        
        # Create a subset of movies excluding the reference movie
        other_movies = self.df[self.df.index != movie_idx].copy()
        
        # Filter for movies with ratings
        other_movies = other_movies.dropna(subset=['rating'])
        
        # Calculate genre similarity - proportion of genres that match
        def genre_similarity(movie_row):
            movie_genres = []
            
            # Try to get genres from different possible formats
            if isinstance(movie_row.get('genres_list'), list):
                movie_genres = movie_row['genres_list']
            elif isinstance(movie_row.get('genres_list'), str):
                try:
                    movie_genres = ast.literal_eval(movie_row['genres_list'])
                except:
                    pass
            
            if not movie_genres and isinstance(movie_row.get('genres'), str):
                movie_genres = movie_row['genres'].split('|')
            
            movie_genres_set = set(movie_genres)
            if not movie_genres_set or not ref_genres:
                return 0
            
            return len(ref_genres.intersection(movie_genres_set)) / len(ref_genres.union(movie_genres_set))
        
        # Calculate genre similarities
        other_movies['genre_similarity'] = other_movies.apply(genre_similarity, axis=1)
        
        # Calculate rating similarity (penalize large rating differences)
        if pd.notna(ref_movie.get('rating')):
            other_movies['rating_diff'] = abs(other_movies['rating'] - ref_movie['rating'])
            max_rating_diff = other_movies['rating_diff'].max() if not other_movies.empty else 1
            other_movies['rating_similarity'] = 1 - (other_movies['rating_diff'] / max_rating_diff)
        else:
            other_movies['rating_similarity'] = 0.5  # Neutral if reference rating is missing
        
        # Calculate overall similarity score (weighted average)
        other_movies['score'] = (
            (0.6 * other_movies['genre_similarity']) + 
            (0.4 * other_movies['rating_similarity'])
        )
        
        # Get top recommendations
        recommendations = other_movies.sort_values(
            'score', ascending=False
        ).head(n_recommendations)[
            ['title', 'year', 'genres', 'rating', 'director', 'score']
        ]
        
        return recommendations
    
    def get_hybrid_recommendations(self, movie_title, n_recommendations=5):
        """Hybrid recommendation combining content-based and collaborative filtering
        
        Args:
            movie_title (str): Title of the reference movie
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            pandas.DataFrame: Dataframe of recommended movies
        """
        # Get content-based recommendations
        content_recs = self.get_content_based_recommendations(movie_title, n_recommendations*2)
        if content_recs.empty:
            return pd.DataFrame()
        
        # Get collaborative filtering recommendations
        cf_recs = self.get_collaborative_recommendations(movie_title, n_recommendations*2)
        
        # Merge recommendations
        # Convert to dictionaries for faster lookup
        cf_scores = dict(zip(cf_recs['title'], cf_recs['score']))
        
        # Add CF score to content recommendations
        content_recs['cf_score'] = content_recs['title'].apply(
            lambda x: cf_scores.get(x, 0)
        )
        
        # Calculate hybrid score (weighted average)
        content_recs['hybrid_score'] = (
            (0.7 * content_recs['score']) + 
            (0.3 * content_recs['cf_score'])
        )
        
        # Get top hybrid recommendations
        hybrid_recs = content_recs.sort_values(
            'hybrid_score', ascending=False
        ).head(n_recommendations)[
            ['title', 'year', 'genres', 'rating', 'director', 'hybrid_score']
        ]
        
        # Rename hybrid_score to score for consistency with other methods
        hybrid_recs = hybrid_recs.rename(columns={'hybrid_score': 'score'})
        
        return hybrid_recs

if __name__ == "__main__":
    # Create recommender instance
    recommender = MovieRecommender()
    
    # Build models
    recommender.build_knn_model()
    
    # Example: Get recommendations for a movie
    test_movie = "The Shawshank Redemption"
    print(f"\nContent-based recommendations for '{test_movie}':")
    content_recs = recommender.get_content_based_recommendations(test_movie)
    print(content_recs)
    
    print(f"\nCollaborative filtering recommendations for '{test_movie}':")
    cf_recs = recommender.get_collaborative_recommendations(test_movie)
    print(cf_recs)
    
    print(f"\nHybrid recommendations for '{test_movie}':")
    hybrid_recs = recommender.get_hybrid_recommendations(test_movie)
    print(hybrid_recs) 