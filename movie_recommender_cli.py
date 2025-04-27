#!/usr/bin/env python3

import argparse
import pandas as pd
from movie_recommendation import MovieRecommender

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Movie Recommendation System CLI')
    parser.add_argument('--movie', type=str, help='Movie title to get recommendations for')
    parser.add_argument('--method', type=str, default='hybrid', 
                       choices=['content', 'collaborative', 'hybrid'],
                       help='Recommendation method to use')
    parser.add_argument('--num', type=int, default=5, help='Number of recommendations to show')
    parser.add_argument('--list-movies', action='store_true', help='List available movies')
    parser.add_argument('--search', type=str, help='Search for a movie by partial title')
    
    args = parser.parse_args()
    
    # Initialize recommender
    print("Initializing movie recommender...")
    recommender = MovieRecommender()
    
    # List movies if requested
    if args.list_movies:
        print("\nAvailable movies:")
        for title in sorted(recommender.df['title'].unique()):
            print(f"- {title}")
        return
    
    # Search for movies if requested
    if args.search:
        search_term = args.search.lower()
        matches = recommender.df[recommender.df['title'].str.lower().str.contains(search_term)]
        
        if matches.empty:
            print(f"No movies found matching '{args.search}'")
        else:
            print(f"\nMovies matching '{args.search}':")
            for title in sorted(matches['title'].unique()):
                print(f"- {title}")
        return
    
    # Require movie title for recommendations
    if not args.movie:
        parser.error("--movie is required when not using --search or --list-movies")
    
    # Build the KNN model (also builds content model)
    if args.method in ['content', 'hybrid']:
        recommender.build_knn_model()
    
    # Get recommendations based on specified method
    movie_title = args.movie
    n_recommendations = args.num
    
    if args.method == 'content':
        print(f"\nGetting content-based recommendations for '{movie_title}'...")
        recommendations = recommender.recommend_similar_movies(movie_title, n_recommendations)
        score_col = 'similarity_score'
    
    elif args.method == 'collaborative':
        print(f"\nGetting collaborative filtering recommendations for '{movie_title}'...")
        recommendations = recommender.collaborative_filtering(movie_title, n_recommendations)
        score_col = 'cf_score'
    
    else:  # hybrid
        print(f"\nGetting hybrid recommendations for '{movie_title}'...")
        recommendations = recommender.hybrid_recommendations(movie_title, n_recommendations)
        score_col = 'hybrid_score'
    
    # Display recommendations
    if recommendations.empty:
        print(f"Could not generate recommendations for '{movie_title}'")
        print("Use --search to find a valid movie title")
    else:
        print("\nTop recommendations:")
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"{i}. {row['title']} ({row['year']}) - {row['genres_clean']}")
            print(f"   Rating: {row['combined_rating']:.1f}/10")
            print(f"   Director: {row['director_clean']}")
            print(f"   Score: {row[score_col]:.4f}")
            print()

if __name__ == "__main__":
    main() 