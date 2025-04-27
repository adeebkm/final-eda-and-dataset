from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import traceback
import numpy as np
import json
from movie_recommendation import MovieRecommender

# Add a custom JSON encoder to handle NaN values
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NpEncoder

# Initialize the recommender system
recommender = MovieRecommender()
print("Building recommendation models...")
recommender.build_knn_model()  # This also builds the content model

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/search')
def search():
    search_term = request.args.get('query', '').lower()
    if not search_term:
        return jsonify([])
    
    # First look for exact matches
    exact_matches = recommender.df[recommender.df['title'].str.lower() == search_term]
    
    # Then look for partial matches
    partial_matches = recommender.df[
        ~recommender.df['title'].str.lower().isin(exact_matches['title'].str.lower()) &
        recommender.df['title'].str.lower().str.contains(search_term, regex=False)
    ]
    
    # Combine results with exact matches first
    combined_results = pd.concat([exact_matches, partial_matches])
    
    # Limit to 10 results
    results = combined_results.head(10)
    
    # Return more detailed movie information
    movie_list = []
    for _, movie in results.iterrows():
        movie_info = {
            'title': movie['title'],
            'year': movie.get('year', ''),
            'genres': movie.get('genres_clean', movie.get('genres', '')),
            'rating': float(movie.get('combined_rating', 0)) if pd.notna(movie.get('combined_rating')) else None
        }
        movie_list.append(movie_info)
    
    return jsonify(movie_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie_title = data.get('movie', '')
    method = data.get('method', 'hybrid')
    num = int(data.get('num', 5))
    
    print(f"Recommendation request: movie='{movie_title}', method='{method}', num={num}")
    
    if not movie_title:
        return jsonify({'error': 'No movie title provided'})
    
    try:
        # First try exact match
        movie_row = recommender.df[recommender.df['title'].str.lower() == movie_title.lower()]
        
        # If no exact match, try partial match
        if movie_row.empty:
            print(f"No exact match for '{movie_title}', trying partial match...")
            movie_row = recommender.df[recommender.df['title'].str.lower().str.contains(movie_title.lower(), regex=False)]
        
        if movie_row.empty:
            print(f"Movie '{movie_title}' not found in database")
            return jsonify({'error': f'Movie "{movie_title}" not found in our database'})
        
        # Get the first matching movie if multiple matches
        movie_row = movie_row.iloc[0]
        exact_title = movie_row['title']
        print(f"Found movie: '{exact_title}'")
        
        # Get recommendations based on method
        print(f"Getting {method} recommendations...")
        
        try:
            if method == 'content':
                print("Calling recommend_similar_movies...")
                recommendations = recommender.recommend_similar_movies(exact_title, num)
                print(f"Got {len(recommendations)} content recommendations")
            elif method == 'collaborative':
                print("Calling collaborative_filtering...")
                # Add detailed logging for collaborative filtering
                try:
                    # First check if the plot_clean column exists
                    if 'plot_clean' not in recommender.df.columns:
                        print("WARNING: 'plot_clean' column not found in dataset. This may affect sentiment analysis.")
                    
                    # Get the movie and check if it has a plot
                    movie_row = recommender.df[recommender.df['title'] == exact_title].iloc[0]
                    if pd.isna(movie_row.get('plot_clean', None)) or movie_row.get('plot_clean', '') == '':
                        print(f"WARNING: Movie '{exact_title}' has no plot content for sentiment analysis")
                    
                    # Print all column names for debugging
                    print(f"Dataset columns: {recommender.df.columns.tolist()}")
                    
                    # Continue with regular collaborative filtering
                    recommendations = recommender.collaborative_filtering(exact_title, num)
                    print(f"Got {len(recommendations)} collaborative recommendations")
                    print(f"Collaborative columns: {recommendations.columns.tolist()}")
                except Exception as collab_error:
                    print(f"Detailed collaborative filtering error: {str(collab_error)}")
                    print(traceback.format_exc())
                    raise
            else:  # hybrid
                print("Calling hybrid_recommendations...")
                recommendations = recommender.hybrid_recommendations(exact_title, num)
                print(f"Got {len(recommendations)} hybrid recommendations")
                
            print(f"Recommendations columns: {recommendations.columns.tolist()}")
        except Exception as method_error:
            print(f"Error in recommendation method: {str(method_error)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error in recommendation method: {str(method_error)}'})
        
        # Format the response
        result = {
            'movie': exact_title,
            'method': method,
            'recommendations': []
        }
        
        # Add detailed information for each recommendation
        try:
            for _, rec in recommendations.iterrows():
                print(f"Processing recommendation: {rec['title']}")
                
                # Handle each field safely, replacing NaN with None
                year = None if pd.isna(rec.get('year')) else rec.get('year', '')
                
                # Handle genres (check for both genres_clean and genres columns)
                genres = None
                if 'genres_clean' in rec and not pd.isna(rec['genres_clean']):
                    genres = rec['genres_clean']
                elif 'genres' in rec and not pd.isna(rec['genres']):
                    genres = rec['genres']
                
                # Handle director (check for both director_clean and director columns)
                director = None
                if 'director_clean' in rec and not pd.isna(rec['director_clean']):
                    director = rec['director_clean']
                elif 'director' in rec and not pd.isna(rec['director']):
                    director = rec['director']
                
                # Handle rating with extra care (check various rating column names)
                rating = None
                for rating_col in ['combined_rating', 'rating', 'vote_average']:
                    if rating_col in rec and not pd.isna(rec[rating_col]):
                        try:
                            rating = float(rec[rating_col])
                            break
                        except:
                            continue
                
                # Handle different score column names
                score = None
                for score_col in ['score', 'similarity_score', 'hybrid_score', 'cf_score']:
                    if score_col in rec and not pd.isna(rec[score_col]):
                        try:
                            # Convert score to percentage (0-100 range)
                            score = float(rec[score_col]) * 100
                            break
                        except:
                            continue
                
                # Add sentiment similarity if available
                sentiment_similarity = None
                if 'sentiment_similarity' in rec and not pd.isna(rec['sentiment_similarity']):
                    try:
                        sentiment_similarity = float(rec['sentiment_similarity']) * 100
                    except:
                        pass
                
                rec_info = {
                    'title': rec['title'],
                    'year': year,
                    'genres': genres,
                    'director': director,
                    'rating': rating,
                    'score': score,
                    'sentiment': sentiment_similarity
                }
                
                result['recommendations'].append(rec_info)
        except Exception as format_error:
            print(f"Error formatting recommendations: {str(format_error)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error formatting recommendations: {str(format_error)}'})
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Error getting recommendations: {str(e)}'})

@app.route('/movies')
def get_all_movies():
    """Return all movie titles in the dataset for the frontend to use"""
    return jsonify(recommender.df['title'].tolist())

if __name__ == '__main__':
    app.run(debug=True, port=8080) 