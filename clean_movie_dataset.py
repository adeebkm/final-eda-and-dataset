import pandas as pd
import numpy as np
import re
from ast import literal_eval

def clean_dataset(input_file='movie_dataset.csv', output_file='movie_dataset_clean.csv'):
    """Clean and preprocess the movie dataset
    
    Args:
        input_file (str): Path to the raw dataset
        output_file (str): Path to save the cleaned dataset
        
    Returns:
        pandas.DataFrame: Cleaned dataset
    """
    print(f"Loading dataset from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Dataset file '{input_file}' not found.")
        return None
    
    print(f"Original dataset shape: {df.shape}")
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Fill missing values
    print("Filling missing values...")
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna('')
        else:
            df_clean[col] = df_clean[col].fillna(0)
    
    # Ensure title column exists
    if 'title' not in df_clean.columns:
        if 'title_x' in df_clean.columns:
            df_clean['title'] = df_clean['title_x']
        elif 'name' in df_clean.columns:
            df_clean['title'] = df_clean['name']
        else:
            print("Warning: No title column found in dataset")
    
    # Create a clean year column if it doesn't exist
    if 'year' not in df_clean.columns:
        if 'release_date' in df_clean.columns:
            df_clean['year'] = pd.to_datetime(
                df_clean['release_date'], 
                errors='coerce'
            ).dt.year
        elif 'release_year' in df_clean.columns:
            df_clean['year'] = df_clean['release_year']
    
    # Ensure genres are in a consistent format
    if 'genres' in df_clean.columns:
        # Handle different genre formats
        def format_genres(row):
            genres = row.get('genres', '')
            
            # If genres is a string representation of a list/dict
            if isinstance(genres, str) and (genres.startswith('[') or genres.startswith('{')):
                try:
                    # Try to parse as literal eval (list of dicts or list of strings)
                    parsed = literal_eval(genres)
                    
                    # Handle list of dictionaries format
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                        return '|'.join([g.get('name', '') for g in parsed if g.get('name')])
                    # Handle list of strings format
                    elif isinstance(parsed, list):
                        return '|'.join(parsed)
                except:
                    # If parsing fails, leave as is
                    return genres
            
            # Return the original if no transformation needed
            return genres
        
        df_clean['genres'] = df_clean.apply(format_genres, axis=1)
        
        # Create a list version of genres for easier processing
        df_clean['genres_list'] = df_clean['genres'].apply(
            lambda x: x.split('|') if isinstance(x, str) and x else []
        )
    
    # Create content features for recommendation
    print("Creating content features...")
    
    # Get relevant columns for content features
    feature_cols = []
    for col in ['overview', 'plot', 'summary', 'description', 'synopsis']:
        if col in df_clean.columns:
            feature_cols.append(col)
    
    # Add keywords if available
    if 'keywords' in df_clean.columns:
        feature_cols.append('keywords')
    
    # Add cast information if available
    for col in ['cast', 'actors', 'stars']:
        if col in df_clean.columns:
            feature_cols.append(col)
    
    # Add director information if available
    if 'director' in df_clean.columns:
        feature_cols.append('director')
    
    # Add genres
    if 'genres' in df_clean.columns:
        feature_cols.append('genres')
    
    # Create content features by combining all relevant text
    if feature_cols:
        df_clean['content_features'] = df_clean[feature_cols].apply(
            lambda row: ' '.join([str(row[col]) for col in feature_cols if pd.notna(row[col])]),
            axis=1
        )
    else:
        # If no content columns available, use title and year as fallback
        df_clean['content_features'] = df_clean['title'] + ' ' + df_clean['year'].astype(str)
        print("Warning: No content columns found. Using title and year as content features.")
    
    # Ensure rating column exists
    if 'rating' not in df_clean.columns and 'vote_average' in df_clean.columns:
        df_clean['rating'] = df_clean['vote_average']
    
    # Save cleaned dataset
    print(f"Saving cleaned dataset to {output_file}...")
    df_clean.to_csv(output_file, index=False)
    print(f"Cleaned dataset shape: {df_clean.shape}")
    
    return df_clean

if __name__ == "__main__":
    # Clean the dataset when run as a script
    clean_dataset() 