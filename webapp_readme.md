# Movie Recommendation Web Application

This is a web interface for the movie recommendation system, allowing users to get movie recommendations using three different approaches: content-based filtering, collaborative filtering with sentiment analysis, and a hybrid approach.

## Features

- **Movie Search**: Search for movies by title with instant results
- **Multiple Recommendation Methods**:
  - **Content-Based Filtering**: Uses movie features like plot, genres, directors, and cast
  - **Collaborative Filtering with Sentiment Analysis**: Uses genre similarities, rating patterns, and emotional tone matching
  - **Hybrid Approach**: Combines both methods for better recommendations
- **Configurable Options**: Choose the number of recommendations to display (3, 5, or 10)
- **Responsive Design**: Works on desktop and mobile devices
- **Detailed Movie Information**: View year, genres, directors, ratings, and similarity scores
- **Sentiment Matching**: See how well the emotional tone matches between movies (collaborative filtering only)
- **Educational Information**: Learn about each recommendation algorithm

## How It Works

The application has a Flask backend that integrates with the movie recommendation system. The frontend is built with HTML, CSS, and vanilla JavaScript, utilizing Bootstrap for styling.
 
### Key Components:

1. **Backend (Flask)**:
   - `app.py`: Main Flask application with API endpoints
   - Integration with the existing recommendation engine
   - JSON serialization with proper handling of NaN values
   - Robust error handling for all endpoints
   - Adaptive column selection for different dataset formats

2. **Frontend**:
   - `index.html`: Main page with search, selection, and result display
   - `style.css`: Custom styling for the application
   - `script.js`: Frontend functionality and API interaction

3. **Features in Detail**:
   - **Search Functionality**: Real-time search results with both exact and partial matching
   - **Method Selection**: Choose between three recommendation algorithms
   - **Recommendation Results**: Detailed movie cards with all relevant information
   - **Sentiment Analysis**: Mood matching based on plot sentiment analysis (collaborative filtering)
   - **Similarity Scores**: All similarity scores presented as percentages

## Installation & Setup

1. Install required dependencies:
   ```bash
   pip install flask pandas scikit-learn textblob
   ```

2. Download NLTK data required by TextBlob:
   ```bash
   python -m textblob.download_corpora
   ```

3. Ensure you have the movie dataset and recommendation system files:
   - `movie_dataset_clean.csv` (or the original that will be cleaned on first run)
   - `movie_recommendation.py`
   - `clean_movie_dataset.py`

4. Start the web server:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

## Usage Guide

1. **Search for a movie**:
   - Type a movie title in the search box
   - Select from the displayed results

2. **Choose recommendation settings**:
   - Select the recommendation method:
     - **Content-Based**: Finding movies with similar content features
     - **Collaborative**: Finding movies with similar genres, ratings, and emotional tone
     - **Hybrid**: Combining both approaches
   - Choose the number of recommendations (3, 5, or 10)

3. **Get Recommendations**:
   - Click the "Get Recommendations" button
   - View the recommended movies
   - Each card shows the movie's title, year, genres, rating, director, and similarity score
   - For collaborative filtering, a mood match percentage is also displayed

## Technical Details

The web application passes your movie selection to the recommendation system, which:

1. For content-based filtering:
   - Analyzes the movie's plot, genres, directors, and cast
   - Uses TF-IDF vectorization and KNN to find similar movies
   - Returns movies with similar content characteristics
   
2. For collaborative filtering with sentiment analysis:
   - Finds movies with similar genres (50% weight)
   - Finds movies with similar ratings (30% weight)
   - Uses TextBlob to analyze plot sentiment and match emotional tone (20% weight)
   - Combines these factors for a balanced recommendation
   
3. For hybrid recommendations:
   - Combines the scores from both methods (70% content, 30% collaborative)
   - Returns a balanced set of recommendations

## Sentiment Analysis Details

The sentiment analysis feature:
- Analyzes the emotional tone of movie plots using TextBlob
- Calculates a polarity score between -1 (very negative) and 1 (very positive)
- Compares the sentiment direction and magnitude between movies
- Gives higher similarity to movies with matching emotional tones
- Adapts gracefully if plot data is unavailable

## Error Handling

The application includes robust error handling:
- Graceful degradation if certain data is missing
- Adaptive column selection based on available data
- Clear error messages for troubleshooting
- Proper handling of NaN values and missing data
- Fallback strategies when sentiment analysis isn't possible

## Future Enhancements

- User accounts and personalized recommendations
- More detailed movie information and external links
- Movie posters and visual content
- User feedback on recommendations
- More advanced filtering options
- Advanced NLP for deeper content understanding
- Diversity and novelty metrics for recommendations 