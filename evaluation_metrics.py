import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
import os
import random

class RecommendationEvaluator:
    def __init__(self, dataset_path: str = 'movie_dataset_clean.csv'):
        """Initialize the evaluator with the dataset."""
        self.df = pd.read_csv(dataset_path)
        self.metrics = {}
        
    def calculate_diversity(self, recommendations: List[str]) -> float:
        """Calculate the diversity of recommendations based on genres.
        
        Diversity is measured as the ratio of unique genres to the number of recommendations.
        Higher values (closer to 1.0) indicate more diverse recommendations across different genres.
        
        Args:
            recommendations: List of movie titles that were recommended
            
        Returns:
            float: Diversity score between 0.0 and potentially >1.0 (if many genres per movie)
        """
        genres = set()
        for movie in recommendations:
            movie_genres = self.df[self.df['title'] == movie]['genres'].iloc[0].split('|')
            genres.update(movie_genres)
        return len(genres) / len(recommendations)
    
    def calculate_novelty(self, recommendations: List[str], user_history: List[str]) -> float:
        """Calculate how novel the recommendations are compared to user history.
        
        Novelty measures how many recommended movies are new to the user (not in their history).
        A score of 1.0 means all recommendations are new, while 0.0 means all have been seen before.
        
        Args:
            recommendations: List of movie titles that were recommended
            user_history: List of movie titles the user has already seen
            
        Returns:
            float: Novelty score between 0.0 and 1.0
        """
        if not user_history:
            return 1.0
        novel_count = sum(1 for movie in recommendations if movie not in user_history)
        return novel_count / len(recommendations)
    
    def calculate_serendipity(self, recommendations: List[str], 
                            user_history: List[str], 
                            similarity_threshold: float = 0.5) -> float:
        """Calculate how surprising but relevant the recommendations are.
        
        Serendipity measures recommendations that are novel (not in history) but still relevant
        (similar enough to user's preferences). It identifies "pleasant surprises" - recommendations
        that users wouldn't have found on their own but will enjoy.
        
        Args:
            recommendations: List of movie titles that were recommended
            user_history: List of movie titles the user has already seen
            similarity_threshold: Minimum Jaccard similarity of genres to consider relevant
            
        Returns:
            float: Serendipity score between 0.0 and 1.0
        """
        if not user_history:
            return 0.0
        
        serendipitous_count = 0
        for movie in recommendations:
            if movie not in user_history:
                # Check if the movie is somewhat similar to user's history
                movie_genres = set(self.df[self.df['title'] == movie]['genres'].iloc[0].split('|'))
                for hist_movie in user_history:
                    hist_genres = set(self.df[self.df['title'] == hist_movie]['genres'].iloc[0].split('|'))
                    # Jaccard similarity = intersection / union
                    similarity = len(movie_genres.intersection(hist_genres)) / len(movie_genres.union(hist_genres))
                    if similarity >= similarity_threshold:
                        serendipitous_count += 1
                        break
        
        return serendipitous_count / len(recommendations)
    
    def evaluate_recommendations(self, 
                               recommendations: List[List[str]], 
                               user_histories: List[List[str]]) -> Dict:
        """Evaluate a set of recommendations using multiple metrics.
        
        This method calculates diversity, novelty, and serendipity for each recommendation set,
        then aggregates the results with means and standard deviations.
        
        Args:
            recommendations: List of recommendation sets (each set is a list of movie titles)
            user_histories: List of user histories (each history is a list of movie titles)
            
        Returns:
            Dict: Results containing individual and aggregated metrics
        """
        results = {
            'diversity': [],
            'novelty': [],
            'serendipity': []
        }
        
        for recs, history in zip(recommendations, user_histories):
            results['diversity'].append(self.calculate_diversity(recs))
            results['novelty'].append(self.calculate_novelty(recs, history))
            results['serendipity'].append(self.calculate_serendipity(recs, history))
        
        # Calculate averages
        for metric in ['diversity', 'novelty', 'serendipity']:
            results[f'avg_{metric}'] = np.mean(results[metric])
            results[f'std_{metric}'] = np.std(results[metric])
        
        return results
    
    def plot_metrics(self, results: Dict, save_path: str = 'evaluation_metrics.png'):
        """Plot the evaluation metrics as a bar chart with error bars.
        
        Args:
            results: Dictionary containing the evaluation results
            save_path: Path where to save the plot image
        """
        metrics = ['diversity', 'novelty', 'serendipity']
        values = [results[f'avg_{metric}'] for metric in metrics]
        errors = [results[f'std_{metric}'] for metric in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, yerr=errors, capsize=5)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title('Recommendation System Evaluation Metrics')
        plt.ylabel('Score (0-1 scale)')
        plt.ylim(0, 1)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig(save_path)
        plt.close()
    
    def save_results(self, results: Dict, save_path: str = 'evaluation_results.json'):
        """Save evaluation results to a JSON file.
        
        Args:
            results: Dictionary containing the evaluation results
            save_path: Path where to save the JSON file
        """
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    def generate_synthetic_recommendations(self, num_users: int = 20, rec_size: int = 5, 
                                         history_size: int = 3) -> Tuple[List[List[str]], List[List[str]]]:
        """Generate synthetic recommendations and user histories for testing.
        
        This creates representative test data to evaluate the metrics calculation.
        
        Args:
            num_users: Number of synthetic users to generate
            rec_size: Number of recommendations per user
            history_size: Number of movies in each user's history
            
        Returns:
            Tuple[List[List[str]], List[List[str]]]: Recommendations and user histories
        """
        # Get a list of available movie titles
        available_movies = self.df['title'].tolist()
        
        # Generate synthetic recommendations and user histories
        recommendations = []
        user_histories = []
        
        for _ in range(num_users):
            # Generate user history
            history = random.sample(available_movies, min(history_size, len(available_movies)))
            user_histories.append(history)
            
            # Generate recommendations
            # Ensure recommendations don't include all history items to maintain novelty
            remaining_movies = [m for m in available_movies if m not in history]
            recs = random.sample(remaining_movies, min(rec_size, len(remaining_movies)))
            recommendations.append(recs)
            
        return recommendations, user_histories

def main():
    # Initialize evaluator
    evaluator = RecommendationEvaluator()
    
    # Generate synthetic recommendations for testing
    # Generates 20 users, each with 5 recommendations and 3 history items
    recommendations, user_histories = evaluator.generate_synthetic_recommendations(num_users=20, rec_size=5, history_size=3)
    
    # Print summary of synthetic data
    total_movies = len(set(movie for rec_list in recommendations for movie in rec_list))
    print(f"Generated recommendations for {len(recommendations)} users")
    print(f"Total unique movies in recommendations: {total_movies}")
    
    # Evaluate recommendations
    results = evaluator.evaluate_recommendations(recommendations, user_histories)
    
    # Plot and save results
    evaluator.plot_metrics(results)
    evaluator.save_results(results)
    
    # Print results with explanations
    print("\nEvaluation Results:")
    print(f"Average Diversity: {results['avg_diversity']:.2f} ± {results['std_diversity']:.2f}")
    print("  (Higher diversity means more variety in genres across recommendations)")
    
    print(f"Average Novelty: {results['avg_novelty']:.2f} ± {results['std_novelty']:.2f}")
    print("  (Higher novelty means more recommended movies are new to users)")
    
    print(f"Average Serendipity: {results['avg_serendipity']:.2f} ± {results['std_serendipity']:.2f}")
    print("  (Higher serendipity means more recommendations are surprising but relevant)")

if __name__ == "__main__":
    main() 