document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchResults = document.getElementById('search-results');
    const selectedMovieInput = document.getElementById('selected-movie');
    const getRecommendationsBtn = document.getElementById('get-recommendations');
    const numRecommendations = document.getElementById('num-recommendations');
    const methodRadios = document.querySelectorAll('input[name="method"]');
    const recommendationsContainer = document.getElementById('recommendations-container');
    const recommendationsDiv = document.getElementById('recommendations');
    const movieTitleSpan = document.getElementById('movie-title');
    const methodUsedSpan = document.getElementById('method-used');
    const loadingDiv = document.getElementById('loading');
    const errorContainer = document.getElementById('error-container');
    const movieCardTemplate = document.getElementById('movie-card-template');
    
    // Event Listeners
    searchButton.addEventListener('click', handleSearch);
    searchInput.addEventListener('keyup', event => {
        if (event.key === 'Enter') {
            handleSearch();
        }
    });
    
    getRecommendationsBtn.addEventListener('click', getRecommendations);
    
    // When a movie is selected, enable the get recommendations button
    selectedMovieInput.addEventListener('input', () => {
        getRecommendationsBtn.disabled = selectedMovieInput.value.trim() === '';
    });
    
    // Initialize with popular movies
    fetchPopularMovies();
    
    // Methods
    function fetchPopularMovies() {
        // Fetch all movies
        fetch('/movies')
            .then(response => response.json())
            .then(movies => {
                // Display some popular movies as suggestions
                const popularMovies = [
                    "The Shawshank Redemption",
                    "The Godfather",
                    "The Dark Knight",
                    "Pulp Fiction",
                    "Forrest Gump"
                ];
                
                // Filter out movies that actually exist in our database
                const availablePopularMovies = popularMovies.filter(
                    movie => movies.some(m => m.toLowerCase().includes(movie.toLowerCase()))
                );
                
                if (availablePopularMovies.length > 0) {
                    searchResults.innerHTML = '<div class="list-group-item disabled">Popular movies:</div>';
                    availablePopularMovies.forEach(movie => {
                        // Find the exact movie in our dataset
                        const exactMovie = movies.find(m => 
                            m.toLowerCase() === movie.toLowerCase() || 
                            m.toLowerCase().includes(movie.toLowerCase())
                        );
                        
                        if (exactMovie) {
                            const listItem = document.createElement('button');
                            listItem.className = 'list-group-item list-group-item-action';
                            listItem.textContent = exactMovie;
                            listItem.addEventListener('click', () => selectMovie(exactMovie));
                            searchResults.appendChild(listItem);
                        }
                    });
                }
            })
            .catch(error => {
                console.error('Error fetching popular movies:', error);
            });
    }
    
    function handleSearch() {
        const query = searchInput.value.trim();
        if (query === '') {
            return;
        }
        
        // Clear previous results
        searchResults.innerHTML = '';
        
        // Show loading indicator in the search results
        searchResults.innerHTML = '<div class="list-group-item text-center"><div class="spinner-border spinner-border-sm" role="status"></div> Searching...</div>';
        
        // Fetch search results
        fetch(`/search?query=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(results => {
                // Clear loading indicator
                searchResults.innerHTML = '';
                
                if (results.length === 0) {
                    searchResults.innerHTML = '<div class="list-group-item">No movies found. Try a different search term.</div>';
                    return;
                }
                
                // Display results
                results.forEach(movie => {
                    const listItem = document.createElement('button');
                    listItem.className = 'list-group-item list-group-item-action';
                    
                    // Create a more detailed display for each movie
                    const titleElem = document.createElement('div');
                    titleElem.className = 'fw-bold';
                    titleElem.textContent = movie.title;
                    
                    listItem.appendChild(titleElem);
                    
                    // Add year and genres if available
                    if (movie.year || movie.genres) {
                        const details = document.createElement('small');
                        details.className = 'text-muted d-block';
                        
                        let detailsText = '';
                        if (movie.year) {
                            detailsText += movie.year;
                        }
                        
                        if (movie.genres) {
                            if (detailsText) {
                                detailsText += ' • ';
                            }
                            detailsText += movie.genres;
                        }
                        
                        details.textContent = detailsText;
                        listItem.appendChild(details);
                    }
                    
                    listItem.addEventListener('click', () => selectMovie(movie.title));
                    searchResults.appendChild(listItem);
                });
            })
            .catch(error => {
                console.error('Error searching for movies:', error);
                searchResults.innerHTML = '<div class="list-group-item text-danger">Error searching for movies. Please try again.</div>';
            });
    }
    
    function selectMovie(title) {
        selectedMovieInput.value = title;
        searchResults.innerHTML = ''; // Clear search results
        getRecommendationsBtn.disabled = false;
    }
    
    function getRecommendations() {
        const movie = selectedMovieInput.value.trim();
        if (movie === '') {
            return;
        }
        
        // Get selected method
        let method = 'hybrid';
        methodRadios.forEach(radio => {
            if (radio.checked) {
                method = radio.value;
            }
        });
        
        // Show loading indicator
        loadingDiv.classList.remove('d-none');
        recommendationsContainer.classList.add('d-none');
        errorContainer.classList.add('d-none');
        
        // Prepare request data
        const requestData = {
            movie: movie,
            method: method,
            num: parseInt(numRecommendations.value)
        };
        
        // Fetch recommendations
        fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                loadingDiv.classList.add('d-none');
                
                if (data.error) {
                    // Show error with more details
                    errorContainer.innerHTML = `<strong>Error:</strong> ${data.error}`;
                    errorContainer.classList.remove('d-none');
                    
                    // Scroll to error
                    errorContainer.scrollIntoView({ behavior: 'smooth' });
                    return;
                }
                
                // Display recommendations
                displayRecommendations(data);
            })
            .catch(error => {
                console.error('Error getting recommendations:', error);
                loadingDiv.classList.add('d-none');
                errorContainer.innerHTML = '<strong>Error:</strong> Unable to get recommendations. Please try again later.';
                errorContainer.classList.remove('d-none');
                
                // Scroll to error
                errorContainer.scrollIntoView({ behavior: 'smooth' });
            });
    }
    
    function displayRecommendations(data) {
        // Set movie title and method
        movieTitleSpan.textContent = data.movie;
        
        // Set method name with proper formatting
        let methodName = data.method;
        if (methodName === 'content') {
            methodName = 'Content-Based Filtering';
        } else if (methodName === 'collaborative') {
            methodName = 'Collaborative Filtering';
        } else {
            methodName = 'Hybrid';
        }
        methodUsedSpan.textContent = methodName;
        
        // Clear previous recommendations
        recommendationsDiv.innerHTML = '';
        
        // Create a card for each recommendation
        data.recommendations.forEach(movie => {
            const movieCard = document.importNode(movieCardTemplate.content, true);
            
            // Fill in movie data
            movieCard.querySelector('.movie-title').textContent = movie.title;
            movieCard.querySelector('.movie-year').textContent = movie.year || 'Unknown Year';
            movieCard.querySelector('.movie-genres').textContent = movie.genres || 'No genre information';
            
            const ratingElement = movieCard.querySelector('.movie-rating');
            if (movie.rating) {
                ratingElement.textContent = movie.rating.toFixed(1);
            } else {
                ratingElement.parentElement.textContent = 'No rating available';
            }
            
            // Format score as percentage
            movieCard.querySelector('.movie-score').textContent = movie.score ? movie.score.toFixed(1) + '%' : 'N/A';
            movieCard.querySelector('.movie-director').textContent = movie.director || 'Unknown';
            
            // Handle sentiment information if available
            const sentimentContainer = movieCard.querySelector('.sentiment-container');
            const sentimentValue = movieCard.querySelector('.movie-sentiment');
            
            if (movie.sentiment && data.method === 'collaborative') {
                sentimentValue.textContent = movie.sentiment.toFixed(1) + '%';
                sentimentContainer.classList.remove('d-none');
            } else {
                sentimentContainer.classList.add('d-none');
            }
            
            recommendationsDiv.appendChild(movieCard);
        });
        
        // Show recommendations container
        recommendationsContainer.classList.remove('d-none');
        
        // Scroll to recommendations
        recommendationsContainer.scrollIntoView({ behavior: 'smooth' });
    }
}); 