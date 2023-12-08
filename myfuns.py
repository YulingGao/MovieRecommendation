import pandas as pd
import requests
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

######################

ratings = pd.read_csv('ratings.csv', sep=',', index_col=0)

mean_ratings = ratings.mean()

mean_ratings_df = pd.DataFrame({'movie_id': mean_ratings.index, 'rating': mean_ratings.values})

mean_ratings_df = mean_ratings_df.sort_values(by='rating', ascending=False)

mean_ratings_df['movie_id'] = mean_ratings_df['movie_id'].str.replace('m', '').astype(int)

#####################
#####################
top_recommendations_dict = {}
#####################
#####################
sorted_similarity_df = pd.read_csv('similarity.csv', index_col=0)
#####################

def get_displayed_movies():
    return movies.head(100)

def get_recommended_movies(new_user_ratings):
    predictions = []
    newuser = pd.Series(np.nan, index=ratings.columns)
    
    for movie_id, rating in new_user_ratings.items():
        movie_id_with_prefix = 'm' + str(movie_id)
        newuser[movie_id_with_prefix] = rating
    
    for l, rating in enumerate(newuser):     
        movie_name = newuser.index[l]
        nearest_neighbors = sorted_similarity_df.loc[movie_name].nlargest(30).index
        upp = sorted_similarity_df.loc[movie_name, nearest_neighbors] * newuser[nearest_neighbors]
        index2 = upp.notna()
        low = np.abs(sorted_similarity_df.loc[movie_name, nearest_neighbors][index2])

        prediction = np.sum(upp) / np.sum(low)

        predictions.append((movie_name, prediction.round(7)))

    predictions = np.array(predictions)
    predictions[:, 1] = predictions[:, 1].astype(float)

    non_nan_movie_ids = [movie_id for movie_id, rating in new_user_ratings.items() if not np.isnan(rating)]
    mask = ~np.isnan(predictions[:, 1].astype(float))

    filtered_predictions_array = predictions[mask]

    sorted_predictions_array = sorted(filtered_predictions_array, key=lambda x: x[1], reverse=True)
    sorted_predictions_array = [item for item in sorted_predictions_array if item[0] not in non_nan_movie_ids]
    top10 = sorted_predictions_array[:10]
    top10 = np.array(top10)
    col_names = top10[:, 0]

    col_names2 = [int(movie_id[1:]) for movie_id in col_names]

    filtered_movies = movies[movies['movie_id'].isin(col_names2)]
    
    return filtered_movies

def get_popular_movies(genre: str):
    if genre not in genres:
        return pd.DataFrame()

    # Check if recommendations for the genre are already computed
    if genre in top_recommendations_dict:
        return top_recommendations_dict[genre]
    
    genre_movies = movies[movies['genres'].str.contains(genre)]

    mean_ratings_genre = mean_ratings_df[mean_ratings_df['movie_id'].isin(genre_movies['movie_id'])]

    # Sort movies based on mean ratings in descending order
    sorted_movies = genre_movies.merge(mean_ratings_genre, left_on='movie_id', right_on='movie_id')
    sorted_movies = sorted_movies.sort_values(by='rating', ascending=False)

    # Select the top 10 movies
    top_10_movies = sorted_movies.head(10)

    # Save recommendations to the dictionary
    top_recommendations_dict[genre] = top_10_movies

    return top_10_movies
