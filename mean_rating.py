import pandas as pd
import numpy as np

ratings = pd.read_csv('ratings.csv', sep=',', index_col=0)

mean_ratings = ratings.mean()

mean_ratings_df = pd.DataFrame({'movie_id': mean_ratings.index, 'rating': mean_ratings.values})

mean_ratings_df = mean_ratings_df.sort_values(by='rating', ascending=False)

mean_ratings_df['movie_id'] = mean_ratings_df['movie_id'].str.replace('m', '').astype(int)
mean_ratings_df.to_csv('mean_ratings_df.csv')