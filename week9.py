import matplotlib.pyplot as pyplot
import numpy as np 
import pandas as pd 

ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

data = pd.merge(ratings, movies, on='movieId')
average_rating = pd.DataFrame(data.groupby('title')['rating'].mean())
average_rating['total_ratings'] = pd.DataFrame(data.groupby('title')['rating'].count())

rating_matrix = data.pivot_table(index='userId', columns='title', values='rating')

chosen_movie = rating_matrix['Avatar (2009)']
similar_movies = rating_matrix.corrwith(chosen_movie)

correlation = pd.DataFrame(similar_movies,columns=['Correlation'])
correlation.dropna(inplace=True)

correlation = correlation.join(average_rating['total_ratings'])

recommendation = correlation[correlation['total_ratings']>100].sort_values('Correlation', ascending=False)

recommendation = recommendation.merge(movies, on='title')
print(recommendation.head(10))

#print(average_rating.sort_values('total_ratings', ascending=False).head(10))
