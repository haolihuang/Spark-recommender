#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 18:39:00 2021

@author: Hao-Li Huang
"""

import math
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS

sc = SparkContext()
sc.setLogLevel('ERROR') # Print only error messages; suppress info messages

# Read rating file as Resilient Distributed Dataset (RDD)
ratings_raw = sc.textFile('ratings.csv')

# Parse lines into tuple (user_id, product_id, rating)
# from 'user_id, product_id, rating, timestamp'
ratings = ratings_raw.map(lambda line: line.split(',')).map(
    lambda token: (int(token[0]), int(token[1]), float(token[2])))

# Split into training, validation, and test sets (60%, 20%, 20%)
train, val, test = ratings.randomSplit([3,1,1])

# Drop ratings of validation and test sets for the predictAll method
predict_val = val.map(lambda x: (x[0], x[1]))
predict_test = test.map(lambda x: (x[0], x[1]))

# Hyperparameters for training recommendation system
iterations = 10 # We can plot the RSME against n_iter to determine best n_iter
regularization = 0.1
trial_ranks = [4, 8, 12]

lowest_rsme = float('inf')

# We only vary the rank k here
# Make an ALS model for each k and record the RMSE
for k in trial_ranks:
    model = ALS.train(train, k, iterations = iterations, 
                      lambda_ = regularization)
    # Coercing ((u, p), r) tuple format to accomodate join
    predictions = model.predictAll(predict_val).map(
        lambda r: ((r[0], r[1]), r[2]))
    ratings_and_preds = val.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    
    # Compute RSME
    rsme = math.sqrt(ratings_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    print('For k = {}, the RMSE is {}'.format(k, rsme))
    # Find smallest RSME
    if rsme < lowest_rsme:
        lowest_rsme = rsme
        best_k = k 
        
print('The best rank is size {}'.format(best_k))
# This is a relatively small dataset, so we also have a small best_k
# However, when a larger dataset is used, likely the best_k would also be larger


# Train model using best_k and compute test error
model = ALS.train(train, best_k, iterations = iterations,
                  lambda_ = regularization)
predictions = model.predictAll(predict_test).map(
    lambda r: ((r[0], r[1]), r[2]))
ratings_and_preds = test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
rsme = math.sqrt(ratings_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
print('For testing data the RMSE is {}'.format(rsme))


# Add a new user. Assuming ID 0 is unused.
new_user_id = 0
# Tuples of (user_id, product_id, rating)
new_user = [
     (0,100,4), # City Hall (1996)
     (0,237,1), # Forget Paris (1995)
     (0,44,4), # Mortal Kombat (1995)
     (0,25,5), # etc....
     (0,456,3),
     (0,849,3),
     (0,778,2),
     (0,909,3),
     (0,478,5),
     (0,248,4)
    ]
# Put the new user data as RDD
new_user_rdd = sc.parallelize(new_user)

# Add the new user data to ratings
updated_ratings = ratings.union(new_user_rdd)
# Update model; train using the whole data set
updated_model = ALS.train(updated_ratings, best_k, iterations = iterations,
                  lambda_ = regularization)

# For more readable output, load movie names from movie.csv
movies_raw = sc.textFile('movies.csv')
# Parse lines to tuple (product_id, product_name)
# from 'product_id, product_name'
movies = movies_raw.map(lambda line: line.split(',')).map(
    lambda token: (int(token[0]), token[1]))

new_user_rated_ids = map(lambda x: x[1], new_user)
# Creates (new_user_id, product_id) tuples for unrated movies 
new_user_unrated = movies.filter(lambda m: m[0] not in new_user_rated_ids).map(
    lambda m: (new_user_id, m[0]))

# Get recommendations
new_user_recommendations = updated_model.predictAll(new_user_unrated)

# Drop new_user_id so we end up having (product_id, rating)
prod_rating = new_user_recommendations.map(lambda r: (r[1], r[2]))
# Join with movies and get (product_name, rating)
new_user_recommendation_titled = movies.join(prod_rating).map(lambda t: t[1])

# Top recommendations
top_recommends = new_user_recommendation_titled.takeOrdered(10, 
                                                        key = lambda x: -x[1])
for line in top_recommends:
    print(line)
    
    
# Example: Look up rating using (user_id, product_id) 
# user_id = 0 and product_id = 800 (Lone Star (1996))
one_movie = sc.parallelize([(0, 800)])
rating = updated_model.predictAll(one_movie)
print(rating.collect())
    
    


