# Spark Recommender

In the python shell, load pyspark using the commend `module load spark`. Then, the file can be run using the `spark -submit spark_recommender.py` commend if the `spark_recommender.py` file is located in the current directory.

## Overview
This is a take-home exercise from the XSEDE Big Data and Machine Learning Workshop. 

I loaded the rating data as Resilient Distributed Dataset (RDD) on Spark, transformed the data, and trained a recommendation system using the `pyspark.mllib.ALS` package. I only optimized the rank `k` in this exercise, but a grid search is possible. 

After the best rank `k` was determined, we can come up with recommendations for new users. Given a new user with some movie ratings, we can train the model again and extract the recommendations for the new user.

