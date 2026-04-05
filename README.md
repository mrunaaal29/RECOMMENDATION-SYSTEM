# RECOMMENDATION-SYSTEM

*COMPANY *: CODTECH IT SOLUTIONS

*NAME *: MRUNAL TAYDE

*INTERN ID *: CTIS7006

*DOMAIN *: MACHINE LEARNING

*DURATION *: 4 WEEEKS

*MENTOR *: NEELA SANTOSH

Movie Recommendation System using Implicit Feedback & ALS Model

1. Project Overview

This project implements a Movie Recommendation System using the MovieLens 1M dataset and the Implicit Matrix Factorization (ALS) model. The system predicts movie preferences for users based on implicit feedback derived from their explicit ratings. Unlike traditional recommendation engines that directly use explicit scores, this project converts ratings into confidence weights, allowing the ALS model to operate effectively in an implicit recommendation setting.

The goal of this project is to build a scalable, matrix-factorization-based recommender capable of generating personalized movie suggestions and evaluating the model using standard ranking metrics such as Precision@K, Recall@K, and MAP@K.

This repository contains data preprocessing, model training using the implicit library, recommendation generation, and evaluation using a Leave-One-Out (LOO) methodology commonly used in recommendation system research.

2. Dataset Information

This project uses the MovieLens 1M Dataset, created by GroupLens Research.

Dataset Link (Kaggle Mirror)

https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset

Official MovieLens Download Page

https://grouplens.org/datasets/movielens/1m/

About the Dataset
1 million movie ratings
6,000+ users
4,000+ movies
Ratings are given on a scale of 1 to 5
Includes:
ratings.dat → UserID :: MovieID :: Rating :: Timestamp
movies.dat → MovieID :: Title :: Genres
users.dat → User demographic information (optional)

For this project, only ratings and movies data are used.
Ratings are mapped to implicit confidence scores so the ALS model can learn from positive feedback strength rather than raw rating values.

3. Technologies and Tools Used
Programming Language
Python 3
Libraries
pandas → Data loading and preprocessing
scipy.sparse → Building sparse interaction matrices
implicit → ALS model for implicit feedback
numpy → Numerical operations
tqdm → Progress bars
matplotlib (optional) → Visualization
Editor / Platform
Jupyter Notebook
VS Code / PyCharm (optional)
Anaconda environment
4. Methodology
Step 1: Data Preprocessing
Load ratings and movies

Convert explicit ratings to implicit confidence scores using the formula:

confidence = 1 + (rating / 5) * 40
Create user–item interaction matrix (CSR Sparse Matrix)
Step 2: Model Training (ALS – Alternating Least Squares)

The implicit library's ALS algorithm is used with factorization:

50 latent factors
Regularization = 0.1
20 iterations
Trained on the item-user matrix (as required by implicit)
Step 3: Personalized Movie Recommendations

For any user:

Retrieve top-N recommended movies
Ensure already-watched items are filtered out
Map internal item indices back to actual MovieIDs and titles
Step 4: Evaluation Metrics

Implemented metrics:

Precision@10
Recall@10
MAP@10
NDCG@10 (optional)

Using Leave-One-Out evaluation:

For every user, hide one “last rated” movie
Train on the remaining interactions
Check if the hidden movie appears in the top-K recommendations
5. Results Summary

Using ALS with implicit confidence signals, the model produced:

Precision@10 ≈ 0.0016
Recall@10 ≈ 0.0016
MAP@10 ≈ 0.00025

These values are typical for implicit matrix factorization models on large sparse datasets under strict leave-one-out evaluation.

6. Applications

This project demonstrates concepts applicable to:

OTT platforms (Netflix, Amazon Prime, Hotstar)
E-commerce product recommendations
Music streaming services (Spotify, Gaana)
News feed personalization
Online advertising ranking
Book or content recommendation platforms

The pipeline can be extended to larger datasets, multiple signals (views, clicks, time spent), and advanced ranking models like BPR or Neural Recommenders.

7. How to Run the Project
Download the dataset
Place ratings.dat and movies.dat inside a folder like ml-1m/

Install dependencies:

pip install implicit scipy pandas tqdm
Run the notebook or script to train the model
Generate recommendations or compute evaluation metrics
