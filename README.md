# Recommendation Systems Notebook

## Overview

This notebook implements various **recommendation system techniques**, showcasing the progression from simpler models to more advanced and hybrid approaches. It covers:

1. **Collaborative Filtering (CF)** using Singular Value Decomposition (SVD) for predicting user-item interactions.
2. **Hybrid Recommendation System** that combines CF with content-based filtering using genre similarity.
3. **Neural Network (NN)-Based Hybrid Recommender**, leveraging both SVD predictions and content similarity as input features.
4. **MLflow Integration** for experiment tracking, model logging, and reproducibility.

The examples demonstrate **accuracy evaluation**, **loss visualization**, **personalized recommendations**, and **model experiment reproducibility**.

---

## Key Sections

### 1. Collaborative Filtering with SVD

#### Description
This section uses Collaborative Filtering (CF) with **Singular Value Decomposition (SVD)** to predict ratings for user-item interactions. It works on the **MovieLens 100K dataset**.

#### Workflow:
- **Import Required Libraries**: Packages like `pandas`, `surprise`, and evaluation tools.
- **Dataset Loading**: Loads and processes the MovieLens dataset.
- **Train-Test Split**: Splits the dataset into 80% training and 20% testing data.
- **Model Training**: Trains an SVD model using the training dataset.
- **Evaluation**: Metrics like **RMSE** and **MAE** are computed for error analysis.
- **Top-N Recommendations**: A utility function generates the top-N recommendations for a user based on predicted ratings.

---

### 2. Hybrid Recommendation Systems

#### Description
Introduces a **hybrid approach**, combining collaborative filtering (SVD-based) with content similarity. The similarity is calculated using **TF-IDF vectorization** of movie genres.

#### Workflow:
- **Load MovieLens Data**: Loads ratings and movie metadata.
- **Data Transformation for Content-Based Filtering**:
    - Encodes genre information into a single string for each movie.
    - Calculates **TF-IDF vectors** for genres and computes **cosine similarity**.
- **Hybrid Scoring Function**:
    - Combines predictions from SVD and content-based similarity.
    - An adjustable `alpha` parameter sets the weighting between the two models.
- **Top-N Hybrid Recommendations**: Generates predictions using the weighted hybrid function.

---

### 3. Hybrid Recommendations with Neural Networks (NN)

#### Description
Builds a more advanced hybrid recommendation system using a **multi-layer neural network (NN)**. This leverages **SVD predictions** and **content similarity** to predict user-item ratings with increased accuracy.

#### Workflow:
1. **Data Preparation**: Combines SVD predictions, content similarity scores, and real user ratings into a training dataset.
2. **Neural Network Architecture**:
    - A fully-connected feedforward NN with dropout, batch normalization, and ReLU activation.
    - Trained to predict ratings based on input features (SVD and content similarity scores).
3. **NN Training**:
    - Includes early stopping based on validation loss.
    - Logs training and validation losses for visualization.
4. **Evaluation**: Tests the model on unseen data and calculates test loss.
5. **Top-N Recommendations via NN**: A function predicts user-specific recommendations using NN and ranks them by predicted scores.

---

### 4. MLflow Integration

#### Description
The MLflow library is used for experiment tracking, versioning, and model logging.

#### Workflow:
- **Set Up MLflow Tracking**: Connects to the MLflow server and initializes an experiment.
- **Parameter Logging**: Logs hyperparameters like learning rate, dropout rate, and NN architecture.
- **Metrics Logging**: Tracks validation loss, test loss, and other performance metrics.
- **Model Logging**: Saves the trained NN model artifact in the MLflow registry for reproducibility.

---

## Dependencies

The notebook depends on the following libraries:

- **Data Manipulation**: `pandas`, `numpy`
- **Collaborative Filtering**: `surprise` (for SVD)
- **Content-Based Filtering**: `sklearn` (`TfidfVectorizer`, `cosine_similarity`)
- **Neural Networks**: `torch` (PyTorch framework for building NN)
- **Experiment Tracking**: `mlflow`
- **Visualization**: `matplotlib`

## Results

1. **Models Evaluated**:
    - Collaborative Filtering (SVD): Metrics like RMSE and MAE provide a baseline for predictions.
    - Hybrid with Weighting: Combines CF and content-based approaches effectively.
    - NN-Based Hybrid: Improves accuracy by learning weighted combinations dynamically.

2. **Visualization**: Plots showing the NN training and validation losses over epochs.
3. **Experiment Logging**: All experiments with hyperparameters and metrics are logged in MLflow for reproducibility.

## How to Use

1. **Prepare the Environment**:
    - Install required libraries using `pip` or `conda`.
    - Configure the MLflow server (if using MLflow for experiment tracking).

2. **Run the Notebook**:
    - Follow sections sequentially to train and evaluate models.
    - Update the `test_user` ID to get recommendations for specific users.

3. **Experiment Tracking with MLflow**:
    - Run the MLflow cell in the notebook to log parameters, metrics, and the trained NN model.

---

This documentation acts as a guide for understanding and reproducing the recommendation system techniques presented. Each section builds on the previous, showcasing the evolution from SVD-based CF to a fully-featured NN-based hybrid recommender.
