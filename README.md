This notebook explores examples of recommendation systems, covering collaborative filtering (CF), hybrid approaches, and deep learning techniques. It begins with CF using Singular Value Decomposition (SVD) for generating recommendations based on user-item interactions. The workflow includes importing necessary libraries, loading built-in datasets, splitting them into training and testing sets, training the SVD model, and evaluating it with metrics like RMSE and MAE.


Next, hybrid approaches are introduced, combining CF with content-based filtering using genre similarity calculated via TF-IDF and cosine similarity. A hybrid recommendation function integrates scores from both models, providing personalized suggestions.


The notebook then extends the hybrid approach with a Neural Network (NN), using features from SVD and content similarity to predict ratings. It covers model preparation, training, validation, and evaluation, with early stopping and loss visualization.


Finally, MLflow is integrated to log hyperparameters, metrics, and the trained NN model, demonstrating experiment tracking and reproducibility. The notebook provides end-to-end examples of implementing, evaluating, and enhancing recommendation systems with modern techniques.
