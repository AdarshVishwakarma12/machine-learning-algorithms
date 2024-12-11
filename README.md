# Machine Learning Algorithms
**This repository contains scratch implementations of popular machine learning algorithms.**

## Overview
This repository showcases implementations of machine learning algorithms built from scratch without relying on popular machine learning libraries such as Scikit-learn, PyTorch, or TensorFlow. The main focus is on understanding the underlying mechanics of these algorithms. Currently, the repository includes:

### Implemented Algorithms
1. **K-Means Clustering**
   - A classic unsupervised learning algorithm for clustering data points into distinct groups.
   - **Key Features:**
     - Centroid initialization and updates.
     - Iterative assignment of points to clusters.
     - Convergence detection.
   - **Limitation:** May lack optimizations such as accelerated computations, and handling large dataset.

2. **Regression Tree**
   - A supervised learning algorithm used for both regression and decision-making tasks.
   - **Key Features:**
     - Recursive binary splitting to create decision nodes.
     - Calculation of splitting criteria using sum of sqaured residual
     - Predicting continuous target variables.
   - **Limitation:** May lack advanced features like boosting or bagging available in modern ensemble methods.

## Disclaimer
> These implementations are designed for educational purposes and may not match the performance of well-optimized libraries such as Scikit-learn, TensorFlow, or PyTorch in terms of:
> - Speed
> - Automatic regularization
> - Advanced tuning mechanisms
> - Robustness against edge cases

While significant effort has been made to ensure correctness, some bugs or inefficiencies might still exist. 
Contributions to improve the code are welcome!
