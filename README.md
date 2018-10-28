# A collection of useful ML questions(with answers yet to be added)

### Questions:

#### 0. What is Bias, what is Variance and what does Bias-Variance trade-off(or decomposition) mean?

This is a concept that is well known in the context of supervised learning where we have some labeled data and we want to estimate an unknown function **c(X)**
using a function with known format and parameters, called hypothesis function (i.e. **h(X)**).

[Wikipedia's](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) definitions for bias and variance are as follows:

* The bias is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
* The variance is an error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting)

Consider h(X) ~ c(X) then we have: c(X) = h(X) + bias error + variance error + irreducible error; apart from the third term (i.e. irreducible error) we can reduce the first two types of errors.
bias_error originates from the assumptions we make about the characteristics of our models, for example we assume that the relationship between input and output is linear (like in linear regression);
while creating out prediction models, we have a subset of all labeled data (training data) and our guide for knowing how good our model is based on its 
performance on this limited set, this creates a problem where the training data set is relatively small (many real world problems) because the variance of error on the unseen data (test data) could be huge. In fact by putting all our effort in improving the 
training score and lowering training error (we have no other choice!) we are doomed to overfit :( ). 

Machine learning algorithms are influenced differently based on their assumptions (bias) about input and output and consequently have different error variances. 
Algorithms that have a high variance are strongly influenced by the characteristics of the training data. This means that the characteristics of the data 
have influences the number and types of parameters used to characterize the hypothesis function [[https://machinelearningmastery.com](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/)]. 

The bias-variance trade-off is a central problem in supervised learning. Ideally, 
one wants to choose a model that both accurately captures the regularities in its training data,
 but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously. High-variance learning methods may be able to represent their training set well but are at risk of overfitting to noisy or unrepresentative training data. In contrast, algorithms with low variance typically produce simpler models that don't tend to overfit but may underfit their training data, failing to capture important regularities [[Wikipedia](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)]. 
 
#### 1. What is the main difference between the ordinary algorithms and machine learning algorithms?
#### 2. What is SVM? how does it work? Explain the math behind it
#### 3. What are L1 and L2 regularizations? Why we may need regularization?
#### 4. How does Decision Trees algorithm work?
#### 5. What are some major types of machine learning problems?
#### 6. What is Probably Approximately Correct(or PAC learning framework)?
#### 7. What are the example applications of machine learning?
#### 8. What are Features, Labels, Training samples, Validation samples, Test samples, Loss function, Hypothesis set and Examples?
#### 9. What are Overfitting and Underfitting?
#### 10. What is Cross-validation? how does it help reduce Overfitting?
#### 11. How much math is involved in the machine learning? what are the prerequisites of ML?
#### 12. What is Kernel Method(or Kernel Trick)?
#### 13. What is Ensemble Learning?
#### 14. What is Manifold Learning?
#### 15. What is Boosting?
#### 16. What is Stochastic Gradient Descent? Describe the idea behind it
#### 17. What is a Statistical Estimator?
#### 18. What is Rademacher complexity?
#### 19. What is VC-dimension?
#### 20. What are Vector, Vector Space and Norm?
#### 21. Why Logistic Regression algorithm has the word regression in it?
#### 22. What is Hashing trick?
#### 23. How does Perceptron algorithm work?
#### 24. What is Representation learning(or Feature learning)?
#### 25. How does Principal Component Analysis(PCA) work?
#### 26. What is Bagging?
#### 27. What is Feature Embedding?
#### 28. What is Similarity Learning?
#### 29. What is Feature Encoding? How an Auto-encoder Neural Network work?
#### 30 Does ML have any limit? What those limits may be?
#### 31 What does the word naive in the name of Naive Bayes family of algorithms stand for?
#### 32. Describe various strategies to handle an imbalanced dataset?
#### 33. Describe various strategies to tune up the hyper-parameters of a particular learning algorithm in general
#### 34. What are some general drawbacks of tree based learning algorithms?
#### 35. What are the differences between ordinary Gradient Boosting and XGBOOST?
#### 36. What is the main difference between Time Series Analysis and Machine Learning?
#### 37. What is the main difference between Data Mining and Machine Learning? Are they the same?
#### 38. What is a Generative learning model?
#### 39  What is a Discriminative learning model?
#### 40. What is the difference between Generative and Discriminative models?
#### 41. What is Case-Based Learning?
#### 42. What is Co-variance Matrix?
#### 43. What is the difference between Correlation and Causation?
#### 44. What is the Curse of Dimensionality? How does it may hinder the learning process?
#### 45. How Dimensionality Reduction help improve the performance of the model?
#### 46. What is Feature Engineering?
#### 47. What is Transfer Learning?
#### 48. What do (Multi-)Collinearity, Autocorrelation, Heteroskedasticity and Homoskedasticity mean?
#### 49. Explain Backpropagation, What are some of its shortcomings?
#### 50. How do Boltzmann Machines work?
#### 51. What is the difference between In-sample evaluation and Holdout evaluation of a learning algorithm?
#### 52. What is Platt Scaling?
#### 53. What is Monotonic(or Isotoonic) Regression?
#### 54. How BrownBoost algorithm works?
#### 55. How Multivariate Adaptive Regression Splines(MARS) algorithm works?
#### 56. What are K-Fold Cross Validation and Stratified Cross Validation?
#### 57. What is K-Scheme (Categorical Feature) Encoding? What are some drawbacks of one-hot categorical feature encoding?
#### 58. What is Locality Sensitive Hashing(LSH)?
#### 59. What are the differences of Arithmetic, Geometric and Harmonic means?
#### 60. What is a Stochastic Process?
#### 61. How Bayesian Optimization work?
#### 62. What is the difference between Bayesian and Frequentist approaches?
#### 63. Why sometimes it is needed to Scale or to Normalise features?
#### 65. What is Singular-Value Decomposition (SVD)? What are some applications of SVD?
#### 66. Define Eigenvector, Eigenvalue, Hessian Matrix, Gradient
#### 67. What is an (Intelligent) Agent?
#### 68. What is Q-Learning?
#### 69. Define Markov Chain and Markov Process
#### 70. Explain how K-means clustering algorithm works, Why is it so popular?
#### 71. How Hierchical clustering algorithm works?
#### 72. What is Discriminant Analysis?
#### 73. What is Multivariate Analysis?
#### 74. What is the Rank of a matrix?
#### 75. How does Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH) algorithm work?
#### 76. What is a Mixture Model?
#### 77. How does Machine Learning modeling fare against old-school mathematical modeling such as modeling a real-world system via use of differential equations?
#### 78. How to pick a suitable kernel for SVM?
#### 79. What is Query-Based Learning?
#### 80. What is Algorithmic Probability?
#### 81. What are  Occam's Razor and Epicurus' principle of multiple explanations?
#### 82. What are Filter and Wrapper based feature selection methods?
#### 83. What is Graph Embedding? What are some of its applications?
#### 84. What is Multi-Armed Bandit Problem?
#### 85. What is Active Learning?
#### 86. What are Hierarchical Temporal Memory algorithms (HTMs)?
#### 87. What are Factorization Machines?
#### 88. What are Probabilistic Graphical Models (PGM)?
#### 89. What is Entity Embedding?
#### 90. What is Feature Augmentation?
#### 91. What is Negative Sub-Sampling?
#### 92. What is a Tensor?
