# A collection of useful ML questions(with answers yet to be added)

### Questions:

#### 0. What is bias, what is variance, and what does bias-variance trade-off (or decomposition) mean?

This is a concept that is well known in the context of supervised learning where we have some labelled data, and we want to estimate an unknown function $c(X)$
using a function with known format and parameters, called hypothesis function (i.e., $h(X)$).

[Wikipedia's](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) definitions for bias and variance are as follows:

* The bias is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).

* The variance is an error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data rather than the intended outputs (overfitting)

Consider $h(X)$ ~ $c(X)$. Then we have $c(X)$ = $h(X)$ + bias error + variance error + irreducible error; apart from the third term (i.e. irreducible error), we can reduce the first two types of errors.
bias_error originates from the assumptions we make about the characteristics of our models. For example, we assume that the relationship between input and output is linear (like in linear regression); while creating our prediction models, we have a subset of all labelled data (training data) and our guide for knowing how good our model is based on its performance on this limited set, this creates a problem where the training data set is relatively small (many real-world problems) because the variance of an error on the unseen data (test data) could be huge. In fact, by putting all our effort into improving the training score and lowering training error (we have no other choice!), we are doomed to overfit. :(

Machine learning algorithms are influenced differently based on their assumptions (bias) about input and output and consequently have different error variances. The characteristics of the training data strongly influence algorithms that have a high variance. This means that the characteristics of the data have influenced the number and types of parameters used to characterise the hypothesis function [[https://machinelearningmastery.com](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/)].

The bias-variance trade-off is a central problem in supervised learning. Ideally, one wants to choose a model that accurately captures the regularities in its training data and generalises well to unseen data. Unfortunately, it is typically impossible to do both simultaneously. High-variance learning methods may be able to represent their training set well but are at risk of overfitting to noisy or unrepresentative training data. In contrast, algorithms with low variance typically produce simpler models that don't tend to overfit but may underfit their training data, failing to capture important regularities [[Wikipedia](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)]. 

#### 1. What is the main difference between ordinary algorithms and machine learning algorithms?

Usually, algorithms in Computer Science are analysed based on the size of the input and how much (memory) space they need to run. At the same time, this still holds for Machine Learning algorithms; there is one more criterion which needs to be answered for ML algorithms, namely, that for ML algorithms, there may be a boundary on how many input instances are necessary for the learning process to succeed. This is important to consider because, in practice, there may not be enough needed samples for the learning process to guarantee the intended results.


#### 2. What is SVM? How does it work? Explain the math behind it

SVM (Support Vector Machine) is a supervised machine learning algorithm for classification and regression analysis. The goal of the SVM algorithm is to find the best line (or hyperplane) that can split the data into two classes in a multi-class classification problem.

The math behind SVM involves finding the optimal hyperplane that maximises the margin between the classes, meaning it has the largest distance between the nearest data points of each class. These nearest data points are support vectors and play a crucial role in defining the optimal hyperplane.

Mathematically, the SVM algorithm tries to find the hyperplane that maximises the following objective function:

$\frac{1}{\left | w \right |} \left[ \text{Margin} \right] = \frac{1}{\left | w \right |} \min_{n} \left[ t_{n}(\left \langle w,x_{n} \right \rangle - b) \right] $

where w is the normal vector to the hyperplane and b is the bias term. The term tn represents the class label (-1 or 1) of the data point xn. The function returns the maximum margin, which is the distance between the hyperplane and the nearest data points.

In summary, SVM works by finding the hyperplane with the maximum margin that separates the classes, and the margin is defined by the support vectors which are the nearest data points to the hyperplane.

#### 3. What are L1 and L2 regularisations? Why we may need regularisation?

L1 and L2 regularisation are methods to prevent overfitting in machine learning models by adding a penalty term to the loss function.

L1 regularisation, also known as Lasso, adds the absolute value of the coefficients as a penalty term to the loss function. This results in some coefficients being exactly equal to zero, effectively performing feature selection.

L2 regularisation, also known as Ridge Regression, adds the squared values of the coefficients as a penalty term to the loss function. This has the effect of shrinking the coefficients towards zero, but unlike L1 regularisation, it does not result in any coefficients being exactly equal to zero.

Regularisation is necessary in many cases because it helps to prevent overfitting. Overfitting occurs when a model fits the training data too well and becomes overly complex, capturing the noise and random fluctuations in the data rather than the underlying patterns. This results in poor generalisation performance on unseen data.

Regularisation helps to avoid overfitting by adding a penalty term to the loss function that discourages the model from assigning too much importance to any one feature. This forces the model to strike a balance between fitting the training data well and having a simple, generalisable structure.

#### 4. How does the decision tree algorithm work?

Decision Trees is a tree-based algorithm used for both classification and regression tasks. It works by recursively partitioning the data into smaller subsets based on the values of the features, with the goal of producing subsets that contain samples that belong to the same class or have similar target values.

The algorithm starts at the root of the tree and selects a feature to split the data based on some criterion, such as information gain or gini impurity. The samples are then divided into branches based on the values of the selected feature, and this process is repeated at each internal node of the tree until a stopping condition is reached.

For example, in a binary classification problem, if a certain feature is selected at a node, the samples are divided into two branches: one for samples with values below a certain threshold and another for samples with values above the threshold. This process continues until a node contains samples of only one class, or until a predetermined maximum depth is reached.

The final result of the Decision Trees algorithm is a tree-like model that can be used to make predictions for new data points. To make a prediction, the algorithm follows a path through the tree based on the values of the features in the sample, until it reaches a leaf node. The prediction is then the class label or target value associated with that leaf node.

In summary, Decision Trees works by recursively dividing the data into smaller subsets based on feature values, with the goal of creating nodes that contain samples with the same class label or target value. The final result is a tree-like model that can be used for making predictions.

#### 5. What are some major types of machine learning problems?
#### 6. What is Probably Approximately Correct(or PAC learning framework)?
#### 7. What are the example applications of machine learning?
#### 8. What are Features, Labels, Training samples, Validation samples, Test samples, Loss function, Hypothesis set and Examples?
#### 9. What are Overfitting and Underfitting?
#### 10. What is Cross-validation? how does it help reduce overfitting?
#### 11. How much math is involved in machine learning? what are the prerequisites of ML?
#### 12. What is Kernel Method(or Kernel Trick)?
#### 13. What is Ensemble Learning?
#### 14. What is Manifold Learning?
#### 15. What is Boosting?
#### 16. What is Stochastic Gradient Descent? Describe the idea behind it
#### 17. What is Statistical Estimator?
#### 18. What is Rademacher complexity?
#### 19. What is VC-dimension?
#### 20. What are Vector, Vector Space and Norm?
#### 21. Why Logistic Regression algorithm has the word regression in it?
#### 22. What is Hashing trick?
Hashing trick (an analogy to another useful technique called kernel trick) is an efficient method of representing (usually a very huge collection of discrete) features as numerical vectors (called feature vectorising).

To understand the main idea behind this method, consider this somehow oversimplified  example:

Assume we have two document samples called document A and document B containing a few words each:

|  Document A |  Document B |
|--------|--------|
|   This is a short sentence.     |    This is a kinda longer sentence that I am writing here.     |

Now we need a hash function and an array for each data sample, our hash function maps the input string into an integer ranging from 1 to the size of the array.

For example, our hash function looks like something like this:

| Input word | Output |
|--------|--------|
|     This   |  1      |
|    is    |     2   |
|      a  |       3 |
|       short |    4    |
|    sentence    |  5      |
|    kinda    |      6  |
|   longer     |      7  |
|    that    |    8    |
|     I   |     9   |
|      am  |     10   |
|    writing    |  11      |
|    here    |      12  |

Now if we hash each word (ignoring the punctuation and space character) from document A and document B using our hash function and use the output integer as the index for the array, we can represent two documents as two same-sized bit-arrays (these arrays are having only 0 and 1s as values). If a word is present in the document, we put 1 into the array cell with the corresponding index for that word and 0 otherwise.

Via application of this scheme vectotized representation of documents A and B would look like this:

|Document| index 1 | index 2 | index 3 | index 4 | index 5 | index 6 | index 7 | index 8 | index 9 | index 10 | index 11 | index 12 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|**A**|1|1|1|1|1|0|0|0|0|0|0|0|
|**B**|1|1|1|0|1|1|1|1|1|1|1|1|

As you can see, in our example we represented two documents containing words (complex feature set) by two very compact arrays of bits (111110000000 for document A and 11101111111 for document B) which each only needs 12 bits to represent each document.

Hashing trick is a mainly useful method for dimensionality reduction on large datasets because i) it can easily vectorise complex features (i.e. words or terms in the text); ii) it can be very efficient when the feature space is very large, and it may not be feasible to hold everything into the main memory during the learning or working with the data because we could use smaller arrays and still get a good representation. One of the famous use-cases of Hashing trick (also known as Feature hashing) is its successful application to the problem of detecting spam emails by using it to embed each sample into a lower dimension.

There are various strategies for implementing Hashing trick including the application of different hash functions and use of more complex values instead of just 0 and 1s, you could check out [Wikipedia article on Hashing trick](https://en.wikipedia.org/wiki/Feature_hashing) and [this aticle](https://alex.smola.org/papers/2009/Weinbergeretal09.pdf) for more information.

#### 23. How does the Perceptron algorithm work?
#### 24. What is Representation learning(or Feature learning)?
#### 25. How does Principal Component Analysis(PCA) work?
#### 26. What is Bagging?
#### 27. What is Feature Embedding?
#### 28. What is Similarity Learning?
#### 29. What is Feature Encoding? How an Auto-encoder Neural Network work?
#### 30 Does ML have any limit? What those limits maybe?
#### 31 What does the word naive in the name of Naive Bayes family of algorithms stand for?
#### 32. Describe various strategies to handle an imbalanced dataset?
#### 33. Describe various strategies to tune up the hyper-parameters of a particular learning algorithm in general
#### 34. What are some general drawbacks of tree-based learning algorithms?
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
#### 53. What is Monotonic(or Isotonic) Regression?
#### 54. How BrownBoost algorithm works?
#### 55. How Multivariate Adaptive Regression Splines(MARS) algorithm works?
#### 56. What are K-Fold Cross-Validation and Stratified Cross Validation?
#### 57. What is K-Scheme (Categorical Feature) Encoding? What are some drawbacks of one-hot categorical feature encoding?
#### 58. What is Locality Sensitive Hashing(LSH)?
#### 59. What are the differences between Arithmetic, Geometric and Harmonic means?
#### 60. What is the Stochastic Process?
#### 61. How Bayesian Optimisation work?
#### 62. What is the difference between Bayesian and Frequentist approaches?
#### 63. Why sometimes it is needed to Scale or to Normalise features?
#### 65. What is Singular-Value Decomposition (SVD)? What are some applications of SVD?
#### 66. Define Eigenvector, Eigenvalue, Hessian Matrix, Gradient
#### 67. What is an (Intelligent) Agent?
#### 68. What is Q-Learning?
#### 69. Define Markov Chain and Markov Process
#### 70. Explain how the K-means clustering algorithm works, Why is it so popular?
#### 71. How Hierarchical clustering algorithm works?
#### 72. What is a Discriminant Analysis?
#### 73. What is Multivariate Analysis?
#### 74. What is the Rank of a matrix?
#### 75. How does Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH) algorithm work?
#### 76. What is a Mixture Model?
#### 77. How does Machine Learning modelling fare against old-school mathematical modelling such as modelling a real-world system via the use of differential equations?
#### 78. How to pick a suitable kernel for SVM?
#### 79. What is Query-Based Learning?
#### 80. What is Algorithmic Probability?
#### 81. What is Occam's Razor and Epicurus' principle of multiple explanations?
#### 82. What are Filter and Wrapper based feature selection methods?
#### 83. What is Graph Embedding? What are some of its applications?
#### 84. What is the Multi-Armed Bandit Problem?
#### 85. What is Active Learning?
#### 86. What are Hierarchical Temporal Memory algorithms (HTMs)?
#### 87. What are Factorisation Machines?
#### 88. What are Probabilistic Graphical Models (PGM)?
#### 89. What is Entity Embedding?
#### 90. What is Feature Augmentation?
#### 91. What is Negative Sub-Sampling?
#### 92. What is a Tensor?

A tensor is a high dimensional array (3 or more dimensions; actually scalars, 1-dimensional arrays, and matrices are considered low order tensors - order in here means the number of dimensions an array has) of numerical values, in other words, tensors are the generalisation of matrices so that usual Lineal Algebraic rules and operations that are applicable to simple arrays and matrices could be used for working in higher dimensions.

Tensors were introduced early in the 20th century but did not receive much attention in Computer Science until recently. Tensor is relatively low-level mathematical objects (much like 1-dimensional arrays and matrices), [this article](https://arxiv.org/pdf/1711.10781) contains a self-contained and thorough introduction to some applications of tensors that demonstrates their various useful use-cases in various setups (from finding unique matrix decompositions more easily to learning on a generative latent or hidden variable model via use of tensor decomposition techniques).


#### 93. What is the Exceptional Model Mining (EMM)?
#### 94. What is Rejection Sampling?
#### 95. What is (Sparse) Dictionary Learning?
#### 96. What is Bonferroni's Principle?
Informally, Bonferroni's principle states that given a rather huge collection of samples representing a set of events, you can calculate the expected number of occurrences for each distinct event in the dataset assuming events are happening totally in random, now if the calculated expected number of a particular event is very larger than the actual number of occurrences that you hope to find you can be very much assured that you find many bogus or invalid events (events that can be like what you are looking for but are observed only due to possible existence of underlying randomness in the dataset).

This is a very good example of a practical Data Mining problem, the main idea is when you try to detect a rare event in a huge dataset you could easily be tricked into falsely believing that you have identified many more number of that particular event than actually there are present in the dataset.
Bonferroni's principle helps us to be aware that sometimes we have to search only for very rare events that are very much unlikely to happen in the random data in order to be confident that there aren't!

You can find more information in [Mining of Massive Datasets](http://mmds.org/) about the Bonferroni's principle.
#### 97. What is Statistical Relational Learning?
