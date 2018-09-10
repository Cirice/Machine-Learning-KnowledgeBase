# A collection of ML questions and answers

### Questions:

#### 0. What is Bias, what is Variance and what is Bias-Variance trade-off?

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
#### 2. What is SVM? how does it work? describe the math behind it.
#### 3. What are L1 and L2 regularizations?
#### 4. How does Decision trees algorithm work?
#### 5. What are some major types of machine learning problems?
#### 6. What is Probably Approximately Correct(or PAC learning framework)?
#### 7. What are the example applications of machine learning?
#### 8. What are Features, Labels, Training samples, Validation samples, Test samples, Loss function, Hypothes set and Examples?
#### 9. What are Overfitting and Underfitting?
#### 10. What is Cross-validation? how does it help reduce Overfitting?
#### 11. How much math is involved in the machine learning? what are the pre-requisites of ML?
#### 12. What are Kernel method?
#### 13. What is Ensemble earning?
#### 14. What is Manifold learning?
#### 15. What is Boosting?
#### 16. What is Stochastic Gradiant Descent? describe the idea behind it.
#### 17. What is a statistical estimator?
#### 18. What is Rademacher complexity?
#### 19. What is VC-dimension?
#### 20. What are Vector, Vector space and Norm?
#### 21. What is Kernel trick?
#### 22. What is Hashing trick?
#### 23. How does Perceptron algorithm work?
