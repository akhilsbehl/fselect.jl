# Detect data types:

* Numerical
  - Integer
  - Float
* Factor <= Ask for ordinality
  - Dummy
  - Low cardinality
  - High cardinality
  - ID
* Date
* Time
  - Second
  - Minute
  - Hour
  - Day
  - Week
  - Month
  - Year

# Data quality selectors

## Missing percentage

* Numerical response - perturbation analyses
* Categorical response - Threshold relative to positive class proportion
* TODO

## No or very small variance
* Numerical response - perturbation analyses
* Categorical response - Threshold relative to positive class proportion
* TODO

# Unsupervised feature selectors

# Supervised feature selectors

## Wrappers
Fit the candidate model per subset to evaluate and choose the best

## Filters
Preselect features based on some f(y, x)

### Categorical response

#### Categorical feature
* Chi-squared
* [Mutual Integer](http://fourier.eng.hmc.edu/e176/lectures/probability/node6.html) 

#### Numerical feature

* ANOVA
* Kendall's rank correlation coefficient

#### Bootstrapping

With high imbalance in the categorical response - use a bootstrapped version of the feature selectors above.

### Numerical response

#### Categorical feature
* ANOVA
* Kendall's rank correlation coefficient

#### Numerical feature

* Pearson's correlation coefficient
* Spearman's rank coefficient
* [Mutual information](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html) 

# Lower dimensionality projectors

# Resources

* https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
* 
