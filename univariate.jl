# Univariate feature selection routines.
# Shamelessly lifted from sklearn.feature_selection._univariate_selection.py

# Author: Akhil S. Behl
# sklearn Authors: V. Michel, B. Thirion, G. Varoquaux, A. Gramfort,
#                  E. Duchesnay. L. Buitinck, A. Joly

# NB:
# 1. Does not explicitly implement or test anything with sparse matrices.
# 2. Does not do most of the sanity checking and edge cases that sklearn has
#    accumulated over the years. Unclear if they are all necessary.

using Distributions
using LinearAlgebra

###########################################################################
#                            Scoring Functions                            #
###########################################################################

"""
Performs a 1-way ANOVA.

The one-way ANOVA tests the null hypothesis that 2 or more groups have
the same population mean. The test is applied to samples from two or
more groups, possibly with differing sizes.

Parameters
----------
samples: Array{<:AbstractFloat,2}: sample measurements for the test

Returns
-------
statistic: float: The computed F-value of the test.
pvalue: float: The associated p-value from the F-distribution.

Notes
-----
The implementation does not check for or treat missing values.

The ANOVA test has important assumptions that must be satisfied in order
for the associated p-value to be valid.

1. The samples are independent
2. Each sample is from a normally distributed population
3. The population standard deviations of the groups are all equal. This
    property is known as homoscedasticity.

If these assumptions are not true for a given set of data, it may still be
possible to use the Kruskal-Wallis H-test although with some loss of power.

References
----------

[1] Lowry, Richard.  Concepts and Applications of Inferential Statistics.
    http://facultysites.vassar.edu/lowry/PDF/c14p1.pdf
    http://vassarstats.net/textbook/ [Ch 13]
"""
function one_way_anova(samples...)

  stype = eltype(samples)
  
  n_groups = convert(stype, length(samples))
  group_sizes = convert.(stype, [size(sample, 1) for sample in samples])
  total_size = sum(group_sizes)

  squared_samples = [sample .^ 2 for sample in samples]
  group_sum_squares = sum.(squared_samples)
  total_sum_squares = sum(group_sum_squares)
  square_of_sums = (sum.(samples)) .^ 2

  # ssd := sum of squared deviates
  ssd = group_sum_squares .- (square_of_sums ./ group_sizes)
  total_ssd = total_sum_squares - sum(sum.(samples)) ^ 2 / total_size
  ssd_within_groups = sum(ssd)
  ssd_between_groups = total_ssd - ssd_within_groups

  # df := degrees of freedom
  df_within_groups = total_size - n_groups
  df_between_groups = n_groups - one(stype)

  # msd := mean of squared deviates
  msd_within_groups = ssd_within_groups / df_within_groups
  msd_between_groups = ssd_between_groups / df_between_groups

  statistic = msd_between_groups / msd_within_groups
  f_dist = Distributions.FDist(df_between_groups, df_within_groups)
  pvalue = Distributions.ccdf(f_dist, statistic)

  return statistic, pvalue

end


"""Compute the ANOVA F-value for the provided sample.

Parameters
----------
X : Array{<:AbstractFloat,2}: shape = [n_samples, n_features]
    The set of regressors that will be tested sequentially.

y : Array{<:AbstractFloat,1}: shape(n_samples)
    The data matrix.

Returns
-------
statistic : array, shape = [n_features,]
    The set of F values.

pvalue : array, shape = [n_features,]
    The set of p-values.

Notes
-----
The implementation does not check for or treat missing values.
"""
function ftest_classification(X, y)

  classes = unique(y)
  statistic = Float64[]
  pvalue = Float64[]

  for c in eachcol(X)
    samples = [view(c, findall(v -> v == class, y), :) for class in classes]
    stat, prob = one_way_anova(samples...)
    push!(statistic, stat)
    push!(pvalue, prob)
  end
  
  return statistic, pvalue

end


"""
Performs a chi-squared test.

Pearson's chi-squared test is used to determine whether there is a statistically significant difference between the expected frequencies and the observed frequencies in one or more categories of a contingency table

Parameters
----------
observed: Array{<:AbstractFloat,1}: observed frequency
expected: Array{<:AbstractFloat,1}: expected frequency

Returns
-------
statistic: float: The computed chi-squared value of the test
pvalue: float: The associated p-value from the F-distribution

Notes
-----
The implementation does not check for or treat missing values.
"""
function chisquare_test(observed, expected)

  nrow, ncol = size(observed)
  df, z, statistic = nrow - 1, zero(Float64), zeros(Float64, ncol)

  for j in 1:ncol
    for i in 1:nrow
      increment = ((observed[i, j] - expected[i, j]) ^ 2) / expected[i, j]
      if isfinite(increment)
        statistic[j] += increment
      end
    end
  end

  cs_dist = Distributions.Chisq(df)
  pvalue = Distributions.ccdf.(cs_dist, statistic)

  return statistic, pvalue

end


"""
Convert a categorical y into dummy variables.
"""
function binarize_classes(y)
  uy = sort(unique(y))
  m, n = size(y, 1), size(uy, 1)
  binarized = Array{Int8}(undef, m, n)
  for (i, value) in enumerate(y)
    for (j, class) in enumerate(uy)
      binarized[i, j] = value == class ? one(Int8) : zero(Int8)
    end
  end
  return binarized
end


"""Compute chi-squared stats between each non-negative feature and class.

This score can be used to select the n_features features with the
highest values for the test chi-squared statistic from X, which must
contain only non-negative features such as booleans or frequencies
(e.g., term counts in document classification), relative to the classes.

Recall that the chi-square test measures dependence between stochastic
variables, so using this function "weeds out" the features that are the
most likely to be independent of class and therefore irrelevant for
classification.

Parameters
----------
X : Array{<:AbstractFloat,2}: shape (n_samples, n_features)
    Sample vectors.

y : Array{<:AbstractFloat,1}: array-like of shape (n_samples,)
    Target vector (class labels).

Returns
-------
statistic : array, shape = (n_features,)
            chisq statistics of each feature.
pvalue : array, shape = (n_features,)
         p-values of each feature.

Notes
-----
Complexity of this algorithm is O(n_classes * n_features).
"""
function chisq(X, y)

  for x in X
    if x < zero(x)
      error("Values in X must be non-negative")
    end
  end

  ybin = binarize_classes(y)
  if size(ybin, 2) == 1
    ybin = hcat(ybin, one(Int8) - ybin)
  end

  observed = ybin'X

  feature_count = sum(X, dims=1)
  class_prob = mean(ybin, dims=1)
  expected = class_prob'feature_count

  return chisquare_test(observed, expected)

end
