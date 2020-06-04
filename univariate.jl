# Univariate feature selection routines.
# Shamelessly lifted from sklearn.feature_selection

# Author: Akhil S. Behl
# sklearn Authors: V. Michel, B. Thirion, G. Varoquaux, A. Gramfort,
#                  E. Duchesnay. L. Buitinck, A. Joly

# NB:
# 1. Does not explicitly implement or test anything with sparse matrices.
# 2. Does not do most of the sanity checking and edge cases that sklearn has
#    accumulated over the years. Unclear if they are all necessary.

using Distributions

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
samples: Array{Float,2}: sample measurements for the test

Returns
-------
statistic: float: The computed F-value of the test.
pvalue: float: The associated p-value from the F-distribution.

Notes
-----
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

  type_ = typeof(samples[1][1])
  
  n_groups = convert(type_, length(samples))
  group_sizes = convert.(type_, [size(sample, 1) for sample in samples])
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
  df_between_groups = n_groups - one(type_)

  # msd := mean of squared deviates
  msd_within_groups = ssd_within_groups / df_within_groups
  msd_between_groups = ssd_between_groups / df_between_groups

  statistic = msd_between_groups / msd_within_groups
  f_dist = Distributions.FDist(df_between_groups, df_within_groups)
  pvalue = 1 - Distributions.cdf(f_dist, statistic)

  return statistic, pvalue

end


"""Compute the ANOVA F-value for the provided sample.

Parameters
----------
X : Array{Float,2}: shape = [n_samples, n_features]
    The set of regressors that will be tested sequentially.

y : Array{Float,1}: shape(n_samples)
    The data matrix.

Returns
-------
statistic : array, shape = [n_features,]
    The set of F values.

pvalue : array, shape = [n_features,]
    The set of p-values.

"""
function ftest(X, y)

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
