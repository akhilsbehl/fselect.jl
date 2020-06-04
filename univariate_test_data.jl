using BenchmarkTools
using DataFrames
using Distributions
using PyCall
using VegaDatasets

PyFS = pyimport("sklearn.feature_selection")

# http://facultysites.vassar.edu/lowry/PDF/c14p1.pdf
features = transpose([
            [27.0 26.2 28.8 33.5 28.8];
            [22.8 23.1 27.7 27.6 24.0];
            [21.9 23.4 20.1 27.8 19.3];
            [23.5 19.6 23.7 20.8 23.9];
           ])

statistic, prob = (6.423139965384153, 0.9953770090473866)
@assert one_way_anova(features...) == (statistic, prob)
@btime one_way_anova(features...)

iris = DataFrame(VegaDatasets.dataset("iris"))
iris_X, iris_y = Matrix(iris[:, 1:4]), Array(iris[!, 5])
@btime ftest_classification(iris_X, iris_y)

fc_answer = ftest_classification(iris_X, iris_y)
py_fc_answer = PyFS.f_classif(iris_X, iris_y)
@assert fc_answer == py_fc_answer
