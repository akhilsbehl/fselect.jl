using BenchmarkTools
using DataFrames
using Distributions
using PyCall
using VegaDatasets

pyfs = pyimport("sklearn.feature_selection")
pypp = pyimport("sklearn.preprocessing")

# http://facultysites.vassar.edu/lowry/PDF/c14p1.pdf
features = [
            [27.0, 26.2, 28.8, 33.5, 28.8],
            [22.8, 23.1, 27.7, 27.6, 24.0],
            [21.9, 23.4, 20.1, 27.8, 19.3],
            [23.5, 19.6, 23.7, 20.8, 23.9],
           ]

statistic, prob = (6.423139965384153, 0.004622990952613463)
@assert one_way_anova(features...) == (statistic, prob)
@btime one_way_anova(features...)

iris = DataFrame(dataset("iris"))
iris_X, iris_y = Matrix(iris[:, 1:4]), Array(iris[!, 5])
@btime ftest_classification(iris_X, iris_y)

@assert all(isapprox.(ftest_classification(iris_X, iris_y),
                      pyfs.f_classif(iris_X, iris_y)))

@btime binarize_classes(iris_y)
@assert binarize_classes(iris_y) == pypp.LabelBinarizer().fit_transform(iris_y)

@btime chisq(iris_X, iris_y)
@assert all(isapprox.(chisq(iris_X, iris_y),
                      pyfs.chi2(iris_X, iris_y)))
