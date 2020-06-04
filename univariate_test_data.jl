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

########################################
#  Collect some datasets to test with  #
########################################

iris = dataset("iris") |> DataFrame
seattle = dataset("seattle-weather") |> DataFrame
wind = dataset("windvectors") |> DataFrame
zip = dataset("zipcodes") |> DataFrame

class_datasets = Dict(
                      :iris => Dict(
                                    :X => abs.(Matrix(iris[:, 1:4])),
                                    :y => Array(iris[!, 5]),
                                   ),
                      :seattle => Dict(
                                       :X => abs.(Matrix(seattle[:, 2:5])),
                                       :y => Array(seattle[!, 6]),
                                      ),
                      :wind => Dict(
                                    :X => abs.(Matrix(wind[:, [1, 2, 5]])),
                                    :y => Array(wind[!, 4]),
                                   ),
                      :zip => Dict(
                                   :X => abs.(Matrix(zip[:, 2:3])),
                                   :y => Array(zip[!, 5]),
                                  ),
                     )

statistic, prob = (6.423139965384153, 0.004622990952613463)
@assert one_way_anova(features...) == (statistic, prob)

for (set, pair) in class_datasets
  print("Testing `ftest` for set: $(set)\n")
  @assert all(isapprox.(ftest_classification(pair[:X], pair[:y]),
                        pyfs.f_classif(pair[:X], pair[:y])))
  print("Testing `binarize_classes` for set: $(set)\n")
  @assert (binarize_classes(pair[:y]) ==
           pypp.LabelBinarizer().fit_transform(pair[:y]))
  print("Testing `chisq` for set: $(set)\n")
  @assert all(isapprox.(chisq(pair[:X], pair[:y]),
                        pyfs.chi2(pair[:X], pair[:y])))
end
