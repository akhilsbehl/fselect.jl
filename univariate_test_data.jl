using BenchmarkTools
using DataFrames
using Distributions
using PyCall
using VegaDatasets

pyfs = pyimport("sklearn.feature_selection")
pypp = pyimport("sklearn.preprocessing")

# http://facultysites.vassar.edu/lowry/PDF/c14p1.pdf
features =
[
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

class_datasets =
Dict(
     :iris => Dict(
                   :X => Matrix(iris[:, 1:4]),
                   :y => Array(iris[!, 5]),
                  ),
     :seattle => Dict(
                      :X => Matrix(seattle[:, 2:5]),
                      :y => Array(seattle[!, 6]),
                     ),
     :wind => Dict(
                   :X => Matrix(wind[:, [1, 2, 5]]),
                   :y => Array(wind[!, 4]),
                  ),
     :zip => Dict(
                  :X => Matrix(zip[:, 2:3]),
                  :y => Array(zip[!, 5]),
                 ),
    )

tests_metadata =
Dict(
     :ftest => quote
       all(isapprox.(ftest_classification(pair[:X], pair[:y]),
                     pyfs.f_classif(pair[:X], pair[:y])))
     end,
     :binarize_classes => quote
       binarize_classes(pair[:y]) ==
       pypp.LabelBinarizer().fit_transform(pair[:y])
     end,
     :chisq => quote
       all(isapprox.(chisq(abs.(pair[:X]), pair[:y]),
                     pyfs.chi2(abs.(pair[:X]), pair[:y])))
     end,
    )

############################
#  Let the testing begin!  #
############################

statistic, prob = (6.423139965384153, 0.004622990952613463)
@assert one_way_anova(features...) == (statistic, prob)

for (data_label, dataset) in class_datasets
  println()
  println("============================================================")
  println("Testing for data_label: $(data_label)")
  for (funcn, expr) in tests_metadata
    global pair = dataset
    println("""$(funcn): $(eval(expr) ? "PASSED" : "FAILED")""")
  end
  println("Finished for data_label: $(data_label)")
  println("============================================================")
end
