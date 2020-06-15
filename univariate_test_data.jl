using BenchmarkTools
using CSV
using DataFrames
using Distributions
using PyCall
using VegaDatasets

pyfs = pyimport("sklearn.feature_selection")
pypp = pyimport("sklearn.preprocessing")

########################################
#  Collect some datasets to test with  #
########################################

class_datasets = ["iris", "seattle-weather", "windvectors", "zipcodes"]
# cd := class_datasets
cd_splits = [[1:4, 5], [2:5, 6], [[1, 2, 5], 4], [2:3, 5]]

class_data = Dict()
for (dname, dsplit) in Iterators.zip(class_datasets, cd_splits)
  df = DataFrame(dataset(dname))
  X, y = Matrix(df[!, dsplit[1]]), df[!, dsplit[2]]
  class_data[dname] = Dict(:X => X, :y => y)
end

# https://github.com/vincentarelbundock/Rdatasets
reg_datasets = ["freedman", "highway", "salaries"]
# rd := reg_datasets
rd_splits = [[2, 3], 4], [3:11, 2], [4:5, 7]]

reg_data = Dict()
for (dname, dsplit) in Iterators.zip(reg_datasets, cd_splits)
  println(dname)
  df = DataFrame(CSV.read("data/$(dname).csv"))
  X, y = Matrix(df[!, dsplit[1]]), df[!, dsplit[2]]
  reg_data[dname] = Dict(:X => X, :y => y)
end

tests_functions =
Dict(
     :ftest_classification =>
     (X, y) -> all(isapprox.(ftest_classification(X, y),
                             pyfs.f_classif(X, y))),
     :binarize_classes =>
     (X, y) -> (binarize_classes(y) == pypp.LabelBinarizer().fit_transform(y)),
     :chisq =>
     (X, y) -> all(isapprox.(chisq(abs.(X), y), pyfs.chi2(abs.(X), y))),
    )

############################
#  Let the testing begin!  #
############################

for (data_label, dataset) in class_data
  println()
  println("============================================================")
  println("Testing for data set: $(data_label)")
  for (test_name, test_func) in tests_functions
    check = test_func(dataset[:X], dataset[:y])
    println("""$(test_name): $(check ? "PASSED" : "FAILED")""")
  end
  println("Finished for data set: $(data_label)")
  println("============================================================")
end

for (data_label, dataset) in reg_data
  println()
  println("============================================================")
  println("Testing for data set: $(data_label)")
  for (test_name, test_func) in tests_functions
    check = test_func(dataset[:X], dataset[:y])
    println("""$(test_name): $(check ? "PASSED" : "FAILED")""")
  end
  println("Finished for data set: $(data_label)")
  println("============================================================")
end

