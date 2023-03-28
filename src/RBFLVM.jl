module RBFLVM

    using Printf, JLD2
    using LinearAlgebra, Optim, Zygote
    using Clustering, Distances
    using MultivariateStats, StatsFuns, Distributions, Random

    include("rbflvm.jl")
    include("loaddata.jl")

    export rbflvm, load_oil, load_iris, load_digits

end
