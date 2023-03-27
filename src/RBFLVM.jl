module RBFLVM

    using Printf, PyPlot, JLD2
    using LinearAlgebra, Optim, Zygote
    using Clustering, Distances
    using MultivariateStats, StatsFuns, Distributions, Random

    # include("rbf.jl")
    include("rbflvm.jl")
    # include("generatedata.jl")
    include("oildata.jl")

    export rbflvm, oildata

end
