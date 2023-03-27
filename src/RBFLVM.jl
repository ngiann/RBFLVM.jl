module RBFLVM

    using Printf, PyPlot
    using LinearAlgebra, Optim, Zygote
    using Clustering, Distances
    using MultivariateStats, StatsFuns, Distributions, Random

    include("rbf.jl")
    include("rbflvm.jl")
    include("generatedata.jl")

    export rbf, rbflvm, generatedata

end
