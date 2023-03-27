module RBFLVM

    using LinearAlgebra, Random, Distances, Distributions, Optim, StatsFuns

    include("rbf.jl")
    include("rbflvm.jl")
    include("generatedata.jl")

    export rbf, rbflvm, generatedata

end
