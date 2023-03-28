function generatedata()

    θ = collect(0:0.025:2π)

    A = rand(2,10)*2

    r = 1
    
    X= [sin.(θ)*r cos.(θ)*r]* A

    X .+ randn(size(X))*0.01

end