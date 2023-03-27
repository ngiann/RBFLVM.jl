function generatedata()


    θ = collect(0:0.08:2π)

    A = randn(2,10)

    r = 2
    
    [sin.(θ)*r cos.(θ)*r]* A



end