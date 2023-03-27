function rbflvm(Y; Q = 2, iterations = iterations)
    
    _D, N = size(Y)

    Z  = randn(2*N)

    r² = rand()*3

    β  = rand()*3

    α  = rand()*3

    rbflvm(Y, Z, r², β, α; Q = Q, iterations = iterations)

end


function rbflvm(Y, Z, r², β, α; Q = 2, iterations = iterations)

    D, N = size(Y)


    function unpack(param)

        @assert(length(param) == N*Q + 3)

        local Z  = reshape(param[1:end-3], 2, N)

        local r² = softplus(param[end-2])

        local β  = softplus(param[end-1])

        local α  = softplus(param[end-0])

        return Z, r², β, α

    end


    
    function loglikelihood(param)
        
        local Z, r², β, α = unpack(param)
        
        local Φ = rbf(Z, r²); @assert(size(Φ, 2) == size(Z, 2) == N)
        
        local ℓ = -sum(Z.^2)/2

        local density = MvNormal(zeros(eltype(param), N), I/β + (Φ'*Φ)/α + 1e-8*I)

        for d = 1:D
            ℓ += logpdf(density, Y[d,:])
        end
        
        return ℓ
        
    end


    objective(p) = -loglikelihood(p)


    # call optimiser

    initparam = [vec(Z); invsoftplus(r²); invsoftplus(β); invsoftplus(α)]

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    result = optimize(objective, initparam, LBFGS(), opt, autodiff=:forward)

    Zopt, r²opt, βopt, αopt = unpack(result.minimizer)


    # calculate posterior

    Φ     = rbf(Zopt, r²opt)
    Sinv  = αopt*I + βopt*(Φ'*Φ)
    μpost = βopt * Sinv \ (Φ' * Y')


    # calculate mean prediction

    function meanpred(z)
        
        local ϕ = rbf(z, Zopt, r²)
        
        return vec(ϕ*μpost)

    end
    

    return Zopt, meanpred # r²opt, βopt, αopt

end