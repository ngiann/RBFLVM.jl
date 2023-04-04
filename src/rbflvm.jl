"""
# Basic use:

    pred, centres, Z = rbflvm(X; iterations=1000, M=30, Q = 2, initmode=:pca)


# Example

```julia-repl
julia> X, labels = load_oil();
julia> pred, c, Z = rbflvm(X; iterations=1000, M=30, Q = 2, initmode=:pca)
julia> using PyPlot # must be independently installed
julia> for l in unique(labels)
            idx = findall(labels .== l)
            plot(Z[1,idx],Z[2,idx],"o")
       end
```
"""
function rbflvm(Y; Q = 2, iterations = 1, M = 10, JITTER = 0.0, η=0.0, initmode=:pca, seed = 1)

        @assert(initmode == :pca || initmode == :random)

        rg = MersenneTwister(seed)

        D = size(Y, 1)
        N = size(Y, 2)
        α = 1/1000.0 # prior precision on weights
        
        @printf("Running %dD-rbflvm with %d data items of dim %d\n", Q, N, D)
        @printf("\t initialising in %s mode\n", string(initmode))
        @printf("\t random seed is %d\n", seed)

        #------------------------------------------#
        # initialise latent coordinates x with pca #
        #------------------------------------------#
    
        X = initmode == :pca ?  
            MultivariateStats.predict(MultivariateStats.fit(PCA, Y, maxoutdim = Q) , Y) : 0.1*randn(rg, Q, N)

    
        #------------------------------------------#
        function unpack(param)
        #------------------------------------------#
  
            local MARK = 0
    
            local X = reshape(param[MARK+1:MARK+Q*N], Q, N)
    
            MARK += Q*N
    
            local centres = reshape(param[MARK+1:MARK+Q*M], Q, M)
    
            MARK += Q*M
    
            local β, r = 1/exp(param[end-1]), param[end]
    
            MARK +=2 
    
            @assert(MARK == length(param))
   
            return X, centres, β, r
    
        end
    
    
        #------------------------------------------------#
        function marginalloglikelihood(X, centres, β, r)
        #------------------------------------------------#

            local Φ = designmatrix(X, centres, r)
            
            local ℓ = zero(eltype(X))

            local lik = MvNormal(zeros(N), (1/β)*I + (Φ*Φ')/α)
            
            for d in 1:D
            
                ℓ += logpdf(lik, Y[d,:])
            
            end
    
            return ℓ - η*sum(X.^2)
    
        end
    
        #-----------------------------------------------------#
        function marginalloglikelihood_fast(X, centres, β, r)
        #-----------------------------------------------------#
    
            local Φ = designmatrix(X, centres, r)
    
            local ℓ = zero(eltype(X))
    
            # local Σ = I/β + (Φ*Φ')/α
    
           local Σinv, logdetΣ = let
    
                ΦTΦ = Φ'*Φ
                
                β*I - β*I * Φ*((I*α + β*ΦTΦ) \ (Φ'*(β*I))),
    
                -N*log(β) - M*log(α) + logdet(I*α + β*ΦTΦ)
    
            end
    
            ℓ += - 0.5*N*D*log(2π) - 0.5*D*logdetΣ - 0.5*tr(Y*(Σinv*Y'))
    
            return ℓ - η*sum(X.^2)
    
        end

        #-----------------------------------------------------#
        function calculatevariationalposterior(X, centres, β, r)
        #-----------------------------------------------------#
    
            local Φ = designmatrix(X, centres, r)

            local C⁻¹ = α*I + β*(Φ'*Φ) # Note: same for all dimensions

            local μ = β * (C⁻¹\(Φ'*Y'))

            
            return μ, C⁻¹

        end
    

        #-----------------------------------------------------#
        function lowerbound(X, centres, β, r, μ, C⁻¹)
        #-----------------------------------------------------#
    
            local Φ = designmatrix(X, centres, r)

            local lb = zero(eltype(centres))
           
            local C = (C⁻¹)\I; C= (C + C')/2

            
            # contribution from likelihood

            lb += -0.5*(N*D)*log(2π) + 0.5*(N*D)*log(β) 
           
            lb += -0.5 * β * sum(abs2.(Y' - Φ*μ))

            lb += D * (-0.5 * β * tr(Φ*C*Φ'))


            # contribution from prior

            lb += -0.5*(M*D)*log(2π) + 0.5*(M*D)*log(α)
            
            lb += -0.5 * α * sum(abs2.(μ))

            lb += D * (-0.5 * α * tr(C))


            # contribution from entropy

            lb += D*M*0.5*(1+log(2π)) + 0.5*D*logdet(C)

            # add penalty on latent coordinates

            return lb - η*sum(X.^2)

        end
    
    
        #--------------------------------------------
        # Run optimisation
        #--------------------------------------------
       
        initp = [vec(X);vec(producecentres(X, M));randn(rg, 2)*3]
    
        ############ NUMERICAL TEST ############
        let
            @show marginalloglikelihood(unpack(initp)...)
            @show marginalloglikelihood_fast(unpack(initp)...)
            
            
            local μ, C⁻¹ = calculatevariationalposterior(unpack(initp)...)
            @show lowerbound(unpack(initp)..., μ, C⁻¹)
            return 
        end
    
        objective(x) = -marginalloglikelihood_fast(unpack(x)...)
    
        gradobjective!(s, x) = copyto!(s, Zygote.gradient(objective, x)[1])

        opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 10)
    
        result = optimize(objective, gradobjective!, initp, LBFGS(), opt)
    
        Xopt, centresopt, σ²opt, ropt = unpack(result.minimizer)
        
      
        #--------------------------------------------
        # Calculate posterior of weights
        #--------------------------------------------
    
        Φ = designmatrix(Xopt, centresopt, ropt)
    
        Σpostinv = (1/σ₀²)*I + (1/σ²opt)*Φ'*Φ # Eq. 3.54 in PRML
    
        μpost    = (1/σ²opt)*(Σpostinv\(Φ'*Y'))  # Eq. 3.53 in PRML
    
    
    
        #--------------------------------------------
        # Plot projections
        #--------------------------------------------
    
        # if Q == 2
        #     figure(1); cla(); plot(Xopt[1,:], Xopt[2,:], "o"); axis("equal")
        # elseif Q == 3
        #     figure(1); cla(); plot3D(Xopt[1,:], Xopt[2,:], Xopt[3,:], "o"); axis("equal")
        # end
    
        #------------------------------------------------#
        function predict(X₊)
        #------------------------------------------------#
    
            local Φ₊ = designmatrix(X₊, centresopt, ropt)
    
            local μpred = Φ₊ * μpost                  # Eq. 3.58 in PRML
    
            local Σpred = σ²opt*I + Φ₊*(Σpostinv\Φ₊') # Eq. 3.59 in PRML
    
            return μpred, Σpred
    
        end
    
    
        return predict, centresopt, Xopt
    
    end
    
    
    #-------------------------------------------#
    function producecentres(X, K)
    #-------------------------------------------#
    
        kmeans(X, K, maxiter = 10_000).centers 
    
    end
    
    
    #-------------------------------------------#
    function designmatrix(X, centres, r::T)  where T<:Real
    #-------------------------------------------#
        
        N = size(X, 2)
    
        D² = pairwise(SqEuclidean(), X, centres)
    
        return exp.(-D²/(2*r^2))
        [exp.(-D²/(2*r^2)) ones(N)]
    
    end
        