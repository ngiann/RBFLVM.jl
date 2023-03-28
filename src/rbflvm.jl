"""
# Basic use:

    pred, centres, Z = rbflvm(X; iterations=1000, M=30, Q = 2, initmode=:pca)


# Example

```julia-repl
julia> X,labels = oildata();
julia> pred, c, Z = rbflvm(X; iterations=1000, M=30, Q = 2, initmode=:pca)
julia> using PyPlot # must be independently installed
julia> for l in unique(labels)
            idx = findall(labels .== l)
            plot(Z[1,idx],Z[2,idx],"o")
       end
```
"""
function rbflvm(Y; Q = 2, iterations = 1, M = 10, JITTER = 1e-8, initmode=:pca, seed = 1)

        @assert(initmode == :pca || initmode == :random)

        rg = MersenneTwister(seed)

        D = size(Y, 1)
        N = size(Y, 2)
        σ₀² = 100.0 # prior on weights
        
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
    
            local σ², r = exp(param[end-1]), param[end]
    
            MARK +=2 
    
            @assert(MARK == length(param))
   
            return X, centres, σ², r
    
        end
    
    
        #------------------------------------------------#
        function marginalloglikelihood(X, centres, σ², r)
        #------------------------------------------------#

            local Φ = designmatrix(X, centres, r)
            
            local ℓ = zero(eltype(X))
    
            local lik = MvNormal(zeros(N), (σ² + JITTER)*I + (Φ*Φ')/σ₀²)
            
            for d in 1:D
            
                ℓ += logpdf(lik, Y[d,:])
            
            end
    
            return ℓ - 1e-6*sum(X.^2)
    
        end
    
        #-----------------------------------------------------#
        function marginalloglikelihood_fast(X, centres, σ², r)
        #-----------------------------------------------------#
    
            local Φ = designmatrix(X, centres, r)
    
            local ℓ = zero(eltype(X))
    
            # Σ = (σ² + JITTER)*I + (Φ*Φ')/σ₀²
    
           local Σinv, logdetΣ = let
    
                α = 1/(σ² + JITTER)
    
                ΦTΦ = Φ'*Φ
                
                α*I - α*I * (1/σ₀²) * Φ*((I + (α/σ₀²)*ΦTΦ) \ (Φ'*(α*I))),
    
                N*log(σ² + JITTER) + logdet(I + (α/σ₀²)*ΦTΦ)
    
            end
    
            ℓ += - 0.5*N*D*log(2π) - 0.5*D*logdetΣ - 0.5*tr(Y*(Σinv*Y'))
    
            return ℓ - 1e-6*sum(X.^2)#pairwise(SqEuclidean(), X, centres))
    
        end
    
    
        #--------------------------------------------
        # Run optimisation
        #--------------------------------------------
       
        initp = [vec(X);vec(producecentres(X, M));randn(rg, 2)*3]
    
        ############ NUMERICAL TEST ############
        # @time @show marginalloglikelihood(unpack(initp)...)
        # @time @show marginalloglikelihood_fast(unpack(initp)...)
        # return 
    
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
    
        [exp.(-D²/(2*r^2)) ones(N)]
    
    end
        