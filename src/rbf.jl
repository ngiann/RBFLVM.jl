rbf(X, r²) = rbf(X, X, r²)

function rbf(X, Y, r²)

    D² = pairwise(SqEuclidean(), X, Y, dims=2)

    exp.(- D² / (2*r²))

end