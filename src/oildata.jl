function oildata()

    

    data = JLD2.load(joinpath(dirname(pathof(RBFLVM)))*"/oil.jld2")

    X = Matrix(data["T"]')
    L = data["labels"]

    @printf("Returning %d number of data items of dimension %d and corresponding labels\n", size(X,2), size(X,1))

    X, L
end