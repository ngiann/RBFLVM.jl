function loaddata(dataset)

    data = JLD2.load(joinpath(dirname(pathof(RBFLVM)))*"/"*dataset*".jld2")

    X = Matrix(data["T"]')
    L = data["labels"]

    @printf("Loading dataset %s\n", dataset)
    @printf("There are %d number of data items of dimension %d\n", size(X,2), size(X,1))
    @printf("Returning data and labels\n")

    return X, L

end

load_oil()    = loaddata("oil")
load_iris()   = loaddata("iris")
load_digits() = loaddata("digits")
