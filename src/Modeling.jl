module Modeling

using cuDNN
using CUDA
using DelimitedFiles
using DocStringExtensions
using JLD2
using Flux
using HDF5
using Plots
using Printf
using ProgressMeter
using Statistics
using StatsBase
using YAML

using RadCalNet.Database: loaddatabase

# Do not even bother working without a GPU
# @assert CUDA.functional()

# Set graphics backend.
gr()

# Type aliases for non-default types.
M32 = Matrix{Float32}
V32 = Vector{Float32}

"""
    ModelData(fpath::String; f_train::Float64 = 0.7)

Load HDF5 database stored under `fpath` and performs standardized workflow
of data preparation for model training. The data is split under training and
testing datasets with a fraction of training data of `f_train`.

$(TYPEDFIELDS)
"""
struct ModelData
    "Scaler used for data transformation."
    scaler::ZScoreTransform{Float32,V32}

    "Matrix of training input data."
    X_train::M32

    "Matrix of training output data."
    Y_train::M32

    "Matrix of testing input data."
    X_tests::M32

    "Matrix of testing output data."
    Y_tests::M32

    "Number of model inputs."
    n_inputs::Int64

    "Number of model outputs."
    n_outputs::Int64

    function ModelData(fpath::String; f_train::Float64 = 0.7)
        # Read database and drop eventual duplicates.
        data = unique(loaddatabase(fpath), dims=1)

        # Index of predictors X and targets Y.
        X_cols = [3, 4, 5, 6, 8, 9, 10]
        Y_cols = [24, 26]

        # Compute index at which data must be split.
        split = (f_train * size(data)[1] |> round |> x -> convert(Int64, x))

        # Perform train-test split.
        train, tests = data[1:split, 1:end], data[split+1:end, 1:end]

        # Get matrices for predictors and targets.
        X_train, Y_train = train[1:end, X_cols], train[1:end, Y_cols]
        X_tests, Y_tests = tests[1:end, X_cols], tests[1:end, Y_cols]

        # Frameworks expected entries per column.
        X_train, Y_train = transpose(X_train), transpose(Y_train)
        X_tests, Y_tests = transpose(X_tests), transpose(Y_tests)

        # Apply standard scaling to data.
        scaler = fit(ZScoreTransform, X_train, dims=2)
        X_train = StatsBase.transform(scaler, X_train)
        X_tests = StatsBase.transform(scaler, X_tests)

        return new(scaler, X_train, Y_train, X_tests, Y_tests,
                   length(X_cols), length(Y_cols))
    end
end

"""
    ModelTrainer(
        data::ModelData,
        model::Chain;
        batch::Int64=64,
        epochs::Int64=100,
        η::Float64=0.001,
        β::Tuple{Float64,Float64}=(0.9, 0.999),
        ϵ::Float64=1.0e-08
    )

Holds standardized model training parameters and data.

$(TYPEDFIELDS)
"""
struct ModelTrainer
    "Batch size in training loop."
    batch::Int64

    "Number of epochs to train each time."
    epochs::Int64

    "Database structure used for training/testing."
    data::ModelData

    "Multi-layer perceptron used for modeling."
    model::Chain

    "Internal Adam optimizer."
    optim::NamedTuple

    "History of losses."
    losses::V32

    function ModelTrainer(
        data::ModelData,
        model::Chain;
        batch::Int64=64,
        epochs::Int64=100,
        η::Float64=0.001,
        β::Tuple{Float64,Float64}=(0.9, 0.999),
        ϵ::Float64=1.0e-08
    )
        model = model |> gpu
        optim = Flux.setup(Flux.Adam(η, β, ϵ), model)
        new(batch, epochs, data, model, optim, Float32[])
    end
end

"""
    dumpscaler(scaler::ZScoreTransform{Float32,V32}, saveas::String)

Write z-score `scaler` mean and scale to provided `saveas` YAML file.
"""
function dumpscaler(scaler::ZScoreTransform{Float32,V32}, saveas::String)
    data = Dict("mean" => scaler.mean, "scale" => scaler.scale)
    YAML.write_file(saveas, data)
end

"""
    loadscaler(fname::String)::Function

Load z-scaler in functional format from YAML `fname` file.
"""
function loadscaler(fname::String)::Function
    scaler = YAML.load_file(fname)
    μ = convert(Vector{Float32}, scaler["mean"])
    σ = convert(Vector{Float32}, scaler["scale"])
    return (x -> @. (x - μ) / σ)
end

"""
    makemodel(layers::Vector{Tuple{Int64, Any}}; bn = false)::Chain

Create a multi-layer perceptron for learning radiative properties
with the provided `layers`. If `bn` is true, then batch normalization
after each layer. The final layer has by default a sigmoid function
to ensure physical outputs in range [0, 1].
"""
function makemodel(layers::Vector{Tuple{Int64, Any}}; bn = false)::Chain
    chained = []

    for (a, b) in zip(layers[1:end-1], layers[2:end])
        push!(chained, Dense(a[1] => b[1], a[2]))
        if bn
            push!(chained, BatchNorm(b[1]))
        end
    end

    return Chain(chained..., sigmoid)
end

"""
    defaultmodel()

Build model structure with which RadCalNet is trained.
"""
function defaultmodel()
    return makemodel([
        (7,   identity),
        (100, leakyrelu),
        (50,  leakyrelu),
        (20,  leakyrelu),
        (20,  leakyrelu),
        (10,  leakyrelu),
        (10,  leakyrelu),
        (5,   leakyrelu),
        (2,   ())
    ])
end

""" Get sample of indexes for data retrieval.  """
function samplecols(nmax::Int64, num::Int64)::Vector{Int64}
  return rand(1:nmax, min(num, nmax))
end

""" Get training data for data loader construction. """
function train(data::ModelData, num::Int64)::Tuple{M32,M32}
    cols = samplecols(size(data.X_train)[2], num)
    return (data.X_train[:, cols], data.Y_train[:, cols])
end

""" Get testing data for data loader construction. """
function tests(data::ModelData, num::Int64)::Tuple{M32,M32}
  cols = samplecols(size(data.X_tests)[2], num)
  return (data.X_tests[:, cols], data.Y_tests[:, cols])
end

"""
    trainonce!(trainer::ModelTrainer; num = 1_000)

Train model and keep track of loss for the number of epochs in `trainer`
using its internal data and parameters. Use `num` data points.
"""
function trainonce!(trainer::ModelTrainer; num = 1_000)::V32
    loader = Flux.DataLoader(train(trainer.data, num) |> gpu,
                             batchsize=trainer.batch, shuffle=false)

    @showprogress for _ in 1:trainer.epochs
        for (x, y) in loader
            loss, grads = Flux.withgradient(trainer.model) do m
                Flux.mse(m(x), y)
            end
            Flux.update!(trainer.optim, trainer.model, grads[1])
            push!(trainer.losses, loss)
        end
    end

    return trainer.losses
end

"""
    plottests(trainer::ModelTrainer; num::Int64)

Evaluate model over `num` data points and compare the data to
the expected values as computed from RadCal. Makes use of test
data only - never seem by the model during training.
"""
function plottests(trainer::ModelTrainer; num::Int64)
    X_tests, Y_tests = tests(trainer.data, num)
    Y_preds = trainer.model(X_tests |> gpu) |> cpu

    εt, εp = Y_tests[1, :], Y_preds[1, :]
    τt, τp = Y_tests[2, :], Y_preds[2, :]

    εlims = [0.0, 0.5]
    τlims = [0.0, 1.0]

    l = @layout [a b]

    pε = scatter(εt, εp, xlims=εlims, ylims=εlims,
                 xlabel="ε from RadCal",
                 ylabel="ε from neural network",
                 label=nothing,
                 markerstrokewidth=0.0,
                 markeralpha=0.5,
                 markercolor="#000000")

    pτ = scatter(τt, τp, xlims=τlims, ylims=τlims,
                 xlabel="τ from RadCal",
                 ylabel="τ from neural network",
                 label=nothing,
                 markerstrokewidth=0.0,
                 markeralpha=0.5,
                 markercolor="#000000")

    plot!(pε, εlims, εlims, color="#FF0000", label=nothing)
    plot!(pτ, τlims, τlims, color="#FF0000", label=nothing)

    return plot(pε, pτ, layout=l, size=(1200, 600),
                margin=5Plots.mm)
end

end # (module Modeling)