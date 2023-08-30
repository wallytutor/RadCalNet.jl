module RadCalNet

VERSION = "v1.0.0"

include("Database.jl")
include("Modeling.jl")

using Flux
using JLD2
using RadCalNet.Modeling: defaultmodel, loadscaler

"""
    getradcalnet(;
        scale = true,
        fscaler = nothing,
        fmstate = nothing
    )

Load trained model and scaler to compose RadCalNet. If testing new
models, it might be useful to use `fscaler` and `fmstate` to point
to specific versions of scaler and model state files.
"""
function getradcalnet(;
        scale = true,
        fscaler = nothing,
        fmstate = nothing
    )
    here = "$(dirname(@__FILE__))/$(VERSION)"
    fscaler = isnothing(fscaler) ? "$(here)/scaler.yaml" : fscaler
    fmstate = isnothing(fmstate) ? "$(here)/model.jld2"  : fmstate

    model = defaultmodel()
    scaler = (scale) ? loadscaler(fscaler) : identity
    mstate = JLD2.load(fmstate, "model_state")
    Flux.loadmodel!(model, mstate)

    return (x -> x |> scaler |> model)
end

""" 
    model(x::Vector{Float32})::Vector{Float32}

Main model interface for emissivity and transmissivity.
"""
const model = getradcalnet()

end # (module RadCalNet)
