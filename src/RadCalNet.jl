module RadCalNet

# XXX: this is the version of the trained model, not the package!
MODELVERSION = "v1.0.0"

include("Database.jl")
include("Modeling.jl")

using Flux
using JLD2
using RadCalNet.Modeling: defaultmodel, loadscaler

const RADCALROOT = "$(dirname(@__FILE__))/$(MODELVERSION)"
const FILESCALER = "$(RADCALROOT)/scaler.yaml"
const FILEMODEL  = "$(RADCALROOT)/model.jld2"

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
    fscaler = isnothing(fscaler) ? FILESCALER : fscaler
    fmstate = isnothing(fmstate) ? FILEMODEL  : fmstate

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
