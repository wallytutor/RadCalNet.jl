# -*- coding: utf-8 -*-
using RadCalNet.Modeling: ModelData, ModelTrainer
using RadCalNet.Modeling: dumpscaler, defaultmodel
using RadCalNet.Modeling: trainonce!, plottests, tests
using Flux
using JLD2
using Plots
import Random
import RadCalNet

""" Minimal loss plot in logarithmic scale. """
function plotlosses(losses)
    plot(losses;
         xaxis=(:log10, "Iteration"),
         yaxis=(:log10, "Loss"),
         label="Per batch")
end

""" Dump trained model to JLD2 file for deployment. """
function dumpmodel(trainer::ModelTrainer, saveas::String)
    model = trainer.model |> cpu
    jldsave(saveas; model_state = Flux.state(model))
end

# Provide a seed at start-up for *maybe* reproducible builds.
Random.seed!(42)

here = dirname(@__FILE__)
fscaler = "$(here)/scaler.yaml"
fmstate = "$(here)/model.jld2"

data = ModelData("$(here)/database.h5")
dumpscaler(data.scaler, fscaler)

model = defaultmodel()
trainer = ModelTrainer(data, model, batch = 10_000, epochs = 100)

Flux.adjust!(trainer.optim, 0.01)
trainonce!(trainer; num = 10_000)

Flux.adjust!(trainer.optim, 0.001)
trainonce!(trainer; num = 1_000_000)
trainonce!(trainer; num = 1_000_000)
trainonce!(trainer; num = 1_000_000)

Flux.adjust!(trainer.optim, 0.0001)
trainonce!(trainer; num = 100_000) |> plotlosses

plottests(trainer; num = 10_000)

X_tests, Y_tests = tests(trainer.data, 2_000_000)
error = Flux.mae(trainer.model(X_tests |> gpu) |> cpu, Y_tests)

dumpmodel(trainer, fmstate)

# Test dumped version of model.
radcal = RadCalNet.getradcalnet(
    scale = false,
    fscaler = fscaler,
    fmstate = fmstate
)
error = Flux.mae(radcal(X_tests), Y_tests)
