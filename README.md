# RadCalNet

[![Build Status](https://github.com/wallytutor/RadCalNet.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/wallytutor/RadCalNet.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/wallytutor/RadCalNet.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/wallytutor/RadCalNet.jl)

Radiation properties machine learning model trained on RadCal.

## About

In this project we use the re-implementation of [RadCal](https://github.com/firemodels/radcal) to generate data and train a machine learning model for the prediction of radiative properties, *i.e.* emissivity and transmissivity, of common combustion flue gases. This is done because for real-time calls required, for instance, in CFD applications, use of RADCAL directly might be computationally prohibitive. Thus, a neural network is trained with Tensorflow based on the simulated data and then transformed into C-code that can be called from external programs (Fluent, OpenFOAM, ...). All relevant parameters required by the model are commited in this directory.

For details of validity ranges and sample space, please check function `datasampler!` at [database.jl](data/database.jl), where random sampling is provided. Indexing of species array is documented at `runradcalinput` in [module RadCalNet.Database](src/Database.jl).

Below we display the quality of fitting of model. One must notice that fitting of emissivity still needs a few adjustments, while transmissivity is well predicted over the whole range.

![Model testing](data/testing.png)

## Usage

The following snippet illustrates everything the model was designed to do, so don't loose your time looking for a documentation page: simply load the model and compute the required properties.

```julia
import RadCalNet

x = Float32[1200.0; 1000.0; 2.0; 1.0; 0.1; 0.2; 0.1]
y = RadCalNet.model(x)
```

The array of inputs `x` is defined below, and `y` provides gas emissitivy and transmissivity, respectively. Notice that `x` must be a column vector with entries of type `Float32`.

| Index | Quantity          | Units | Minimum | Maximum |
| :---: | :--------------:  | :---: | :-----: | :-----: |
| 1     | Wall temperature  | K     | 300     | 2500    |
| 2     | Gas temperature   | K     | 300     | 2500    |
| 3     | Depth             | m     | 0.1     | 3.0     |
| 4     | Pressure          | atm   | 0.5     | 1.5     |
| 5     | CO2 mole fraction | -     | 0.0     | 0.25    |
| 6     | H2O mole fraction | -     | 0.0     | 0.30    |
| 7     | CO  mole fraction | -     | 0.0     | 0.20    |

## To-do's

- [ ] Broaden sample space over the whole RadCal composition spectrum.
- [ ] Define data loading on GPU/CPU though a flag when recovering model.
- [ ] Make `Float64` interface and/or compatibility.
- [ ] Create database for testing outside of sampling points.
