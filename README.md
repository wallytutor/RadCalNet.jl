# RadCalNet

[![Build Status](https://github.com/wallytutor/RadCalNet.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/wallytutor/RadCalNet.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/wallytutor/RadCalNet.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/wallytutor/RadCalNet.jl)

Radiation properties machine learning model trained on RadCal.

## About

In this project we use the re-implementation of [RadCal](https://github.com/firemodels/radcal) to generate data and train a machine learning model for the prediction of radiative properties, *i.e.* emissivity and transmissivity, of common combustion flue gases. This is done because for real-time calls required, for instance, in CFD applications, use of RADCAL directly might be computationally prohibitive. Thus, a neural network is trained with Tensorflow based on the simulated data and then transformed into C-code that can be called from external programs (Fluent, OpenFOAM, ...). All relevant parameters required by the model are commited in this directory.
