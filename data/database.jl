# -*- coding: utf-8 -*-
# Database creation for RADCAL
#
# This file contains the required tooling for generating a database
# for radiation properties model conception based on RadCal. The sample
# space of the current model is found below in `datasampler!`.

import Random
import RadCalNet.Database as rd

# Provide a seed at start-up for *maybe* reproducible builds.
Random.seed!(42)

"""
    datasampler!(X::Vector{Float64})::Tuple

Custom sample space to generate entries with `createcustomdatabase`.
"""
function datasampler!(X::Vector{Float64})::Tuple
    X[1] = rand(0.0:0.01:0.25)
    X[2] = rand(0.0:0.01:0.30)
    X[3] = rand(0.0:0.01:0.20)
    X[end] = 1.0 - sum(X[1:3])

    T = rand(300.0:10.0:2500.0)
    L = rand(0.1:0.1:3.0)
    P = rand(0.5:0.5:1.5)
    FV = 0.0
    TWALL = rand(300.0:10.0:2500.0)

    return T, L, P, FV, TWALL
end

"""
    sampledatabase()

Create a sample database for usage instruction and verification.
"""
function sampledatabase()
    testname = "test.dat"

    rd.createcustomdatabase(;
        sampler!   = datasampler!,
        repeats    = 3,
        samplesize = 3,
        cleanup    = true,
        saveas     = testname,
        override   = true
    )

    A = rd.loaddatabase(testname)

    isfile(testname) && rm(testname)

    return A
end

A = sampledatabase()

# If random seed is working this should be the output (truncated).
#
# 9×26 Matrix{Float32}:
# …  1650.0  1850.0  2.1  0.5  0.0  …  0.127997       1.4369f5   0.784483
#     890.0  1760.0  2.0  1.0  0.0     0.256072   52761.9        0.493376
#     470.0   720.0  1.3  1.0  0.0     0.209679    1712.82       0.743175
#    2190.0  1170.0  1.0  1.0  0.0     0.0917254      3.80101f5  0.88582
#     360.0   640.0  2.2  1.0  0.0     0.418247    1442.63       0.479691
# …  1320.0  1870.0  0.5  1.0  0.0  …  0.0997078  71325.7        0.79875
#    1970.0  2500.0  1.1  1.5  0.0     0.0812446      3.0696f5   0.815462
#     540.0  1510.0  1.9  1.0  0.0     0.253691   24945.7        0.478712
#    1100.0   780.0  2.0  0.5  0.0     0.234852   21784.6        0.7443
#
# Uncomment the next line for database generation with defaults.
# createcustomdatabase(; sampler! = datasampler!, cleanup = true)
