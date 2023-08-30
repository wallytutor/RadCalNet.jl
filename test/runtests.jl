using RadCalNet.Modeling: ModelData, tests
using Flux: mae
using Test
import RadCalNet

const MODEL = RadCalNet.getradcalnet(scale = false)
const DATA = ModelData("$(dirname(@__FILE__))/../data/database.h5")
const LOSS = 0.0022

@testset "RadCalNet.jl" begin
    @test (tests(DATA, 2_000_000) |> a->mae(MODEL(a[1]), a[2])) <= LOSS
end
