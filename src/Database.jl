module Database

using DelimitedFiles
using HDF5
using Plots
using Printf

export runradcalinput
export createcustomdatabase
export loaddatabase

"""
    runradcalinput(;
        X::Dict{String, Float64} = Dict{String, Float64}(),
        T::Float64 = 300.0,
        L::Float64 = 1.0,
        P::Float64 = 1.0,
        FV::Float64 = 0.0,
        OMMIN::Float64 = 50.0,
        OMMAX::Float64 = 10000.0,
        TWALL::Float64 = 500.0,
        radcalexe::String = "radcal_win_64.exe"
    )::Vector{Float64}

Create RADCAL.IN from template file and dump to disk.

**NOTE:** the user is responsible to provide a vector `X` of mole
fractions of species that sums up to one. If this is not respected
RADCAL fails. The Following list provides the indexes of available
species in vector `X`.

| Index | Species | Index | Species | Index | Species |
| ----: | :------ | ----: | :------ | ----: | :------ |
| 1     | CO2     | 6     | C2H6    | 11    | CH3OH   |
| 2     | H2O     | 7     | C3H6    | 12    | MMA     |
| 3     | CO      | 8     | C3H8    | 13    | O2      |
| 4     | CH4     | 9     | C7H8    | 14    | N2      |
| 5     | C2H4    | 10    | C7H16   |       |         |
"""
function runradcalinput(;
        X::Vector{Float64},
        T::Float64 = 300.0,
        L::Float64 = 1.0,
        P::Float64 = 1.0,
        FV::Float64 = 0.0,
        OMMIN::Float64 = 50.0,
        OMMAX::Float64 = 10000.0,
        TWALL::Float64 = 500.0,
        radcalexe::String = "radcal_win_64.exe"
    )::Vector{Float64}
    case = """\
        CASE:
        &HEADER TITLE="CASE" CHID="CASE" /
        &BAND OMMIN = $(@sprintf("%.16E", OMMIN))
              OMMAX = $(@sprintf("%.16E", OMMAX)) /
        &WALL TWALL = $(@sprintf("%.16E", TWALL)) /
        &PATH_SEGMENT
            T        = $(@sprintf("%.16E", T))
            LENGTH   = $(@sprintf("%.16E", L))
            PRESSURE = $(@sprintf("%.16E", P))
            XCO2     = $(@sprintf("%.16E", X[1]))
            XH2O     = $(@sprintf("%.16E", X[2]))
            XCO      = $(@sprintf("%.16E", X[3]))
            XCH4     = $(@sprintf("%.16E", X[4]))
            XC2H4    = $(@sprintf("%.16E", X[5]))
            XC2H6    = $(@sprintf("%.16E", X[6]))
            XC3H6    = $(@sprintf("%.16E", X[7]))
            XC3H8    = $(@sprintf("%.16E", X[8]))
            XC7H8    = $(@sprintf("%.16E", X[9]))
            XC7H16   = $(@sprintf("%.16E", X[10]))
            XCH3OH   = $(@sprintf("%.16E", X[11]))
            XMMA     = $(@sprintf("%.16E", X[12]))
            XO2      = $(@sprintf("%.16E", X[13]))
            XN2      = $(@sprintf("%.16E", X[14]))
            FV       = $(@sprintf("%.16E", FV)) /\
        """

    write("RADCAL.IN", case)

    run(`$radcalexe RADCAL.in`)

    data = readlines("RADCAL.OUT")

    @assert !startswith(data[1], "ERROR") "RadCal ailed: $(data[1])"

    f = r->(r |> strip |> s->split(s, "\t") |> last |> s->parse(Float64, s))

    return [OMMIN, OMMAX, TWALL, T, L, P, FV, X..., map(f, data[5:9])...]
end

"""
    createcustomdatabase(;
        sampler!::Function,
        repeats::Int64 = 100,
        samplesize::Int64 = 50_000,
        cleanup::Bool = false,
        saveas::String = "database.h5",
        OMMIN::Float64 = 50.0,
        OMMAX::Float64 = 10000.0,
        override::Bool = false
    )

Creates a custom database by generating a number `repeats` of samples
of `samplesize` rows. Inputs for `runradcalinput` are to be generated
by a `sampler!` user-defined function which modifies in place an array
of compositions, and returns `T`, `L`, `P`, `FV`, `TWALL` for setting
up a simulation. Files are temporarilly stored under `data/` with a
sequential numbered naming during database creation and aggregated
in a HDF5 file named after `saveas`. The choice to aggregate files
after an initial dump is because generation can be interrupted and
manually recovered in an easier way and avoiding any risk of data
losses - database creation can take a very long time. If `cleanup` is
`true`, all intermediate files are removed.
"""
function createcustomdatabase(;
        sampler!::Function,
        repeats::Int64 = 100,
        samplesize::Int64 = 50_000,
        cleanup::Bool = false,
        saveas::String = "database.h5",
        OMMIN::Float64 = 50.0,
        OMMAX::Float64 = 10000.0,
        override::Bool = false
    )
    if isfile(saveas) && !override
        @warn ("Database $(saveas) already exists.")
        return
    end

    X = zeros(14)
    table = zeros(samplesize, 26)

    !isdir("tmp") && mkdir("tmp")
    fs = open("tmp/tmp.csv", "a")

    for fnum in 1:repeats
        for k in 1:samplesize
            T, L, P, FV, TWALL = sampler!(X)

            try
                result = runradcalinput(;
                    X = X,
                    T = T,
                    L = L,
                    P = P,
                    FV = FV,
                    TWALL = TWALL,
                    OMMIN = OMMIN,
                    OMMAX = OMMAX
                )

                table[k, 1:end] = result
            catch e
                println(e)
            end
        end

        fname = "tmp/block$(fnum).csv"
        writedlm(fname,  table, ',')
        write(fs, read(fname))
    end

    close(fs)

    h5open(saveas, "w") do fid
        data = readdlm("tmp/tmp.csv", ',', Float32)
        g = create_group(fid, "data")
        dset = create_dataset(g, "table", Float32, size(data))
        write(dset, data)
    end

    rm("RADCAL.IN")
    rm("RADCAL.OUT")
    rm("TRANS_CASE.TEC")

    if cleanup
        rm("tmp/"; force = true, recursive = true)
    end
end

"""
    loaddatabase(fname::String)

Retrieve database from HDF5 file and access table as a matrix.
"""
function loaddatabase(fname::String)
    h5open(fname, "r") do fid
        return read(fid["data"]["table"])
    end
end

end # (module Database)