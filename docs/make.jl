using NetworkPopulations
using Documenter

DocMeta.setdocmeta!(NetworkPopulations, :DocTestSetup, :(using NetworkPopulations); recursive=true)

makedocs(;
    modules=[NetworkPopulations],
    authors="George Bolt g.bolt@lancaster.ac.uk",
    repo="https://github.com/gmbolt/NetworkPopulations.jl/blob/{commit}{path}#{line}",
    sitename="NetworkPopulations.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gmbolt.github.io/NetworkPopulations.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gmbolt/NetworkPopulations.jl",
    devbranch="master",
)
