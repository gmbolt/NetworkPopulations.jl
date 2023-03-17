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
        edit_link="main",
        assets=String[],
        mathengine=Documenter.MathJax2(Dict(:TeX => Dict(
            :Macros => Dict(
                :E => ["\\mathcal{E}"],
                :G => ["\\mathcal{G}"],
                :S => ["\\mathcal{S}"],
                :bar => ["\\langle#1|", 1], # Can pass args (https://docs.mathjax.org/en/v2.7-latest/options/)
            ),
        )))
    ),
    pages=[
        "Introduction" => "index.md",
        "Standard Networks" => ["./networks/intro.md"],
        "Interaction Networks" => [
            "./interaction_networks/intro.md",
            "./interaction_networks/samplers.md"
        ],
        "Reference" => "reference.md"
    ]
)

deploydocs(;
    repo="github.com/gmbolt/NetworkPopulations.jl",
    devbranch="main"
)
