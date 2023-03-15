module NetworkPopulations

using Distributed

# Alises for referring to Paths/Interaction Sequences/Samples of Interaction Sequences (purely for readability)
include("aliases.jl")

# Types
include("Types/PathDistributions.jl")
include("Types/MarkovModels.jl")
include("Types/distributions.jl")
include("Types/hollywood_model.jl")

# Data Processing
# include("Data Processing/EdgeCounts.jl")
include("Data Processing/VertexCounts.jl")
include("Data Processing/PathSequences.jl")
include("Data Processing/Multigraphs.jl")
include("Data Processing/string_to_int.jl")
include("Data Processing/high_order_heuristic.jl")
include("Data Processing/remove_repeats.jl")
include("Data Processing/aggregate_graph.jl")
include("Data Processing/read_json.jl")
# Metric Models 
# ===============

# Model types and utils
# ---------------------
include("Metric Models/SPF.jl")
include("Metric Models/SIS.jl")
include("Metric Models/SIM.jl")
include("Metric Models/utils.jl")

# Model sampling 
# ---------------
# Types
include("Metric Models/Model Sampling/Types/initalisers.jl")
include("Metric Models/Model Sampling/Types/SIM_samplers.jl")
include("Metric Models/Model Sampling/Types/SIS_samplers.jl")
include("Metric Models/Model Sampling/Types/SPF_samplers.jl")
include("Metric Models/Model Sampling/Types/outputs.jl")
# SIS 
include("Metric Models/Model Sampling/SIS/SIS_sampler.jl")
include("Metric Models/Model Sampling/SIS/SIS_centered_simple.jl")
include("Metric Models/Model Sampling/SIS/SIS_gibbs.jl")
include("Metric Models/Model Sampling/SIS/split_merge/SIS_split_merge_helpers.jl")
include("Metric Models/Model Sampling/SIS/split_merge/SIS_split_merge.jl")
# SIM 
include("Metric Models/Model Sampling/SIM/SIM_sampler.jl")
include("Metric Models/Model Sampling/SIM/SIM_subpath.jl")
include("Metric Models/Model Sampling/SIM/SIM_gibbs.jl")
include("Metric Models/Model Sampling/SIM/SIM_length_centered.jl")
include("Metric Models/Model Sampling/SIM/SIM_proportional.jl")
# SPF 
include("Metric Models/Model Sampling/SPF/SPF_sampler.jl")
include("Metric Models/Model Sampling/SPF/SPF_dc_sampler.jl")
# Summaries 
include("Metric Models/Model Sampling/Summaries/plot_recipes.jl")
include("Metric Models/Model Sampling/Summaries/misc_summaries.jl")

# Posterior Sampling
# ------------------
# Types
include("Metric Models/Posterior Sampling/Types/SIM_samplers.jl")
include("Metric Models/Posterior Sampling/Types/SIS_samplers.jl")
include("Metric Models/Posterior Sampling/Types/SPF_samplers.jl")
include("Metric Models/Posterior Sampling/Types/outputs.jl")
include("Metric Models/Posterior Sampling/Types/predictives.jl")
# Sampler files
include("Metric Models/Posterior Sampling/auxiliary_terms_eval.jl")
include("Metric Models/Posterior Sampling/cooccurrence_matrices.jl")
include("Metric Models/Posterior Sampling/informed_insertion_distribution.jl")
include("Metric Models/Posterior Sampling/iex_SIM.jl")
include("Metric Models/Posterior Sampling/iex_SIM_split_merge.jl")
include("Metric Models/Posterior Sampling/iex_SIM_proportional.jl")
include("Metric Models/Posterior Sampling/iex_SIM_dependent.jl")
include("Metric Models/Posterior Sampling/iex_SIM_with_kick.jl")
include("Metric Models/Posterior Sampling/iex_SIS.jl")
include("Metric Models/Posterior Sampling/iex_SPF.jl")
# Summaries
include("Metric Models/Posterior Sampling/Summaries/plot_recipes.jl")
include("Metric Models/Posterior Sampling/Summaries/misc_summaries.jl")
include("Metric Models/Posterior Sampling/Summaries/predictive_summaries.jl")


# iMCMC Moves (new structure)
include("Metric Models/iMCMC Moves/imcmc_move_type.jl")
include("Metric Models/iMCMC Moves/edit_allocation_move.jl")
include("Metric Models/iMCMC Moves/permutation_move.jl")
include("Metric Models/iMCMC Moves/insert_delete_move.jl")
include("Metric Models/iMCMC Moves/split_merge_move.jl")
include("Metric Models/iMCMC Moves/mixture_move.jl")
include("Metric Models/Samplers/model_sampler.jl")
include("Metric Models/Samplers/posterior_sampler.jl")


# Graphs Models
# =============
include("Graph Models/utils.jl")
include("Graph Models/CER/CER.jl")
include("Graph Models/CER/CER_posterior.jl")

include("Graph Models/SNF/SNF.jl")
include("Graph Models/SNF/MCMC Moves/move_type.jl")
include("Graph Models/SNF/MCMC Moves/gibbs_moves.jl")
# include("Graph Models/SNF/SNF_samplers.jl")
# include("Graph Models/SNF/SNF_multigraph.jl")
# include("Graph Models/SNF/SNF_multigraph_gibbs.jl")
include("Graph Models/SNF/Samplers/SNF_model_sampler.jl")
include("Graph Models/SNF/Samplers/SNF_posterior_sampler.jl")
# include("Graph Models/SNF/SNF_multigraph_gibbs_posterior.jl")

end
