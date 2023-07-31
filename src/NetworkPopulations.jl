module NetworkPopulations

# Alises for referring to Paths/Interaction Sequences/Samples of Interaction Sequences (purely for readability)
include("aliases.jl")

# Types
include("Types/PathDistributions.jl")
include("Types/MarkovModels.jl")
include("Types/distributions.jl")

# Data Processing Helpers
# include("data_processing/EdgeCounts.jl")
# include("data_processing/VertexCounts.jl")
# include("data_processing/PathSequences.jl")
include("data_processing/multigraphs.jl")
include("data_processing/string_to_int.jl")
include("data_processing/high_order_heuristic.jl")
include("data_processing/remove_repeats.jl")
include("data_processing/aggregate_graph.jl")
include("data_processing/read_json.jl")
# Metric Models 
# ===============

# Model types and utils
# ---------------------
include("inter_network_models/SPF.jl")
include("inter_network_models/SIS.jl")
include("inter_network_models/SIM.jl")
include("inter_network_models/utils.jl")


# Model sampling 
# ---------------
# Types
# include("inter_network_models/Model Sampling/Types/initalisers.jl")
# include("inter_network_models/Model Sampling/Types/SIM_samplers.jl")
# include("inter_network_models/Model Sampling/Types/SIS_samplers.jl")
# include("inter_network_models/Model Sampling/Types/SPF_samplers.jl")
# include("inter_network_models/Model Sampling/Types/outputs.jl")
# SIS 
# include("inter_network_models/Model Sampling/SIS/SIS_sampler.jl")
# include("inter_network_models/Model Sampling/SIS/SIS_centered_simple.jl")
# include("inter_network_models/Model Sampling/SIS/SIS_gibbs.jl")
# include("inter_network_models/Model Sampling/SIS/split_merge/SIS_split_merge_helpers.jl")
# include("inter_network_models/Model Sampling/SIS/split_merge/SIS_split_merge.jl")
# SIM 
# include("inter_network_models/Model Sampling/SIM/SIM_sampler.jl")
# include("inter_network_models/Model Sampling/SIM/SIM_subpath.jl")
# include("inter_network_models/Model Sampling/SIM/SIM_gibbs.jl")
# include("inter_network_models/Model Sampling/SIM/SIM_length_centered.jl")
# include("inter_network_models/Model Sampling/SIM/SIM_proportional.jl")
# SPF 
# include("inter_network_models/Model Sampling/SPF/SPF_sampler.jl")
# include("inter_network_models/Model Sampling/SPF/SPF_dc_sampler.jl")
# Summaries 
# include("inter_network_models/Model Sampling/Summaries/plot_recipes.jl")
# include("inter_network_models/Model Sampling/Summaries/misc_summaries.jl")

# Posterior Sampling
# ------------------
# Types
# include("inter_network_models/Posterior Sampling/Types/SIM_samplers.jl")
# include("inter_network_models/Posterior Sampling/Types/SIS_samplers.jl")
# include("inter_network_models/Posterior Sampling/Types/SPF_samplers.jl")
# include("inter_network_models/Posterior Sampling/Types/outputs.jl")
# include("inter_network_models/Posterior Sampling/Types/predictives.jl")
# # Sampler files
# include("inter_network_models/Posterior Sampling/auxiliary_terms_eval.jl")
# include("inter_network_models/Posterior Sampling/cooccurrence_matrices.jl")
# include("inter_network_models/Posterior Sampling/informed_insertion_distribution.jl")
# include("inter_network_models/Posterior Sampling/iex_SIM.jl")
# include("inter_network_models/Posterior Sampling/iex_SIM_split_merge.jl")
# include("inter_network_models/Posterior Sampling/iex_SIM_proportional.jl")
# include("inter_network_models/Posterior Sampling/iex_SIM_dependent.jl")
# include("inter_network_models/Posterior Sampling/iex_SIM_with_kick.jl")
# include("inter_network_models/Posterior Sampling/iex_SIS.jl")
# include("inter_network_models/Posterior Sampling/iex_SPF.jl")
# # Summaries
# include("inter_network_models/Posterior Sampling/Summaries/plot_recipes.jl")
# include("inter_network_models/Posterior Sampling/Summaries/misc_summaries.jl")
# include("inter_network_models/Posterior Sampling/Summaries/predictive_summaries.jl")


# iMCMC Moves (new structure)
include("inter_network_models/imcmc_moves/imcmc_move_type.jl")
include("inter_network_models/imcmc_moves/edit_allocation_move.jl")
include("inter_network_models/imcmc_moves/permutation_move.jl")
include("inter_network_models/imcmc_moves/insert_delete_move.jl")
include("inter_network_models/imcmc_moves/insert_delete_centered_move.jl")
include("inter_network_models/imcmc_moves/split_merge_move.jl")
include("inter_network_models/imcmc_moves/mixture_move.jl")
include("inter_network_models/samplers/outputs.jl")
include("inter_network_models/samplers/model_sampler.jl")
include("inter_network_models/samplers/posterior_sampler.jl")
include("inter_network_models/samplers/summaryplot.jl")
include("inter_network_models/samplers/legacy_constructors.jl")

include("inter_network_models/summaries/predictives/missing_entry.jl")
include("inter_network_models/summaries/predictives/posterior_predictive.jl")
include("inter_network_models/summaries/misc_summaries.jl")

# Graph Models
# =============
include("graph_models/utils.jl")
include("graph_models/CER/CER.jl")
include("graph_models/CER/CER_posterior.jl")

include("graph_models/SNF/SNF.jl")
include("graph_models/SNF/mcmc_moves/move_type.jl")
include("graph_models/SNF/mcmc_moves/gibbs_moves.jl")
# include("graph_models/SNF/SNF_samplers.jl")
# include("graph_models/SNF/SNF_multigraph.jl")
# include("graph_models/SNF/SNF_multigraph_gibbs.jl")
include("graph_models/SNF/samplers/SNF_model_sampler.jl")
include("graph_models/SNF/samplers/SNF_posterior_sampler.jl")
# include("graph_models/SNF/SNF_multigraph_gibbs_posterior.jl")

# Hollywood model
include("Types/hollywood_model.jl")

end
