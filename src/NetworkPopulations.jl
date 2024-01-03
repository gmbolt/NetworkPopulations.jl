module NetworkPopulations

# Alises for referring to Paths/Interaction Sequences/Samples of Interaction Sequences (purely for readability)
include("aliases.jl")

# Types
# include("Types/PathDistributions.jl")
# include("Types/MarkovModels.jl")
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


# Interaction network models
# =========================

# Model types and utils
# ---------------------
include("inter_network_models/SPF.jl")
include("inter_network_models/SIS.jl")
include("inter_network_models/SIM.jl")
include("inter_network_models/utils.jl")
include("inter_network_models/initalisers.jl")



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
include("graph_models/SNF/samplers/SNF_model_sampler.jl")
include("graph_models/SNF/samplers/SNF_posterior_sampler.jl")

# Hollywood model
# ===============
include("Types/hollywood_model.jl")

end
