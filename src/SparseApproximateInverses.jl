module SparseApproximateInverses

using LinearAlgebra
using SparseArrays
using LinearMaps: FunctionMap
using Random: seed!

export frob_inner_prod, frob_norm, frob_norm_squared, res_frob_norm
include("frob-inner-prod.jl")

export mr, pmr_spd, pmr_r, pmr_l, pmr_spd_spai, pmr_l_spai
include("matrix-iteration-mr.jl")

export lompr_spd, lopmr_r, lopmr_spd_spai, lompr_split_spai
include("matrix-iteration-lomr.jl")

export cg, pcg, pcg_spai
include("matrix-iteration-cg.jl")

export sd, psd_spd, psd_r, psd_l
include("matrix-iteration-sd.jl")

export ncg, npcg_spd
include("matrix-iteration-ncg.jl")

export pcg
include("vector-iteration-pcg.jl")

export self_precond_mr, self_precond_lomr
include("self-preconditioned-variants.jl")

include("dropping-strategies.jl")

end