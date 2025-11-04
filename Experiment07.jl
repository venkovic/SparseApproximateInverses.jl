push!(LOAD_PATH, ".")
using MySPAI: pmr_spai, pcg_spai, lopmr_spai

using Random: seed!
using SparseArrays
using LinearAlgebra
using Arpack
using MatrixMarket: mmread, mmwrite
using NPZ


matrix_source = "../matrix-market/"


matrix = "Poisson32k"# \in {"bundle1", 
                   #      "4bw100eigs20k", 
                   #      "4bw100eigs20k2", 
                   #      "rand20k",
                   #      "rand20k2",
                   #      "wathen100", 
                   #      "Poisson32k"}


if matrix == "bundle1" # Computer graphics problem
  A = mmread(matrix_source * "bundle1.mtx")
  n = A.n # n = 10,581 | nnz = 770,811
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "4bw100eigs20k" # Custom matrix
  A = mmread(matrix_source * "4bw100eigs20k.mtx")
  n = A.n # n = 20,000 | nnz = 179,980
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "4bw100eigs20k2" # Custom matrix
  A = mmread(matrix_source * "4bw100eigs20k2.mtx")
  n = A.n # n = 20,000 | nnz = 179,980
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "rand20k"
  A = mmread(matrix_source * "rand20k.mtx")
  n = A.n # n = 20,000 | nnz = 99,772
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "rand20k2"
  A = mmread(matrix_source * "rand20k2.mtx")
  n = A.n # n = 20,000 | nnz = 99,772
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "wathen100" # Random 2D/3D problem
  A = mmread(matrix_source * "wathen100.mtx")
  n = A.n # n = 30,401 | nnz = 471,601
  # pmr and pcr make no progress after 100 iterations
  itmax = Dict("pmr"=>100, "pcr"=>100, "lopmr"=>100)
elseif matrix == "Poisson32k" # Random Poisson PDE
  A = mmread(matrix_source * "Poisson32k.mtx")
  n = A.n # n = 31,839 | nnz = 221,375 
  # pmr and pcr make no progress after 100 iterations
  itmax = Dict("pmr"=>100, "pcr"=>100, "lopmr"=>100)
end

tol = 0
s = .03

Pr = spdiagm(diag(A).^-1)
M0 = spdiagm(ones(n))


dt = @elapsed _, R_norm, pcg_iters = pcg_spai(A, Pr, copy(M0), itmax["pcr"], tol, s, eval_pcg=true)
npzwrite("data/Experiment07_" * matrix * "_R_norm_pcg_spai.npz", R_norm)
npzwrite("data/Experiment07_" * matrix * "_pcg_iters_pcg_spai.npz", pcg_iters)

dt = @elapsed _, R_norm, pcg_iters = lopmr_spai(A, Pr, copy(M0), itmax["lopmr"], tol, s, eval_pcg=true)
npzwrite("data/Experiment07_" * matrix * "_R_norm_lopmr_spai.npz", R_norm)
npzwrite("data/Experiment07_" * matrix * "_pcg_iters_lopmr_spai.npz", pcg_iters)


dt = @elapsed _, R_norm, pcg_iters = pmr_spai(A, Pr, copy(M0), itmax["pmr"], tol, s, eval_pcg=true)
npzwrite("data/Experiment07_" * matrix * "_R_norm_pmr_spai.npz", R_norm)
npzwrite("data/Experiment07_" * matrix * "_pcg_iters_pmr_spai.npz", pcg_iters)
