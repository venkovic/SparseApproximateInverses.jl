push!(LOAD_PATH, ".")
using MySPAI: pcg

using Random: seed!
using LinearAlgebra
using MatrixMarket: mmread, mmwrite
using NPZ


matrix_source = "../matrix-market/"


matrix = "rand20k2"  # \in      {"bundle1", 
                          #      "4bw100eigs20k", 
                          #      "rand20k"
                          #      "rand20k2"
                          #      "wathen100", 
                          #      "Poisson32k"}


if matrix == "bundle1" # Computer graphics problem
  A = mmread(matrix_source * "bundle1.mtx")
  n = A.n # n = 10,581 | nnz = 770,811
elseif matrix == "4bw100eigs20k" # Custom matrix
  A = mmread(matrix_source * "4bw100eigs20k.mtx")
  n = A.n # n = 20,000 | nnz = 179,980
elseif matrix == "rand20k"
  A = mmread(matrix_source * "rand20k.mtx")
  n = A.n # n = 20,000 | nnz = 99,772
elseif matrix == "rand20k2"
  A = mmread(matrix_source * "rand20k2.mtx")
  n = A.n # n = 20,000 | nnz = 99,772
elseif matrix == "wathen100" # Random 2D/3D problem
  A = mmread(matrix_source * "wathen100.mtx")
  n = A.n # n = 30,401 | nnz = 471,601
elseif matrix == "Poisson32k" # Random Poisson PDE
  A = mmread(matrix_source * "Poisson_SExp_sig21.0_L0.1_DoF32000_K.mtx")
  n = A.n # n = 31,839 | nnz = 221,375 
end


tol = 1e-6
itmax = 2_000

seed!(1)
b = A * rand(n)

x, it, res_norm = pcg(A, b, zeros(n), I, tol, itmax)
npzwrite("data/Experiment06_" * matrix * "_cg-res.npz", res_norm)

M = mmread("data/Experiment05_" * matrix * "_M_pmr_spai.mtx")
x, it, res_norm_pmr = pcg(A, b, zeros(n), M, tol, itmax)
npzwrite("data/Experiment06_" * matrix * "_pcg-res_pmr_spai.npz", res_norm_pmr)

M = mmread("data/Experiment05_" * matrix * "_M_pcr_spai.mtx")
x, it, res_norm_pcr = pcg(A, b, zeros(n), M, tol, itmax)
npzwrite("data/Experiment06_" * matrix * "_pcg-res_pcr_spai.npz", res_norm_pcr)

M = mmread("data/Experiment05_" * matrix * "_M_lopmr_spai.mtx")
x, it, res_norm_lopmr = pcg(A, b, zeros(n), M, tol, itmax)
npzwrite("data/Experiment06_" * matrix * "_pcg-res_lopmr_spai.npz", res_norm_lopmr)