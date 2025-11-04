push!(LOAD_PATH, ".")
using MySPAI: mr, sd, ncg, cg, 
              pmr, psd, npcg, pcg, lopmr, lopcg

using Random: seed!
using SparseArrays
using LinearAlgebra
using Arpack
using MatrixMarket: mmread, mmwrite
using NPZ


matrix_source = "../matrix-market/"


matrix = "msc04515"  # \in {"bcsstk21", "tri100eigs4k", "msc04515"}

if matrix == "bcsstk21" # Structural problem
  A = mmread(matrix_source * "bcsstk21.mtx")
  n = A.n # n = 3,600 | nnz = 26,600
  itmax = 1_000
elseif matrix == "tri100eigs4k" # Custom matrix
  seed!(1)  
  n = 4_000; k = 100
  A = spdiagm(0=>vcat(.05 .+ rand(k), ones(n-k)), -1=>rand(n-1)); A = A * A'
  mmwrite(matrix_source * "tri100eigs4k.mtx", A)
  n = A.n # n = 4,000 | nnz = 11,998
  itmax = 300
elseif matrix == "msc04515" # Structural problem
  A = mmread(matrix_source * "msc04515.mtx")
  n = A.n # n = 4,515 | nnz = 97,707
  itmax = 5_000
end


tol = 1e-15
itmax = 100


Pr = spdiagm(diag(A).^-1)
M0 = spdiagm(ones(n))


npzwrite("data/Experiment02_" * matrix * "_spectrum.npz", eigvals(Matrix(A)))


dt_pmr = @elapsed M, R_norm, backward_error = pmr(A, Pr, copy(M0), itmax, tol, stopping_criterion=:backward_error)
npzwrite("data/Experiment02_" * matrix * "_backward_error_pmr.npz", backward_error)
npzwrite("data/Experiment02_" * matrix * "_R_norm_pmr.npz", R_norm)
mmwrite("data/Experiment02_" * matrix * "_M_pmr.mtx", M)
npzwrite("data/Experiment02_" * matrix * "_spectrum_pmr.npz", eigvals(Matrix((M+M')./2.)))

dt_pcg = @elapsed M, R_norm, backward_error = pcg(A, Pr, copy(M0), itmax, tol, stopping_criterion=:backward_error)
npzwrite("data/Experiment02_" * matrix * "_backward_error_pcg.npz", backward_error)
npzwrite("data/Experiment02_" * matrix * "_R_norm_pcg.npz", R_norm)
mmwrite("data/Experiment02_" * matrix * "_M_pcg.mtx", M)
npzwrite("data/Experiment02_" * matrix * "_spectrum_pcg.npz", eigvals(Matrix((M+M')./2.)))

dt_lopmr = @elapsed M, R_norm, backward_error = lopmr(A, Pr, copy(M0), itmax, tol, stopping_criterion=:backward_error)
npzwrite("data/Experiment02_" * matrix * "_backward_error_lopmr.npz", backward_error)
npzwrite("data/Experiment02_" * matrix * "_R_norm_lopmr.npz", R_norm)
mmwrite("data/Experiment02_" * matrix * "_M_lopmr.mtx", M)
npzwrite("data/Experiment02_" * matrix * "_spectrum_lopmr.npz", eigvals(Matrix((M+M')./2.)))