push!(LOAD_PATH, ".")
using MySPAI: mr, sd, ncg, cg, 
              pmr, psd, lpcg, pcg, lopmr, lopcg

using Random: seed!
using SparseArrays
using LinearAlgebra
using Arpack
using MatrixMarket: mmread, mmwrite
using NPZ


matrix_source = "../matrix-market/"


matrix = "triunif4k"  # \in {"bcsstk21", 
                      #      "tri100eigs4k", 
                      #      "triclust4k",
                      #      "triunif4k",
                      #      "msc04515"}

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
elseif matrix == "triclust4k"
  seed!(1)
  n = 4_000
  vals = vcat(Vector(1e-9:1e-9:(n-1)*1e-9), 1.)
  Q, _ = qr(rand(n, n))
  A = Q * spdiagm(vals) * Q'
  F = hessenberg(A)
  A = spdiagm(0=>diag(F.H, 0), 1=>diag(F.H, 1), -1=>diag(F.H, -1))
  mmwrite(matrix_source * "triclust4k.mtx", A)
  n = A.n # n = 4,000 | nnz = 11,998
  itmax = 1_000
elseif matrix == "triunif4k"
  seed!(1)
  n = 4_000
  dt = (1. - 1e-9) / (n - 1)
  vals = Vector(1e-9:dt:1.)
  Q, _ = qr(rand(n, n))
  A = Q * spdiagm(vals) * Q'
  F = hessenberg(A)
  A = spdiagm(0=>diag(F.H, 0), 1=>diag(F.H, 1), -1=>diag(F.H, -1))
  mmwrite(matrix_source * "triunif4k.mtx", A)
  n = A.n # n = 4,000 | nnz = 11,998
  itmax = 1_000
elseif matrix == "msc04515" # Structural problem
  A = mmread(matrix_source * "msc04515.mtx")
  n = A.n # n = 4,515 | nnz = 97,707
  itmax = 5_000
end


tol = 1e-15


Pr = spdiagm(diag(A).^-1)
M0 = spdiagm(ones(n))


npzwrite("data/Experiment01_" * matrix * "_spectrum.npz", eigvals(Matrix(A)))

#dt_mr = @elapsed, R_norm = mr(A, copy(M0), itmax, tol)
#npzwrite("data/Experiment01_" * matrix * "_R_norm_mr.npz", R_norm)
#mmwrite("data/Experiment01_" * matrix * "_M_mr.mtx", M)
#npzwrite("data/Experiment01_" * matrix * "_spectrum_mr.npz", eigvals(Matrix((M+M')./2.)))

#dt_sd = @elapsed M, R_norm = mr(A, copy(M0), itmax, tol)
#npzwrite("data/Experiment01_" * matrix * "_R_norm_sd.npz", R_norm)
#mmwrite("data/Experiment01_" * matrix * "_M_sd.mtx", M)
#npzwrite("data/Experiment01_" * matrix * "_spectrum_sd.npz", eigvals(Matrix((M+M')./2.)))

#dt_ncg = @elapsed M, R_norm = ncg(A, copy(M0), itmax, tol)
#npzwrite("data/Experiment01_" * matrix * "_R_norm_ncg.npz", R_norm)
#mmwrite("data/Experiment01_" * matrix * "_M_ncg.mtx", M)
#npzwrite("data/Experiment01_" * matrix * "_spectrum_ncg.npz", eigvals(Matrix((M+M')./2.)))

#dt_cg = @elapsed M, R_norm = csd(A, copy(M0), itmax, tol)
#npzwrite("data/Experiment01_" * matrix * "_R_norm_cr.npz", R_norm)
#mmwrite("data/Experiment01_" * matrix * "_M_cr.mtx", M)
#npzwrite("data/Experiment01_" * matrix * "_spectrum_cr.npz", eigvals(Matrix((M+M')./2.)))

dt_pmr = @elapsed M, R_norm = pmr(A, Pr, copy(M0), itmax, tol)
npzwrite("data/Experiment01_" * matrix * "_R_norm_pmr.npz", R_norm)
mmwrite("data/Experiment01_" * matrix * "_M_pmr.mtx", M)
npzwrite("data/Experiment01_" * matrix * "_spectrum_pmr.npz", eigvals(Matrix((M+M')./2.)))

dt_psd = @elapsed M, R_norm = psd(A, Pr, copy(M0), itmax, tol)
npzwrite("data/Experiment01_" * matrix * "_R_norm_psd.npz", R_norm)
mmwrite("data/Experiment01_" * matrix * "_M_psd.mtx", M)
npzwrite("data/Experiment01_" * matrix * "_spectrum_psd.npz", eigvals(Matrix((M+M')./2.)))

dt_npcg = @elapsed M, R_norm = npcg(A, Pr, copy(M0), itmax, tol)
npzwrite("data/Experiment01_" * matrix * "_R_norm_npcg.npz", R_norm)
mmwrite("data/Experiment01_" * matrix * "_M_npcg.mtx", M)
npzwrite("data/Experiment01_" * matrix * "_spectrum_npcg.npz", eigvals(Matrix((M+M')./2.)))

dt_pcg = @elapsed M, R_norm = pcg(A, Pr, copy(M0), itmax, tol)
npzwrite("data/Experiment01_" * matrix * "_R_norm_pcg.npz", R_norm)
mmwrite("data/Experiment01_" * matrix * "_M_pcg.mtx", M)
npzwrite("data/Experiment01_" * matrix * "_spectrum_pcg.npz", eigvals(Matrix((M+M')./2.)))

dt_lopmr = @elapsed M, R_norm = lopmr(A, Pr, copy(M0), itmax, tol)
npzwrite("data/Experiment01_" * matrix * "_R_norm_lopmr.npz", R_norm)
mmwrite("data/Experiment01_" * matrix * "_M_lopmr.mtx", M)
npzwrite("data/Experiment01_" * matrix * "_spectrum_lopmr.npz", eigvals(Matrix((M+M')./2.)))

# LOPCG is equivalent to PCG, just more expensive to compute.
#dt_lopcg = @elapsed M, R_norm = lopcg(A, Pr, copy(M0), itmax, tol)
#npzwrite("data/Experiment01_" * matrix * "_R_norm_lopcg.npz", R_norm)
#mmwrite("data/Experiment01_" * matrix * "_M_lopcg.mtx", M)
#npzwrite("data/Experiment01_" * matrix * "_spectrum_lopcg.npz", eigvals(Matrix((M+M')./2.)))