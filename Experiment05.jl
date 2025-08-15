push!(LOAD_PATH, ".")
using MySPAI: pmr_spai, pcr_spai, lopmr_spai

using Random: seed!
using SparseArrays
using LinearAlgebra
using Arpack
using MatrixMarket: mmread, mmwrite
using NPZ


matrix_source = "../matrix-market/"


matrix = "rand20k2"# \in {"bundle1", 
                   #      "4bw100eigs20k", 
                   #      "rand20k",
                   #      "rand20k2",
                   #      "wathen100", 
                   #      "Poisson32k"}


if matrix == "bundle1" # Computer graphics problem
  A = mmread(matrix_source * "bundle1.mtx")
  n = A.n # n = 10,581 | nnz = 770,811
  tol = 10.
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "4bw100eigs20k" # Custom matrix
  A = mmread(matrix_source * "4bw100eigs20k.mtx")
  n = A.n # n = 20,000 | nnz = 179,980
  tol = 6.
  # pmr makes no progress after 5 iterations
  # pcr varies widely throughout iterations due to very small eigval of A, 
  # pcr's SPAI only become SPD after ~ 200 iterations
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "rand20k"
  A = mmread(matrix_source * "rand20k.mtx")
  n = A.n # n = 20,000 | nnz = 99,772
  tol = 10.
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "rand20k2"
  A = mmread(matrix_source * "rand20k2.mtx")
  n = A.n # n = 20,000 | nnz = 99,772
  tol = 10.
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "wathen100" # Random 2D/3D problem
  A = mmread(matrix_source * "wathen100.mtx")
  n = A.n # n = 30,401 | nnz = 471,601
  tol = 30.
  # pmr and pcr make no progress after 100 iterations
  itmax = Dict("pmr"=>100, "pcr"=>100, "lopmr"=>100)
elseif matrix == "Poisson32k" # Random Poisson PDE
  A = mmread(matrix_source * "Poisson_SExp_sig21.0_L0.1_DoF32000_K.mtx")
  n = A.n # n = 31,839 | nnz = 221,375 
  tol = 10.
  # pmr and pcr make no progress after 100 iterations
  itmax = Dict("pmr"=>100, "pcr"=>100, "lopmr"=>100)
end


s = .03

Pr = spdiagm(diag(A).^-1)
M0 = spdiagm(ones(n))


dt = @elapsed M, R_norm = pmr_spai(A, Pr, copy(M0), itmax["pmr"], tol, s)
npzwrite("data/Experiment05_" * matrix * "_R_norm_pmr_spai.npz", R_norm)
mmwrite("data/Experiment05_" * matrix * "_M_pmr_spai.mtx", M)
val_LR = real(eigs(M, nev=1, which=:LR, tol=1e-3, maxiter=2_000)[1][1]); println("val_LR = $val_LR")
val_SR = try
  real(eigs(M, nev=1, which=:SR, tol=1e-3, maxiter=3_000)[1][1])
catch
  real(eigs(M, nev=1, sigma=-100.0, which=:LM, tol=1e-3, maxiter=3000)[1][1])
end
println("val_SR = $val_SR")
npzwrite("data/Experiment05_" * matrix * "_metadata_pmr_spai.npz", 
         [val_LR, val_SR])

dt = @elapsed M, R_norm = pcr_spai(A, Pr, copy(M0), itmax["pcr"], tol, s)
npzwrite("data/Experiment05_" * matrix * "_R_norm_pcr_spai.npz", R_norm)
mmwrite("data/Experiment05_" * matrix * "_M_pcr_spai.mtx", M)
val_LR = real(eigs(M, nev=1, which=:LR, tol=1e-3, maxiter=2_000)[1][1]); println("val_LR = $val_LR")
val_SR = try
  real(eigs(M, nev=1, which=:SR, tol=1e-3, maxiter=3_000)[1][1])
catch
  real(eigs(M, nev=1, sigma=-100.0, which=:LM, tol=1e-3, maxiter=3000)[1][1])
end
println("val_SR = $val_SR")
npzwrite("data/Experiment05_" * matrix * "_metadata_pcr_spai.npz", 
         [val_LR, val_SR])

dt = @elapsed M, R_norm = lopmr_spai(A, Pr, copy(M0), itmax["lopmr"], tol, s)
npzwrite("data/Experiment05_" * matrix * "_R_norm_lopmr_spai.npz", R_norm)
mmwrite("data/Experiment05_" * matrix * "_M_lopmr_spai.mtx", M)
val_LR = real(eigs(M, nev=1, which=:LR, tol=1e-3, maxiter=2_000)[1][1]); println("val_LR = $val_LR")
val_SR = try
  real(eigs(M, nev=1, which=:SR, tol=1e-3, maxiter=3_000)[1][1])
catch
  real(eigs(M, nev=1, sigma=-100.0, which=:LM, tol=1e-3, maxiter=3000)[1][1])
end
println("val_SR = $val_SR")
npzwrite("data/Experiment05_" * matrix * "_metadata_lopmr_spai.npz", 
         [val_LR, val_SR])