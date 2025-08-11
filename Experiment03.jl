push!(LOAD_PATH, ".")
using MySPAI: pmr, pcr, lopmr

using Random: seed!
using SparseArrays
using LinearAlgebra
using Arpack
using MatrixMarket: mmread, mmwrite
using NPZ


matrix_source = "../matrix-market/"


matrix = "rand20k2"  # \in {"bundle1", 
                     #      "4bw100eigs20k", 
                     #      "4bw100eigs20k2"
                     #      "rand20k",
                     #      "rand20k2",
                     #      "wathen100", 
                     #      "Poisson32k"}


if matrix == "bundle1" # Computer graphics problem
  A = mmread(matrix_source * "bundle1.mtx")
  n = A.n # n = 10,581 | nnz = 770,811
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "4bw100eigs20k" # Custom matrix
  seed!(1)  
  n = 20_000; k = 100
  # Banded matrix with bw of 4 and 100 distinct eigvals
  A = spdiagm(0=>vcat(.05*rand(k), ones(n-k)), 
             -1=>.01*rand(n-1), 
             -2=>.01*rand(n-2), 
             -3=>.01*rand(n-3), 
             -4=>.01*rand(n-4))
  A = A * A'
  mmwrite(matrix_source * "4bw100eigs20k.mtx", A)
  n = A.n # n = 20,000 | nnz = 179,980
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "4bw100eigs20k2" # Custom matrix
  seed!(1)  
  n = 20_000; k = 100
  # Banded matrix with bw of 4 and 100 distinct eigvals
  A = spdiagm(0=>vcat(1e5*rand(k), ones(n-k)), 
             -1=>.01*rand(n-1), 
             -2=>.01*rand(n-2), 
             -3=>.01*rand(n-3), 
             -4=>.01*rand(n-4))
  A = A * A'
  mmwrite(matrix_source * "4bw100eigs20k2.mtx", A)
  n = A.n # n = 20,000 | nnz = 179,980
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "rand20k"
  seed!(1)
  n = 20_000
  Is = rand(1:n, n)
  Js = [rand(1:i) for i in Is]
  Is = vcat(1:n, Is)
  Js = vcat(1:n, Js)
  nzvals = vcat(1e4*rand(n), rand(length(Is)-n))
  A = sparse(Is, Js, nzvals)
  A = A * A'
  mmwrite(matrix_source * "rand20k.mtx", A)
  n = A.n # n = 20,000 | nnz = 99,772
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "rand20k2"
  seed!(1)
  n = 20_000
  Is = rand(1:n, n)
  Js = [rand(1:i) for i in Is]
  Is = vcat(1:n, Is)
  Js = vcat(1:n, Js)
  nzvals = vcat(1e2*rand(n), rand(length(Is)-n))
  A = sparse(Is, Js, nzvals)
  A = A * A'
  mmwrite(matrix_source * "rand20k2.mtx", A)
  n = A.n # n = 20,000 | nnz = 99,772
  itmax = Dict("pmr"=>200, "pcr"=>200, "lopmr"=>200)
elseif matrix == "wathen100" # Random 2D/3D problem
  A = mmread(matrix_source * "wathen100.mtx")
  n = A.n # n = 30,401 | nnz = 471,601
  # pmr makes no progress
  itmax = Dict("pmr"=>100, "pcr"=>100, "lopmr"=>100)
elseif matrix == "Poisson32k" # Random Poisson PDE
  A = mmread(matrix_source * "Poisson_SExp_sig21.0_L0.1_DoF32000_K.mtx")
  n = A.n # n = 31,839 | nnz = 221,375 
  itmax = Dict("pmr"=>100, "pcr"=>100, "lopmr"=>100)
end


function check_diagonal_dominance(A)
  n = A.n
  return [abs(A[i, i]) / (sum(abs.(A[i, 1:n])) - abs(A[i, i])) for i in 1:n]
end


tol = 1.
smax = .03


Pr = spdiagm(diag(A).^-1)
M0 = spdiagm(ones(n))


dt = @elapsed M, R_norm = pmr(A, Pr, copy(M0), itmax["pmr"], tol, smax=smax)
npzwrite("data/Experiment03_" * matrix * "_R_norm_pmr.npz", R_norm)
mmwrite("data/Experiment03_" * matrix * "_M_pmr.mtx", M)
M .+= M'; M ./ 2.
val_LR = real(eigs(M, nev=1, which=:LR, tol=1e-3, maxiter=2_000)[1][1]); println("val_LR = $val_LR")
val_SR = try
  real(eigs(M, nev=1, which=:SR, tol=1e-3, maxiter=3_000)[1][1])
catch
  real(eigs(M, nev=1, sigma=-100.0, which=:LM, tol=1e-3, maxiter=3000)[1][1])
end
println("val_SR = $val_SR")
npzwrite("data/Experiment03_" * matrix * "_metadata_pmr.npz", 
         [val_LR, val_SR])

dt = @elapsed M, R_norm = pcr(A, Pr, copy(M0), itmax["pcr"], tol, smax=smax)
npzwrite("data/Experiment03_" * matrix * "_R_norm_pcr.npz", R_norm)
mmwrite("data/Experiment03_" * matrix * "_M_pcr.mtx", M)
M .+= M'; M ./ 2.
val_LR = real(eigs(M, nev=1, which=:LR, tol=1e-3, maxiter=2_000)[1][1]); println("val_LR = $val_LR")
val_SR = try
  real(eigs(M, nev=1, which=:SR, tol=1e-3, maxiter=3_000)[1][1])
catch
  real(eigs(M, nev=1, sigma=-100.0, which=:LM, tol=1e-3, maxiter=3000)[1][1])
end
println("val_SR = $val_SR")
npzwrite("data/Experiment03_" * matrix * "_metadata_pcr.npz", 
         [val_LR, val_SR])

dt = @elapsed M, R_norm = lopmr(A, Pr, copy(M0), itmax["lopmr"], tol, smax=smax)
npzwrite("data/Experiment03_" * matrix * "_R_norm_lopmr.npz", R_norm)
mmwrite("data/Experiment03_" * matrix * "_M_lopmr.mtx", M)
M .+= M'; M ./ 2.
val_LR = real(eigs(M, nev=1, which=:LR, tol=1e-3, maxiter=2_000)[1][1]); println("val_LR = $val_LR")
val_SR = try
  real(eigs(M, nev=1, which=:SR, tol=1e-3, maxiter=3_000)[1][1])
catch
  real(eigs(M, nev=1, sigma=-100.0, which=:LM, tol=1e-3, maxiter=3000)[1][1])
end
println("val_SR = $val_SR")
npzwrite("data/Experiment03_" * matrix * "_metadata_lopmr.npz", 
         [val_LR, val_SR])