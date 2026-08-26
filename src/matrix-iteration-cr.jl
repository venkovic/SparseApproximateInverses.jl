"""
pcr(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Preconditioned conjugate residual method for approximate symmetric matrix inverses with
symmetric positive definite (SPD) preconditioner.

"""
function pcr(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
  n = A.n
  R = spzeros(n, n)
  P = spzeros(n, n)
  Z = spzeros(n, n)
  AP = spzeros(n, n)
  AZ = spzeros(n, n)
  PrAP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  Z .= Pr * R
  P .= Z
  AP .= A * P
  AZ .= AP
  PrAP .= Pr * AP
  Z_AZ_frob = frob_inner_prod(Z, AZ)
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    AP_PrAP_frob = frob_inner_prod(AP, PrAP)
    alpha = Z_AZ_frob / AP_PrAP_frob
    beta = 1. / Z_AZ_frob
    M .+= alpha .* P
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= alpha .* AP
    end
    Z .= Pr * R
    i += 1
    R_norm[i + 1] = frob_norm(R)
    densities[i + 1] = nnz(M) / n^2
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    s = nnz(M) / n^2
    println("pcr it = $i, err = $err, 
             nnz(M)/n^2 = $s, 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
    AZ .= A * Z
    Z_AZ_frob = frob_inner_prod(Z, AZ)
    beta *= Z_AZ_frob
    P .*= beta
    P .+= Z
    AP .= A * P
    PrAP .= Pr * AP
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
pcr_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res, eval_pcg=false, dropping=:hardthresholding)

Preconditioned conjugate residual method for sparse approximate matrix 
inverses with SPD preconditioner.

"""
function pcr_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res, eval_pcg=false, dropping=:hardthresholding)
  n = A.n
  m = round(Int, s * n * n)
  R = spzeros(n, n)
  P = spzeros(n, n)
  Z = spzeros(n, n)
  AP = spzeros(n, n)
  AZ = spzeros(n, n)
  PrAP = spzeros(n, n)
  W = AP
  A_cols_dot_prods = Diagonal(vec(sum(abs2, A, dims=1)))
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  if eval_pcg
    seed!(1)
    pcg_tol = 1e-6
    b = A * rand(n)
    pcg_iters = zeros(Int, itmax + 1)
    _, pcg_iters[1], _ = pcg(A, b, zeros(n), M, 1e-6, n)
    println("pcg iters = $(pcg_iters[1])")
  end
  if dropping == :hardthresholding
    apply_hardthreshold!(M, m)
    R .= I - A * M 
  elseif dropping == :heuristic
    apply_heuristic!(M, W, R, AR, A_cols_dot_prods, A, m)
  end 
  Z .= Pr * R
  P .= Z
  apply_hardthreshold!(P, m)
  AP .= A * P
  AZ .= AP
  PrAP = Pr * AP
  Z_AZ_frob = frob_inner_prod(Z, AZ)
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    AP_PrAP_frob = frob_inner_prod(AP, PrAP)
    alpha = Z_AZ_frob / AP_PrAP_frob
    beta = 1. / Z_AZ_frob
    M .+= alpha .* P
    if dropping == :hardthresholding
      apply_hardthreshold!(M, m)
      R .= I - A * M 
    elseif dropping == :heuristic
      apply_heuristic!(M, W, R, AR, A_cols_dot_prods, A, m)
    end  
    Z .= Pr * R
    i += 1
    R_norm[i + 1] = frob_norm(R)
    densities[i + 1] = nnz(M) / n^2
    err = R_norm[i + 1]
    if eval_pcg
      _, pcg_iters[i+1], _ = pcg(A, b, zeros(n), M, pcg_tol, pcg_iters[1])
      println("pcg iters = $(pcg_iters[i+1])")
    end
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("pcr-spai it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2), 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
      break
    end
    beta *= Z_AZ_frob
    P .*= beta
    P .+= Z
    apply_hardthreshold!(P, m)
    AP .= A * P
    AZ .= A * Z
    PrAP = Pr * AP
    Z_AZ_frob = frob_inner_prod(Z, AZ)
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  if eval_pcg
    return M, R_norm[1:i+1], pcg_iters[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end