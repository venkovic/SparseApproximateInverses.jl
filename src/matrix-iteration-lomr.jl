"""
lopmr_spd(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Locally optimal preconditioned minimal residual method for symmetric
positive definite (SPD) approximate matrix inverses with SPD preconditioner.

"""
function lopmr_spd(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  P = spzeros(n, n)
  AZ = spzeros(n, n)
  AP = spzeros(n, n)
  PrAP = spzeros(n, n)
  PrAZ = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  Z .= Pr * R  
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    AP .= A * P
    AZ .= A * Z
    PrAP .= Pr * AP
    PrAZ .= Pr * AZ
    if i == 0
      AZ_PrAZ_frob = frob_inner_prod(AZ, PrAZ)
      Z_AZ_frob = frob_inner_prod(Z, AZ)
      delta = Z_AZ_frob / AZ_PrAZ_frob
      gamma = 1.
    else
      AZ_PrAZ_frob = frob_inner_prod(AZ, PrAZ)
      AP_PrAP_frob = frob_inner_prod(AP, PrAP)
      AZ_PrAP_frob = frob_inner_prod(AZ, PrAP)
      Z_AZ_frob = frob_inner_prod(Z, AZ)
      Z_AP_frob = frob_inner_prod(Z, AP)
      c = AZ_PrAZ_frob * AP_PrAP_frob - AZ_PrAP_frob ^ 2
      delta = (AP_PrAP_frob * Z_AZ_frob - AZ_PrAP_frob * Z_AP_frob) / c
      gamma = (AZ_PrAZ_frob * Z_AP_frob - AZ_PrAP_frob * Z_AZ_frob) / c
    end
    P .*= gamma
    P .+= delta * Z
    M .+= P
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= delta .* AZ
      R .-= gamma .* AP
    end
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
    println("lopmr it = $i, err = $err, 
             nnz(M)/n^2 = $s,
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
    Z .= Pr * R
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
lopmr_spd_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res, eval_pcg=false, dropping=:hardthresholding)

Locally optimal preconditioned minimal residual method for sparse SPD approximate matrix 
inverses with SPD preconditioner.

"""
function lopmr_spd_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res, eval_pcg=false, dropping=:hardthresholding)
  n = A.n
  m = round(Int, s * n * n)
  R = spzeros(n, n)
  Z = spzeros(n, n)
  P = spzeros(n, n)
  AZ = spzeros(n, n)
  AP = spzeros(n, n)
  PrAP = spzeros(n, n)
  PrAZ = spzeros(n, n)
  AR = AZ
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
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    AP .= A * P
    AZ .= A * Z
    PrAP .= Pr * AP
    PrAZ .= Pr * AZ
    if i == 0
      AZ_PrAZ_frob = frob_inner_prod(AZ, PrAZ)
      Z_AZ_frob = frob_inner_prod(Z, AZ)
      delta = Z_AZ_frob / AZ_PrAZ_frob
      gamma = 1.
    else
      AZ_PrAZ_frob = frob_inner_prod(AZ, PrAZ)
      AP_PrAP_frob = frob_inner_prod(AP, PrAP)
      AZ_PrAP_frob = frob_inner_prod(AZ, PrAP)
      Z_AZ_frob = frob_inner_prod(Z, AZ)
      Z_AP_frob = frob_inner_prod(Z, AP)
      c = AZ_PrAZ_frob * AP_PrAP_frob - AZ_PrAP_frob ^ 2
      delta = (AP_PrAP_frob * Z_AZ_frob - AZ_PrAP_frob * Z_AP_frob) / c
      gamma = (AZ_PrAZ_frob * Z_AP_frob - AZ_PrAP_frob * Z_AZ_frob) / c
    end
    P .*= gamma
    P .+= delta * Z
    M .+= P
    if dropping == :hardthresholding
      apply_hardthreshold!(M, m)
      apply_hardthreshold!(P, m)
      R .= I - A * M 
    elseif dropping == :heuristic
      apply_heuristic!(M, W, R, AR, A_cols_dot_prods, A, m)
      apply_hardthreshold!(P, m)
    end  
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
    println("lopmr-spai it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2), 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
      break
    end
    Z .= Pr * R
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  if eval_pcg
    return M, R_norm[1:i+1], pcg_iters[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
lopmr_r(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Locally optimal right-preconditioned minimal residual method for approximate matrix inverses.

"""
function lopmr_r(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=true)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  Q = spzeros(n, n)
  AZ = spzeros(n, n)
  AQ = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  Z .= Pr * R  
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    AQ .= A * Q
    AZ .= A * Z
    if i == 0
      R_AZ_frob = frob_inner_prod(R, AZ)
      AZ_norm_square = frob_norm_squared(AZ)
      delta = R_AZ_frob / AZ_norm_square
      gamma = 1.
    else
      AQ_norm_square = frob_norm_squared(AQ)
      AZ_norm_square = frob_norm_squared(AZ)
      R_AZ_frob = frob_inner_prod(R, AZ)
      R_AQ_frob = frob_inner_prod(R, AQ)
      AZ_AQ_frob = frob_inner_prod(AZ, AQ)
      c = AZ_norm_square * AQ_norm_square - AZ_AQ_frob ^ 2
      delta = (AQ_norm_square * R_AZ_frob - conj(AZ_AQ_frob) * R_AQ_frob) / c
      gamma = (AZ_norm_square * R_AQ_frob - AZ_AQ_frob * R_AZ_frob) / c
    end
    Q .*= gamma 
    Q .+= delta * Z
    M .+= Q
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= delta .* AZ
      R .-= gamma .* AQ
    end
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
    println("lopmr it = $i, err = $err, 
             nnz(M)/n^2 = $s,
             nnz(Q)/n^2 = $(nnz(Q)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
    Z .= Pr * R
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
lopmr_split_spai(A, L, M, itmax, tol, s; stopping_criterion=:res)

Locally optimal preconditioned minimal residual method for sparse SPD approximate matrix
inverses with SPD preconditioner.

"""
function lopmr_split_spai(A, L, M, itmax, tol, s; stopping_criterion=:res)
  n = A.n
  m = round(Int, s * n * n)
  R = spzeros(n, n)
  P = spzeros(n, n)
  AR = spzeros(n, n)
  AP = spzeros(n, n)
  W = AP
  A_cols_dot_prods = Diagonal(vec(sum(abs2, A, dims=1)))
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  i = 0
  if dropping == :hardthresholding
    apply_hardthreshold!(M, m)
    R .= I - A * M 
  elseif dropping == :heuristic
    apply_heuristic!(M, W, R, AR, A_cols_dot_prods, A, m)
  end 
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  R .= L'R
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    AP .= A * P
    AP .= L'AP
    AR .= L * R
    AR .= A * AR
    AR .= L'AR
    if i == 0
      AZ_PrAZ_frob = frob_norm_squared(AR)
      Z_AZ_frob = frob_inner_prod(R, AR)
      delta = Z_AZ_frob / AZ_PrAZ_frob
      gamma = 1.
    else
      AZ_PrAZ_frob = frob_norm_squared(AR)
      AP_PrAP_frob = frob_norm_squared(AP)
      AZ_PrAP_frob = frob_inner_prod(AR, AP)
      Z_AZ_frob = frob_inner_prod(R, AR)
      Z_AP_frob = frob_inner_prod(R, AP)
      c = AZ_PrAZ_frob * AP_PrAP_frob - AZ_PrAP_frob ^ 2
      delta = (AP_PrAP_frob * Z_AZ_frob - AZ_PrAP_frob * Z_AP_frob) / c
      gamma = (AZ_PrAZ_frob * Z_AP_frob - AZ_PrAP_frob * Z_AZ_frob) / c
    end
    R .= L * R
    M .+= delta .* R
    M .+= gamma .* P
    P .*= gamma / delta
    P .+= R
    if dropping == :hardthresholding
      apply_hardthreshold!(M, m)
      apply_hardthreshold!(P, m)
      R .= I - A * M 
    elseif dropping == :heuristic
      apply_heuristic!(M, W, R, AR, A_cols_dot_prods, A, m)
      apply_hardthreshold!(P, m)
    end  
    i += 1
    R_norm[i + 1] = frob_norm(R)
    densities[i + 1] = nnz(M) / n^2
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("lopmr-spai-split it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2), 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
      break
    end
    R .= L'R
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end