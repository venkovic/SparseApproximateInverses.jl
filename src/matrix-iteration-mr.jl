"""
mr(A, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Minimal residual method for approximate matrix inverses.

"""
function mr(A, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
  n = A.n
  R = spzeros(n, n)
  AR = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  AR .= A * R
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    R_AR_frob = frob_inner_prod(R, AR)
    AR_norm_square = frob_norm_squared(AR)
    alpha = R_AR_frob / AR_norm_square
    M .+= alpha .* R
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= alpha .* AR
    end
    AR .= A * R
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
    println("mr it = $i, err = $err, 
             nnz(M)/n^2 = $s")
    if (err < tol) || (s > smax)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
pmr_spd(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Minimal residual method for approximate matrix inverses with SPD preconditioner.

"""
function pmr_spd(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  AZ = spzeros(n, n)
  PrAZ = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  Z .= Pr * R
  AZ .= A * Z
  PrAZ .= Pr * AZ
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)    
    Z_AZ_frob = frob_inner_prod(Z, AZ)
    AZ_PrAZ_frob = frob_inner_prod(AZ, PrAZ)    
    alpha = Z_AZ_frob / AZ_PrAZ_frob
    M .+= alpha .* Z
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= alpha .* AZ
    end
    Z .= Pr * R
    AZ .= A * Z
    PrAZ .= Pr * AZ
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
    println("pmr it = $i, err = $err, 
             nnz(M)/n^2 = $s")
    if (err < tol) || (s > smax)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
pmr_r(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Right-preconditioned minimal residual method for approximate matrix inverses.

"""
function pmr_r(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  AZ = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  Z .= Pr * R
  AZ .= A * Z
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    R_AZ_frob = frob_inner_prod(R, AZ)
    AZ_norm_square = frob_norm_squared(AZ)
    alpha = R_AZ_frob / AZ_norm_square
    M .+= alpha .* Z
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= alpha .* AZ
    end
    Z .= Pr * R
    AZ .= A * Z
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
    println("pmr it = $i, err = $err, 
             nnz(M)/n^2 = $s")
    if (err < tol) || (s > smax)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
pmr_l(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Left-preconditioned minimal residual (MR) method for approximate matrix inverses.

"""
function pmr_l(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  AZ = spzeros(n, n)
  PrAZ = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  Z .= Pr * R
  AZ = A * Z
  PrAZ .= Pr * AZ
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    Z_PrAZ_frob = frob_inner_prod(Z, PrAZ)
    PrAZ_norm_square = frob_norm_squared(PrAZ)
    alpha = Z_PrAZ_frob / PrAZ_norm_square
    M .+= alpha .* Z
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= alpha * AZ
    end
    Z .= Pr * R
    AZ .= A * Z
    PrAZ .= Pr * AZ
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
    println("pmr it = $i, err = $err, 
             nnz(M)/n^2 = $s")
    if (err < tol) || (s > smax)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
pmr_spd_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res, eval_pcg=false, dropping=:hardthresholding)

Preconditioned minimal residual (MR) method for sparse approximate matrix inverses
with non-zero dropping and SPD preconditioner.

"""
function pmr_spd_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res, eval_pcg=false)
  n = A.n
  m = round(Int, s * n * n)
  R = spzeros(n, n)
  Z = spzeros(n, n)
  AZ = spzeros(n, n)
  PrAZ = spzeros(n, n)
  W = Z
  AR = AZ
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
  AZ .= A * Z
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    Z_AZ_frob = frob_inner_prod(Z, AZ)
    AZ_PrAZ_frob = frob_inner_prod(AZ, PrAZ)
    alpha = Z_AZ_frob / AZ_PrAZ_frob
    M .+= alpha .* Z
    if dropping == :hardthresholding
      apply_hardthreshold!(M, m)
      R .= I - A * M 
    elseif dropping == :heuristic
      apply_heuristic!(M, W, R, AR, A_cols_dot_prods, A, m)
    end  
    Z .= Pr * R
    AZ .= A * Z
    PrAZ = Pr * AZ
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
    println("pmr-spai it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2)")
    if (err < tol)
      break
    end
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
pmr_l_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res, eval_pcg=false)

Left-preconditioned minimal residual (MR) method for sparse approximate matrix inverses
with non-zero dropping.

"""
function pmr_l_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res, eval_pcg=false)
  n = A.n
  m = round(Int, s * n * n)
  R = spzeros(n, n)
  Z = spzeros(n, n)
  AZ = spzeros(n, n)
  W = Z
  AR = AZ
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
  AZ .= A * Z
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    R_AZ_frob = frob_inner_prod(R, AZ)
    AZ_norm_square = frob_norm_squared(AZ)
    alpha = R_AZ_frob / AZ_norm_square
    M .+= alpha .* Z
    if dropping == :hardthresholding
      apply_hardthreshold!(M, m)
      R .= I - A * M 
    elseif dropping == :heuristic
      apply_heuristic!(M, W, R, AR, A_cols_dot_prods, A, m)
    end 
    Z .= Pr * R
    AZ .= A * Z
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
    println("pmr-spai it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2)")
    if (err < tol)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  if eval_pcg
    return M, R_norm[1:i+1], pcg_iters[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end