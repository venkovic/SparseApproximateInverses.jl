"""
lomr(A, M, itmax, tol, s=.03)

Locally optimal minimal residual method for approximate inverses of a definte matrix A.

"""
function lomr(A, M, itmax, tol, s=.03)
  n = A.n
  m = round(Int, s * n * n)
  R = spzeros(n, n)
  P = spzeros(n, n)
  AR = spzeros(n, n)
  AP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  R .= I - A * M
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  while (i < itmax)
    AP .= A * P
    AR .= A * R
    if i == 0
      R_AR_frob = frob_inner_prod(R, AR)
      AR_norm_square = frob_norm_squared(AR)
      delta = R_AR_frob / AR_norm_square
      gamma = 1.
    else
      AR_norm_square = frob_norm_squared(AR)
      AP_norm_square = frob_norm_squared(AP)
      AR_AP_frob = frob_inner_prod(AR, AP)
      R_AR_frob = frob_inner_prod(R, AR)
      R_AP_frob = frob_inner_prod(R, AP)
      c = AR_norm_square * AP_norm_square - AR_AP_frob ^ 2
      delta = (AP_norm_square * R_AR_frob - AR_AP_frob * R_AP_frob) / c
      gamma = (AR_norm_square * R_AP_frob - AR_AP_frob * R_AR_frob) / c
    end
    P .*= gamma
    P .+= delta * R
    M .+= P
    apply_hardthreshold!(M, m)
    apply_hardthreshold!(P, m)
    R .= I - A * M 
    i += 1
    R_norm[i + 1] = frob_norm(R)
    densities[i + 1] = nnz(M) / n^2
    err = R_norm[i + 1]
    s = nnz(M) / n^2
    println("lomr it = $i, err = $err, 
             nnz(M)/n^2 = $s,
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
      break
    end
  end
  return M, R_norm[1:i+1], densities
end

"""
lopmr_spd(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Locally optimal preconditioned minimal residual method for approximate inverses of 
symmetric definite matrix A with a SPD preconditioner Pr.

"""
function lopmr_spd(A, Pr, M, itmax, tol, s=.03)
  n = A.n
  m = round(Int, s * n * n)
  R = spzeros(n, n)
  Z = spzeros(n, n)
  P = spzeros(n, n)
  AZ = spzeros(n, n)
  AP = spzeros(n, n)
  PrAP = spzeros(n, n)
  PrAZ = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  R .= I - A * M
  Z .= Pr * R  
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
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
    apply_hardthreshold!(M, m)
    apply_hardthreshold!(P, m)
    R .= I - A * M
    i += 1
    R_norm[i + 1] = frob_norm(R)
    densities[i + 1] = nnz(M) / n^2
    err = R_norm[i + 1]
    s = nnz(M) / n^2
    println("lopmr_spd it = $i, err = $err, 
             nnz(M)/n^2 = $s,
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
    Z .= Pr * R
  end
  return M, R_norm[1:i+1], densities
end

"""
lopmr_right_preconditioned(A, Pr, M, itmax, tol, smax=.03)

Locally optimal right-preconditioned minimal residual method for approximate inverses of 
definite matrix A.

"""
function lopmr_right_preconditioned(A, Pr, M, itmax, tol, s=.03)
  n = A.n
  m = round(Int, s * n * n)
  R = spzeros(n, n)
  Z = spzeros(n, n)
  Q = spzeros(n, n)
  AZ = spzeros(n, n)
  AQ = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  R .= I - A * M
  Z .= Pr * R  
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
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
    apply_hardthreshold!(M, m)
    apply_hardthreshold!(Q, m)
    i += 1
    R_norm[i + 1] = frob_norm(R)
    densities[i + 1] = nnz(M) / n^2
    err = R_norm[i + 1]
    s = nnz(M) / n^2
    println("lopmr_right_preconditioned it = $i, err = $err, 
             nnz(M)/n^2 = $s,
             nnz(Q)/n^2 = $(nnz(Q)/n^2)")
    if (err < tol)
      break
    end
    Z .= Pr * R
  end
  return M, R_norm[1:i+1], densities
end