inner_thresh = 1e-8

function self_precond_mr(A, M, itmax, ni, tol)
  n = A.n
  M2 = copy(M)
  R = I - A * M
  x = Vector{Float64}(undef, n)
  r = Vector{Float64}(undef, n)
  z = Vector{Float64}(undef, n)
  Az = Vector{Float64}(undef, n)
  R_norm = zeros(itmax + 1)
  i = 0
  R_norm[i + 1] = frob_norm(R)
  while (i < itmax)
    #Threads.@threads for j in 1:n
    for j in 1:n
      ej = sparsevec([j], [1.], n)
      s = M[:, j]
      r .= ej
      r .-= A * s 
      z .= M * r
      ii = 1
      while (norm(r) > inner_thresh) & (ii <= ni)
        Az .= A * z
        alpha = r'Az / Az'Az
        s .+= alpha * z
        r .-= alpha * Az
        z .= M * r
        ii += 1
      end
      M2[:, j] = s
      ii += 1
    end
    M .= M2
    R = I - A * M
    i += 1
    R_norm[i + 1] = frob_norm(R)
    err = R_norm[i + 1]
    s_nnz = nnz(M) / n^2
    println("self_precond_mr it = $i, err = $err, 
             nnz(M)/n^2 = $s_nnz")
    if (err < tol)
      break
    end
  end
  return M, R_norm[1:i+1]
end