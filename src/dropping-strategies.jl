 function apply_heuristic!(M, W, R, AR, D, A, m; Symmetrize=false, FirstDropByEps=false)
  if Symmetrize
    M .+= M'
    M ./= 2.
  end
  
  if FirstDropByEps
    droptol!(M, eps(Float64))

    d_nnz = length(diag(M).nzval)
    if d_nnz < M.n
      println("Warning: $(M.n - d_nnz) diagonal components were dropped in M.")
    end
  end

  R .= I - A * M

  if nnz(M) <= m
    return
  end

  AR .= A * R

  W .= M
  W *= D
  W .+= 2 .* (AR .* (M .!= 0))
  W .*= M

  max = maximum(W.nzval)

  W[diagind(W)] .= max

  if nnz(M) > 2 * m
    thresh = partialsort(W.nzval, m, rev=true)
  else
    thresh = partialsort(W.nzval, nnz(W) - m + 1)
  end

  M .*= (W .> prevfloat(thresh))

  R .= I - A * M
end

function apply_hardthreshold!(P, m, FirstDropByEps=false)
  if FirstDropByEps
    droptol!(P, eps(Float64))
  end

  if nnz(P) <= m
    return
  end

  thresh = partialsort(abs.(P.nzval), m, rev=true)
  droptol!(P, thresh)
end