function dropping_M!(M, W, R, AR, D, A, m)
  M .+= M'
  M ./= 2.
  droptol!(M, eps(Float64))

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

function dropping_P!(P, m)
  droptol!(P, eps(Float64))
  if nnz(P) <= m
    return
  end

  thresh = partialsort(abs.(P.nzval), m, rev=true)
  droptol!(P, thresh)
end