T = Float64

"""
     pcg(A::Union{SparseMatrixCSC{T},
                  FunctionMap},
         b::Array{T,1},
         x::Array{T,1},
         M)

Performs PCG (Saad, 2003).

Saad, Y.
Iterative methods for sparse linear systems
SIAM, 2003, 82.

"""
function pcg(A::Union{SparseMatrixCSC{T},
                      FunctionMap},
             b::Array{T,1},
             x::Array{T,1},
             M,
             tol,
             maxit)
  
  n, = size(x)
  r = Array{T,1}(undef, n)
  z = Array{T,1}(undef, n)
  p = Array{T,1}(undef, n)
  Ap = Array{T,1}(undef, n)
  backward_error = Array{T,1}(undef, n)

  maxit == 0 ? maxit = n : nothing
  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  z .= M * r
  rTz = dot(r, z)
  copyto!(p, z)
  bnorm = norm(b)
  backward_error[it] = sqrt(rTr) / bnorm

  while (it < maxit) && (backward_error[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTz / d
    beta = 1. / rTz
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    z .= M * r
    rTz = dot(r, z)
    beta *= rTz
    axpby!(1, z, beta, p) # p = beta * p + z
    it += 1
    backward_error[it] = sqrt(rTr) / bnorm
  end

  return x, it, backward_error[1:it]
end