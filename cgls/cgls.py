import numpy as np

def cgls(A, b, tol=1e-6, maxit=20, x0=np.array([]), Pl=np.array(1)):
  """
  intend to solve argmin_{x}Pl*||A*x - b||
  Toy, no sanity checks and exception-flag-returns written
  """
  if x0.size == 0: x0 = np.zeros([A.shape[1], b.shape[1]])

  H = lambda X: X.conj().T # Hermitian
  sqtol = np.power(tol, 2)  # sq the res norm tol to avoid sqrt
  sqnorm_p = lambda X: np.real(np.sum(X.conj()*X, axis=0)) # primal/dual sqnorm
  sqnorm_d = lambda X: np.real(np.sum(X.conj()*(Pl.dot(X)), axis=0))
  eps = np.spacing(1)

  bp_sqnorm = sqnorm_p(H(A).dot(Pl.dot(b)))
  bp_sqnorm_sqtol = sqtol*bp_sqnorm

  x  = x0
  rd = b - A.dot(x)         # dual residual (w/o Pl)
  rp = H(A).dot(Pl.dot(rd)) # primal residual
  pp = rp
  rp_sqnorm_old = sqnorm_p(rp)

  k, flag, exceptFlag = 0, False, 0
  while k < maxit and not flag:
    k += 1
    pd = A.dot(pp)
    alpha = rp_sqnorm_old/(sqnorm_d(pd) + eps)
    
    x  = x + alpha*pp
    rd = rd - alpha*pd # w/o Pl
    rp = H(A).dot(Pl.dot(rd))

    rp_sqnorm = sqnorm_p(rp)
    beta = rp_sqnorm/rp_sqnorm_old
    rp_sqnorm_old = rp_sqnorm

    pp = rp + beta*pp

    if rp_sqnorm <= bp_sqnorm_sqtol: flag = True
   
  return x, rp_sqnorm/bp_sqnorm, exceptFlag

def test_main():
  print('%% testing cgls %%')
  A = np.random.rand(5,3)
  b = np.random.rand(5,1)

  x_star = np.linalg.lstsq(A, b)[0]
  x, ratio, exceptFlag = cgls(A, b)

  e = x_star - x
  print('results:')
  print('- error:', e)
  print('- ratio:', ratio)
  print('- eFlag:', exceptFlag)
  print('%% testend cgls %%')

if __name__ == '__main__':
  test_main()
