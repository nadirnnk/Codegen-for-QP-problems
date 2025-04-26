import osqp
import numpy as np
from scipy import sparse

m, n = 15, 6
X = np.random.randn(m, n)
y = np.random.randn(m)
lam = 0.3

P = sparse.block_diag([X.T@X, X.T@X], format='csc')
q = np.hstack([-X.T@y, X.T@y]) + lam*np.ones(2*n)
A = sparse.vstack([
    sparse.hstack([sparse.eye(n), -sparse.eye(n)]),
    sparse.hstack([-sparse.eye(n), -sparse.eye(n)]),
    sparse.eye(2*n)
], format='csc')
l = np.hstack([np.zeros(n), -np.inf*np.ones(n), np.zeros(2*n)])
u = np.hstack([np.inf*np.ones(n), np.zeros(n), np.inf*np.ones(2*n)])

prob = osqp.OSQP()
prob.setup(P, q, A, l, u)
res = prob.solve()
print("Beta:", res.x[:n] - res.x[n:])

from scipy.sparse import csc_matrix, triu
A_csc = csc_matrix(A)  # Convert to CSC
P_triu = triu(P)  # Extract only the upper triangular part
P_csc = csc_matrix(P_triu)  # Convert to CSC
q = np.array(q)  # Convert to dense array
l = np.array(l)  # Convert to dense array
u = np.array(u)  # Convert to dense array
np.savez("CSC", P_data=P_csc.data, P_indices=P_csc.indices, P_ind=P_csc.indptr, A_data=A_csc.data, A_indices=A_csc.indices, A_ind=A_csc.indptr, q=q, l=l, u=u)
np.savez("python_results.npz", x=res.x, obj=res.info.obj_val, status=res.info.status)