import numpy  as np
import casadi as ca
import cvxpy  as cp



def _objective_P(P, x, x_dot, w = 0.0001):
    """ Eq(7) and Eq(8) in https://www.sciencedirect.com/science/article/abs/pii/S0921889014000372"""
    M, N = x.shape

    dv_dx = x @ P 
    dx_dt = x_dot 

    J_total = 0
    for i in range(M):
        dv_dt = ca.dot(dv_dx[i, :], dx_dt[i, :].reshape(1, -1))
        norm_dv_dx = ca.norm_2(dv_dx[i, :])
        norm_dx_dt = np.linalg.norm(dx_dt[i, :])

        psi = ca.if_else(ca.logic_or(norm_dv_dx==0, norm_dx_dt==0), 0, dv_dt/(norm_dv_dx*norm_dx_dt))  # Eq(8)
        J_total += ca.if_else(dv_dt<0, -w*psi**2, psi**2)  # Eq(7)

    return J_total


def _initial_guess(x):
    cov = np.cov(x.T)
    U, S, VT = np.linalg.svd(cov)
    S = S * 100  #expand the eigen value
    cov = U @ np.diag(S) @ VT
    return cov.flatten()



class dsopt_class():
    def __init__(self, x, x_dot, x_att, gamma):
        """
        Parameters:
        ----------

        x:  (M, N) NumPy array of position input, assuming no shift (not ending at origin)

        x_dot: (M, N) NumPy array of position output (velocity)

        x_att: (1, N) NumPy array of attractor

        gamma: (K, M) NumPy array of the mixing function, gamma, over the input data
        """

        # store parameters
        self.x  = x
        self.x_dot = x_dot
        self.x_att = x_att
        self.gamma = gamma

        self.x_sh = x - x_att  # shifted position
        self.M, self.N = x.shape
        self.K = gamma.shape[0]



    def begin(self):
        self._optimize_P()
        self._optimize_A()

        return self.A

    def _optimize_P(self):
        # Define parameters and variables 
        N = self.N
        num_constr = int(N*(N-1)/2) + N + 1
        P = ca.SX.sym('p', N, N)
        g = ca.SX(num_constr, 1)


        # Define constraints
        for i in range(N):
            for j in range(i + 1, N):  
                g[i+j-1] = (P[j, i] - P[i, j])   # Symmetry constraints

        eigen_value = ca.eig_symbolic(P)
        for i in range(N):
            g[N*(N-1)/2+i] = eigen_value[i]      # Positive definiteness constraints

        g[-1] = 1 - ca.sum1(eigen_value)         # Eigenvalue norm constrainst 


        # Define constraint bounds
        lbg=[0.0]*num_constr
        ubg=[0.0]*(num_constr-N-1) + [ca.inf]*N + [0.0]


        # Solve nlp
        nlp = {'x': ca.vec(P), 'f': _objective_P(P, self.x_sh, self.x_dot), 'g':g}
        S = ca.nlpsol('S', 'ipopt', nlp)
        # result = S(x0=[0.1]*N**2, lbg=lbg, ubg=ubg)
        result = S(x0=_initial_guess(self.x_sh), lbg=lbg, ubg=ubg)
        print(result['x'])
        self.P = np.array(result['x']).reshape(N, N)



    def _optimize_A(self):
        M = self.M
        N = self.N
        K = self.K
        P = self.P
        gamma = self.gamma

        # Define variables and constraints
        A_vars = []
        Q_vars = []
        constraints = []

        for k in range(K):
            A_vars.append(cp.Variable((N, N)))
            Q_vars.append(cp.Variable((N, N), symmetric=True))

            epi = 0.001
            epi = epi * -np.eye(N)

            constraints += [A_vars[k].T @ P + P @ A_vars[k] == Q_vars[k]]
            constraints += [Q_vars[k] << epi]


        for k in range(K):
            x_dot_pred_k = A_vars[k] @ self.x_sh.T
            if k == 0:
                x_dot_pred = cp.multiply(np.tile(gamma[k, :], (N, 1)), x_dot_pred_k)
            else:
                x_dot_pred += cp.multiply(np.tile(gamma[k, :], (N, 1)), x_dot_pred_k)


        Objective = cp.norm(x_dot_pred.T-self.x_dot, 'fro')

        prob = cp.Problem(cp.Minimize(Objective), constraints)
        prob.solve(solver=cp.MOSEK, verbose=True)


        A_res = np.zeros((K, N, N))
        for k in range(K):
            A_res[k, :, :] = A_vars[k].value
            print(A_vars[k].value)
        
        self.A = A_res

