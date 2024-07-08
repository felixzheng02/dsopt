import json, os
import numpy  as np
import casadi as ca
import cvxpy  as cp



def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data



def write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)



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


def get_PCA_P(att, K, x_dot, assignment_arr):

    """
    for k in range(K):
        x_dot_mean_k = np.mean(x_dot[assignment_arr==k, :], axis=0)

        eigenvalues, eigenvectors = np.linalg.eigh(gmm_sigma[k])

        idxs = eigenvalues.argsort()

        principal_eigenvector = eigenvectors[:, idxs[-1]]

        cos_sim = np.dot(x_dot_mean_k, principal_eigenvector) / (np.linalg.norm(x_dot_mean_k) * np.linalg.norm(principal_eigenvector))

        print(cos_sim)

        if np.abs(cos_sim) > 0.85:
            filtered_K.append(k)
    """
    
    mean_vec = []
    for k in range(K):
        x_dot_k = x_dot[assignment_arr==k, :]
        print(x_dot_k)
        exit()
        x_dot_mean_k = np.mean(x_dot_k, axis=0)
        mean_vec.append(x_dot_mean_k)
    mean_vec = np.array(mean_vec)


    mean_vec = mean_vec - att[0]
    cov_matrix = np.cov(mean_vec.T, bias=True)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    principal_axes = eigenvectors

    theta = -np.pi/2
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

    #PPP = (R @ e_vec)  @ np.diag(e_val) @ (R @ e_vec).T
    PPP = (R @ eigenvectors) @ np.diag(eigenvalues)* 0.01 @ (R @ eigenvectors).T

    # import matplotlib.pyplot as plt
    # # Plotting
    # plt.figure(figsize=(8, 8))

    # # Plot the original vectors
    # for vector in mean_vec:
    #     plt.quiver(att[0,0], 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.5)

    # # Plot the principal axes
    # origin = np.zeros(2)
    # for i in range(len(principal_axes)):
    #     #plt.quiver(0, 0, principal_axes[0, i], principal_axes[1, i], angles='xy', scale_units='xy', scale=1, color='red', alpha=0.8)
    #     # Scale the principal axes by the corresponding eigenvalue for better visualization
    #     plt.quiver(0, 0, eigenvalues[i] * principal_axes[0, i], eigenvalues[i] * principal_axes[1, i], angles='xy', scale_units='xy', scale=1, color='green', alpha=0.5)

    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    # plt.axhline(0, color='gray', lw=0.5)
    # plt.axvline(0, color='gray', lw=0.5)
    # plt.grid()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.title("Vectors and Principal Axes")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.show()
    # exit()
    return PPP


def plot_P(opti_P, PCA_P, data, att):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches 

    def V(x, y, P):
        X = np.array([x, y])
        return X.T @ P @ X

    fig, ax = plt.subplots(1,2, figsize=(8,4))


    # Calculate the min and max values for x and y
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_margin = x_range * 0.2
    y_margin = y_range * 0.2
    ax[0].set_xlim(x_min - x_margin, x_max + x_margin)
    ax[0].set_ylim(y_min - y_margin, y_max + y_margin)
    ax[1].set_xlim(x_min - x_margin, x_max + x_margin)
    ax[1].set_ylim(y_min - y_margin, y_max + y_margin)

    ax[0].scatter(data[:, 0], data[:, 1], color='k',  label='original data')
    ax[1].scatter(data[:, 0], data[:, 1], color='k',  label='original data')


    e_val, e_vec = np.linalg.eigh(opti_P)
    order = e_val.argsort()[::-1]
    e_val, e_vec = e_val[order], e_vec[:, order]
    x, y = e_vec[:, 0]
    theta = np.degrees(np.arctan2(y, x))
    n_std = 1.7
    width, height = 2 * n_std * np.sqrt(e_val)
    #sample_pts = create_ellipsoid2d(mus[i], np.sqrt(e_val), e_vec)
    ellip = patches.Ellipse(xy=att, width=width, height=height, angle=theta, \
                            edgecolor="#D79B00", facecolor="#FFE6CC", lw=2.0, zorder=100, alpha=0.6)

    x_min, x_max = ax[0].get_xlim()
    y_min, y_max = ax[0].get_ylim()
    plot_sample = 50
    x_mesh,y_mesh = np.meshgrid(np.linspace(x_min,x_max,plot_sample),np.linspace(y_min,y_max,plot_sample))
    Z = np.array([[V(xi, yi, opti_P) for xi in np.linspace(x_min,x_max,plot_sample)] for yi in np.linspace(y_min,y_max,plot_sample)])
    contour = ax[0].contour(x_mesh, y_mesh, Z, levels=np.linspace(Z.min(), Z.max(), 20))
    
    ax[0].scatter(att[0], att[1], c="#D79B00", s=8, zorder=101)
    ax[0].add_patch(ellip)
    ax[0].set_title('Optimization')


    e_val, e_vec = np.linalg.eigh(PCA_P)
    order = e_val.argsort()[::-1]
    e_val, e_vec = e_val[order], e_vec[:, order]
    x, y = e_vec[:, 0]
    theta = np.degrees(np.arctan2(y, x))
    n_std = 1.7
    width, height = 2 * n_std * np.sqrt(e_val)
    #sample_pts = create_ellipsoid2d(mus[i], np.sqrt(e_val), e_vec)
    ellip = patches.Ellipse(xy=att, width=width, height=height, angle=theta, \
                            edgecolor="#D79B00", facecolor="#FFE6CC", lw=2.0, zorder=100, alpha=0.6)
    x_min, x_max = ax[1].get_xlim()
    y_min, y_max = ax[1].get_ylim()
    plot_sample = 50
    x_mesh,y_mesh = np.meshgrid(np.linspace(x_min,x_max,plot_sample),np.linspace(y_min,y_max,plot_sample))
    Z = np.array([[V(xi, yi, PCA_P) for xi in np.linspace(x_min,x_max,plot_sample)] for yi in np.linspace(y_min,y_max,plot_sample)])
    contour = ax[1].contour(x_mesh, y_mesh, Z, levels=np.linspace(Z.min(), Z.max(), 20))
    
    ax[1].scatter(att[0], att[1], c="#D79B00", s=8, zorder=101)
    ax[1].add_patch(ellip)
    ax[1].set_title('PCA')

    plt.show()



class dsopt_class():
    def __init__(self, x, x_dot, x_att, gamma, assignment_arr):
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

        self.assignment_arr = assignment_arr

    def begin(self):
        opti_P = self._optimize_P(PCA=False)
        PCA_P = self._optimize_P(PCA=True)
        plot_P(opti_P, PCA_P, self.x, self.x_att[0])
        exit()
        self._optimize_A()
        # self._logOut()
        
        return self.A



    def _optimize_P(self, PCA=True):
        if not PCA:
            # Define parameters and variables 
            N = self.N
            num_constr = int(N*(N-1)/2) + N + 1
            P = ca.SX.sym('p', N, N)
            g = ca.SX(num_constr, 1)


            # Define constraints
            k = 0
            for i in range(N):
                for j in range(i + 1, N):  
                    g[k] = (P[j, i] - P[i, j])   # Symmetry constraints
                    k += 1

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
            result = S(x0=_initial_guess(self.x_sh), lbg=lbg, ubg=ubg)
            print(result['x'])
            self.P = np.array(result['x']).reshape(N, N)

            return np.array(result['x']).reshape(N, N)

        else:
            print("##############")
            print("Using PCA")
            print("##############")
            P_new = get_PCA_P(self.x_att[0, :], self.K, self.x_dot, self.assignment_arr)

            self.P = P_new

            return P_new






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
        max_norm = 1
        for k in range(K):
            A_vars.append(cp.Variable((N, N)))
            Q_vars.append(cp.Variable((N, N), symmetric=True))

            epi = 0.001
            epi = epi * -np.eye(N)

            constraints += [A_vars[k].T @ P + P @ A_vars[k] == Q_vars[k]]
            constraints += [Q_vars[k] << epi]
            # constraints += [cp.norm(A_vars[k], 'fro') <= max_norm]


        for k in range(K):
            x_dot_pred_k = A_vars[k] @ self.x_sh.T
            if k == 0:
                x_dot_pred  = cp.multiply(np.tile(gamma[k, :], (N, 1)), x_dot_pred_k)
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



    def _logOut(self, js_path=[]):
        """
        If json file exists, overwrite; if not create a new one

        A: K,M,M
        """    

        if len(js_path) == 0:
            js_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'output.json')
        self.original_js = read_json(js_path)

        self.original_js['A'] = self.A.ravel().tolist()
        self.original_js['attractor']= self.x_att.ravel().tolist()
        self.original_js['att_all']= self.x_att.ravel().tolist()
        self.original_js["gripper_open"] = 0

        write_json(self.original_js, js_path)