import numpy as np
from scipy.linalg import eig


# Function to calculate cyclic stiffness matrix (K_cyc_list)
def calc_K_cyc_list(K_const, K_var, k_amplitude, d_cyc_gms_coarse):
    """
    Calculate the cyclic stiffness matrix K_cyc_list.

    Parameters:
        K_const (ndarray): Constant stiffness matrix.
        K_var (ndarray): Variable stiffness matrices (3D array).
        k_amplitude (float): Amplitude scaling factor for variable stiffness.
        d_cyc_gms_coarse (float): Step size for the cyclic vector.

    Returns:
        tuple: K_cyc_list (3D array of stiffness matrices) and cyc_vctr (cyclic vector).
    """
    # Add constant and variable stiffness contributions
    K_cyc_list = K_const[:, :, np.newaxis] + K_var * k_amplitude
    # Create cyclic vector
    cyc_vctr = np.arange(K_cyc_list.shape[2]) * d_cyc_gms_coarse
    return K_cyc_list, cyc_vctr


# Function to calculate the damping matrix (C)
def calc_damping_matrix(K_cyc_list, M_mat, zeta=0.05):
    """
    Calculate the damping matrix (C) using modal damping.

    Parameters:
        K_cyc_list (ndarray): Cyclic stiffness matrix (3D array).
        M_mat (ndarray): Mass matrix.
        zeta (float or ndarray): Damping ratio (default is 0.05).

    Returns:
        ndarray: Damping matrix (C).
    """
    # Calculate the average stiffness matrix
    avgK = np.sum(K_cyc_list, axis=2) / K_cyc_list.shape[2]

    # Solve the eigenvalue problem
    spectral_mtx, modal_mtx = eig(avgK, M_mat)  # Generalized eigenvalue problem
    omega = np.sqrt(np.real(spectral_mtx))  # Natural frequencies (take real parts)

    # Initialize damping matrix
    ndof = len(omega)
    zeta = np.full((ndof, 1), zeta) if np.isscalar(zeta) else zeta  # Ensure zeta is an array
    damping_matrix = np.zeros_like(M_mat)

    # Calculate modal contributions to the damping matrix
    for n in range(ndof):
        modal_mass = np.dot(modal_mtx[:, n].T, np.dot(M_mat, modal_mtx[:, n]))  # Modal mass
        modal_damping = 2 * zeta[n] * omega[n] / modal_mass  # Modal damping
        damping_matrix += (
            M_mat @ (modal_damping * np.outer(modal_mtx[:, n], modal_mtx[:, n])) @ M_mat
        )

    return damping_matrix

# Function to calculate the external force vector (FexVctrT)
def calc_FexVctrT(cyc_motor, shaft_polar_stiff, load, ndof):
    """
    Calculate the external force vector in time (FexVctrT).

    Parameters:
        cyc_motor (ndarray): Cyclic motor torque vector.
        shaft_polar_stiff (float): Shaft polar stiffness.
        load (float): External load.
        ndof (int): Number of degrees of freedom.

    Returns:
        ndarray: External force vector (FexVctrT).
    """
    # Motor torque contribution
    torque_motor = shaft_polar_stiff * cyc_motor
    # Constant load torque
    torque_t = load * np.ones_like(torque_motor)

    # Initialize external force vector
    FexVctrT = np.zeros((ndof, len(cyc_motor)))
    FexVctrT[1, :] += torque_motor.T  # Motor torque applied to DOF 2
    FexVctrT[2, :] += torque_t.T  # Load torque applied to DOF 3

    return FexVctrT