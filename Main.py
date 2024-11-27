from scipy.io import loadmat
from Simulation_Parameter import get_parameters
from calc_euler_lagrange_components import calc_K_cyc_list, calc_damping_matrix, calc_FexVctrT

# Define constants
LOAD = 10  # External load
RPS = 40  # Rotations per second
K_AMPLITUDE = 1.22e5  # Amplitude of Square Wave stiffness

# Step 1: Get parameters from the simulation configuration
cyc_motor, gms_coarse_interval, shaft_polar_stiff, time_vctr, d_cyc_gms_coarse = get_parameters(RPS)

# Step 2: Load matrices from .mat files
M_mat = loadmat('M_matrix.mat')['M']  # Mass matrix
K_const = loadmat('Const_K_matrix.mat')['constK']  # Constant stiffness matrix
K_var = loadmat('Var_K_matrix.mat')['varKList']  # Variable stiffness matrix

# Step 3: calc_Euler_Lagrange_Components
K_cyc_list, cyc_vctr = calc_K_cyc_list(K_const, K_var, K_AMPLITUDE, d_cyc_gms_coarse)
C = calc_damping_matrix(K_cyc_list, M_mat)
FexVctrT = calc_FexVctrT(cyc_motor, shaft_polar_stiff, LOAD, ndof=K_cyc_list.shape[0])

# Step 4: Store results for further use
# DOF order: thetaOut, thetaIn, thetaBrake (or theta_2, theta_1, theta_b)
euler_lagrange_comp = {
    "Fex": FexVctrT,
    "K": K_cyc_list,
    "M": M_mat,
    "C": C,
    "time_vctr_Fex": time_vctr,
    "cyc_vctr_K": cyc_vctr,
}
