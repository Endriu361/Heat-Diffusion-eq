import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import time

# === Domain parameters ===
a = 5
b = 2
n_vals = [4, 8 ,12 ,16]  # Different grid sizes to visualize
max_errors = []
h_vals = []


# === Boundary functions ===
def este_in_GammaD(x, y):
    return np.isclose(y, 0) or np.isclose(y, b)


def este_in_GammaN(x, y):
    return np.isclose(x, 0) or np.isclose(x, a)


def gD(x, y):
    return u(x, y)


def gN(x, y, hx):
    k_val = k(x, y)
    if np.isclose(x, 0):
        du_dn = (u(x + hx, y) - u(x, y)) / hx
    elif np.isclose(x, a):
        du_dn = (u(x, y) - u(x - hx, y)) / hx
    else:
        raise ValueError("gN defined only for x=0 or x=a")
    return -k_val * du_dn


# === QR solver (preserved as requested) ===
def rezolva_sistem_QR(A, b):
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0:
            raise ValueError("Linearly dependent columns - QR failed")
        Q[:, j] = v / R[j, j]
    c = Q.T @ b
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i].item() - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i].item()
    return x.reshape(-1, 1)


# === Exact functions and coefficients ===
def u(x, y):
    return np.sin(2 * np.pi * x / 3) * np.cos(np.pi * y / 2)


def f(x, y, hx, hy):
    # Central second-order derivatives
    du_dx = (u(x + hx, y) - u(x - hx, y)) / (2 * hx)
    du_dy = (u(x, y + hy) - u(x, y - hy)) / (2 * hy)

    dk_dx = (k(x + hx, y) - k(x - hx, y)) / (2 * hx)
    dk_dy = (k(x, y + hy) - k(x, y - hy)) / (2 * hy)

    d2u_dx2 = (u(x + hx, y) - 2 * u(x, y) + u(x - hx, y)) / hx ** 2
    d2u_dy2 = (u(x, y + hy) - 2 * u(x, y) + u(x, y - hy)) / hy ** 2

    term_x = dk_dx * du_dx + k(x, y) * d2u_dx2
    term_y = dk_dy * du_dy + k(x, y) * d2u_dy2

    return -(term_x + term_y)


def k(x, y):
    return 1 + 0.5 * np.sin(2 * np.pi * x) * np.exp(-y)


# === Build sparse system ===
def construieste_sistem_sparse(N_val):
    hx = a / N_val
    hy = b / N_val
    h_vals.append(hx)  # Store grid spacing
    hx2 = hx ** 2
    hy2 = hy ** 2
    n_total = (N_val + 1) ** 2

    # Use LIL format for efficient construction
    A = lil_matrix((n_total, n_total))
    b_vec = np.zeros(n_total)

    node = lambda i, j: i + j * (N_val + 1)

    for j in range(N_val + 1):
        for i in range(N_val + 1):
            idx = node(i, j)
            x = i * hx
            y = j * hy

            if este_in_GammaD(x, y):
                A[idx, idx] = 1
                b_vec[idx] = gD(x, y)

            elif este_in_GammaN(x, y):
                if i == 0:  # Left boundary
                    k_avg = 0.5 * (k(x, y) + k(x + hx, y))
                    A[idx, idx] = -k_avg / hx
                    A[idx, node(i + 1, j)] = k_avg / hx
                    b_vec[idx] = gN(x, y, hx)

                elif i == N_val:  # Right boundary
                    k_avg = 0.5 * (k(x, y) + k(x - hx, y))
                    A[idx, idx] = k_avg / hx
                    A[idx, node(i - 1, j)] = -k_avg / hx
                    b_vec[idx] = gN(x, y, hx)

            else:  # Interior node
                k_c = k(x, y)
                A[idx, node(i, j - 1)] = -k_c / hy2
                A[idx, node(i - 1, j)] = -k_c / hx2
                A[idx, idx] = 2 * k_c * (1 / hx2 + 1 / hy2)
                A[idx, node(i + 1, j)] = -k_c / hx2
                A[idx, node(i, j + 1)] = -k_c / hy2
                b_vec[idx] = f(x, y, hx, hy)

    return csr_matrix(A), b_vec


# === Quadratic spline interpolation (optimized) ===
def spline_patratica_1d(X, Y):
    n = len(X) - 1
    A = np.zeros((3 * n, 3 * n))
    b = np.zeros(3 * n)
    row = 0

    # Point matching conditions
    for i in range(n):
        A[row, 3 * i:3 * i + 3] = [0, 0, 1]  # a*0 + b*0 + c = Y_i
        b[row] = Y[i]
        row += 1

        dx = X[i + 1] - X[i]
        A[row, 3 * i:3 * i + 3] = [dx ** 2, dx, 1]  # a*dx^2 + b*dx + c = Y_{i+1}
        b[row] = Y[i + 1]
        row += 1

    # Continuity of derivatives
    for i in range(n - 1):
        dx = X[i + 1] - X[i]
        A[row, 3 * i:3 * i + 3] = [2 * dx, 1, 0]  # derivative at end of segment i
        A[row, 3 * (i + 1) + 1] = -1  # minus derivative at start of segment i+1
        row += 1

    # Natural spline condition (second derivative zero at start)
    A[row, 0] = 2
    row += 1

    # Solve system
    coef = rezolva_sistem_QR(A[:row], b[:row])

    def eval_spline(x):
        i = np.searchsorted(X, x) - 1
        i = max(0, min(i, n - 1))
        dx = x - X[i]
        a, b, c = coef[3 * i:3 * i + 3].flatten()
        return a * dx ** 2 + b * dx + c

    return eval_spline


def spline_bi2d(X, Y, Z):
    # Vectorized evaluation for efficiency
    row_splines = [spline_patratica_1d(X, Z[j, :]) for j in range(len(Y))]

    def eval_bi2d(x, y):
        z_vals = np.array([spline(x) for spline in row_splines])
        col_spline = spline_patratica_1d(Y, z_vals)
        return col_spline(y)

    return eval_bi2d


# === Visualization function with fixed sizes ===
def plot_solutions(X, Y, Z_num, Z_exact, N_val):
    error = np.abs(Z_num - Z_exact)
    max_error = np.max(error)
    print(f"N = {N_val}, Max error = {max_error:.6f}")

    # Create figure with fixed size
    fig = plt.figure(figsize=(21, 7))

    # Numerical solution
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z_num, cmap='viridis', rstride=1, cstride=1)
    ax1.set_title(f'Numerical Solution (N={N_val})', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_zlabel('u(x,y)', fontsize=12)
    ax1.tick_params(axis='both', labelsize=10)

    # Exact solution (same scale)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z_exact, cmap='plasma', rstride=1, cstride=1)
    ax2.set_title('Exact Solution', fontsize=14)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_zlabel('u(x,y)', fontsize=12)
    ax2.tick_params(axis='both', labelsize=10)

    # Set matching z-axis limits
    z_min = min(np.min(Z_num), np.min(Z_exact))
    z_max = max(np.max(Z_num), np.max(Z_exact))
    ax1.set_zlim(z_min, z_max)
    ax2.set_zlim(z_min, z_max)

    # Error plot
    ax3 = fig.add_subplot(133, projection='3d')
    surf = ax3.plot_surface(X, Y, error, cmap='coolwarm', rstride=1, cstride=1)
    ax3.set_title('Absolute Error', fontsize=14)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.set_zlabel('Error', fontsize=12)
    ax3.tick_params(axis='both', labelsize=10)

    # Add colorbar for error plot
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()

    return max_error


# === Main loop over grid sizes ===
for N_val in n_vals:
    print(f"\n=== Solving for N = {N_val} ===")
    start_time = time.time()

    # Build and solve system
    A_sparse, b_vec = construieste_sistem_sparse(N_val)
    U = rezolva_sistem_QR(A_sparse.toarray(), b_vec)  # Convert sparse to dense

    # Prepare grid
    x_vals = np.linspace(0, a, N_val + 1)
    y_vals = np.linspace(0, b, N_val + 1)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z_vals = U.reshape((N_val + 1, N_val + 1))

    # Create spline interpolant
    spline2d = spline_bi2d(x_vals, y_vals, Z_vals)

    # Evaluate on consistent dense grid (same for all N)
    x_dense = np.linspace(0, a, 100)
    y_dense = np.linspace(0, b, 100)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)

    # Vectorized evaluation
    Z_dense = np.zeros_like(X_dense)
    for j in range(len(y_dense)):
        for i in range(len(x_dense)):
            Z_dense[j, i] = spline2d(X_dense[j, i], Y_dense[j, i])

    # Exact solution on same dense grid
    Z_exact = u(X_dense, Y_dense)

    # Plot and record error
    max_error = plot_solutions(X_dense, Y_dense, Z_dense, Z_exact, N_val)
    max_errors.append(max_error)

    print(f"Solved in {time.time() - start_time:.2f} seconds")

# === Convergence analysis with slope calculation ===
plt.figure(figsize=(10, 7))
plt.plot(h_vals, max_errors, 'bo-', linewidth=2, markersize=8, label='Maximum Error')

# Set log-log scale
plt.xscale('log')
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Grid Spacing (h)', fontsize=12)
plt.ylabel('Maximum Absolute Error', fontsize=12)
plt.title('Convergence Rate Analysis', fontsize=14)

# Calculate convergence rate (slope)
log_h = np.log(np.array(h_vals))
log_e = np.log(np.array(max_errors))
A = np.vstack([log_h, np.ones(len(log_h))]).T
slope, intercept = np.linalg.lstsq(A, log_e, rcond=None)[0]

# Plot regression line
regression_line = np.exp(intercept) * np.array(h_vals) ** slope
plt.plot(h_vals, regression_line, 'r--',
         label=f'Linear Fit: slope = {slope:.4f}')

# Annotate slope values
for i in range(1, len(h_vals)):
    local_slope = (log_e[i] - log_e[i - 1]) / (log_h[i] - log_h[i - 1])
    plt.annotate(f'{local_slope:.2f}',
                 (h_vals[i], max_errors[i]),
                 xytext=(5, -10), textcoords='offset points',
                 fontsize=9)

plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

print(f"\n=== Convergence Analysis Summary ===")
print(f"Global convergence rate: {slope:.4f}")
print("Local convergence rates between consecutive grids:")
for i in range(1, len(h_vals)):
    local_rate = (np.log(max_errors[i]) - np.log(max_errors[i - 1])) / \
                 (np.log(h_vals[i]) - np.log(h_vals[i - 1]))
    print(f"h={h_vals[i - 1]:.4f} to h={h_vals[i]:.4f}: {local_rate:.4f}")