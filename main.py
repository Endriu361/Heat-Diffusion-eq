import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
import time
from scipy.sparse.linalg import spsolve

# =============================================================================
# PARAMETRII DE BAZĂ AI DOMENIULUI
# =============================================================================
a = 5  # Lungimea domeniului pe axa Ox
b = 2  # Lungimea domeniului pe axa Oy
n_vals = [4, 8, 16, 32, 64, 128]  # Dimensiunile grilelor pentru testare
max_errors = []  # Lista pentru erorile maxime
h_vals = []  # Lista pentru pașii de discretizare

# =============================================================================
# FUNCȚII EXACTE ȘI COEFICIENȚI
# =============================================================================
def u(x, y):
    """Soluția analitică a problemei"""
    return np.sin(2 * np.pi * x / 3) * np.cos(np.pi * y / 2)


def k(x, y):
    """Coeficientul de difuzie al ecuației"""
    return 1 + 0.5 * np.sin(2 * np.pi * x) * np.exp(-y)


def f(x, y, hx, hy):
    """Termenul sursă calculat cu diferențe finite"""
    # Calcul derivate parțiale de ordinul 1
    du_dx = (u(x + hx, y) - u(x - hx, y)) / (2 * hx)
    du_dy = (u(x, y + hy) - u(x, y - hy)) / (2 * hy)

    dk_dx = (k(x + hx, y) - k(x - hx, y)) / (2 * hx)
    dk_dy = (k(x, y + hy) - k(x, y - hy)) / (2 * hy)

    # Calcul derivate parțiale de ordinul 2
    d2u_dx2 = (u(x + hx, y) - 2 * u(x, y) + u(x - hx, y)) / hx ** 2
    d2u_dy2 = (u(x, y + hy) - 2 * u(x, y) + u(x, y - hy)) / hy ** 2

    # Calcul termen sursă
    term_x = dk_dx * du_dx + k(x, y) * d2u_dx2
    term_y = dk_dy * du_dy + k(x, y) * d2u_dy2

    return -(term_x + term_y)


# =============================================================================
# FUNCȚII PENTRU CONDIIȚIILE LA LIMITĂ
# =============================================================================

def este_in_GammaD(x, y):
    """Verifică dacă punctul (x,y) aparține frontierei Dirichlet (y=0 sau y=b)"""
    return np.isclose(y, 0) or np.isclose(y, b)


def este_in_GammaN(x, y):
    """Verifică dacă punctul (x,y) aparține frontierei Neumann (x=0 sau x=a)"""
    return np.isclose(x, 0) or np.isclose(x, a)


def gD(x, y):
    """Condiție Dirichlet - valoarea exactă a soluției pe frontieră"""
    return u(x, y)


def gN(x, y, hx):
    """Condiție Neumann - fluxul normal pe frontieră"""
    k_val = k(x, y)
    if np.isclose(x, 0):
        # Derivată normală pe latura stângă
        du_dn = (u(x + hx, y) - u(x, y)) / hx
    elif np.isclose(x, a):
        # Derivată normală pe latura dreaptă
        du_dn = (u(x, y) - u(x - hx, y)) / hx
    else:
        raise ValueError("gN definit doar pentru x=0 sau x=a")
    return -k_val * du_dn


# =============================================================================
# SOLVER QR PENTRU SISTEME LINIARE
# =============================================================================

def rezolva_sistem_QR(A, b):
    """Rezolvă sistemul Ax=b folosind descompunerea QR"""
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    # Factorizare QR cu Gram-Schmidt modificat
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0:
            raise ValueError("Coloane liniar dependente - QR eșuat")
        Q[:, j] = v / R[j, j]

    # Calculul vectorului c = Q^T * b
    c = Q.T @ b

    # Rezolvare sistem triunghiular superior
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i].item() - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i].item()

    return x.reshape(-1, 1)

# =============================================================================
# CONSTRUIREA SISTEMULUI MATRICIAL (FORMAT RAR)
# =============================================================================

def construieste_sistem_sparse(N_val):
    """Construiește sistemul discretizat în format rar"""
    hx = a / N_val
    hy = b / N_val
    h_vals.append(hx)  # Stocare pas de discretizare
    hx2 = hx ** 2
    hy2 = hy ** 2
    n_total = (N_val + 1) ** 2  # Numărul total de noduri

    # Inițializare matrice rară și vector
    A = lil_matrix((n_total, n_total))
    b_vec = np.zeros(n_total)

    # Funcție de mapare nod (i,j) -> index global
    node = lambda i, j: i + j * (N_val + 1)

    # Parcurgerea tuturor nodurilor grilei
    for j in range(N_val + 1):
        for i in range(N_val + 1):
            idx = node(i, j)
            x = i * hx
            y = j * hy

            # Tratare noduri pe frontiera Dirichlet
            if este_in_GammaD(x, y):
                A[idx, idx] = 1
                b_vec[idx] = gD(x, y)

            # Tratare noduri pe frontiera Neumann
            elif este_in_GammaN(x, y):
                if i == 0:  # Latura stângă
                    k_avg = 0.5 * (k(x, y) + k(x + hx, y))
                    A[idx, idx] = -k_avg / hx
                    A[idx, node(i + 1, j)] = k_avg / hx
                    b_vec[idx] = gN(x, y, hx)

                elif i == N_val:  # Latura dreaptă
                    k_avg = 0.5 * (k(x, y) + k(x - hx, y))
                    A[idx, idx] = k_avg / hx
                    A[idx, node(i - 1, j)] = -k_avg / hx
                    b_vec[idx] = gN(x, y, hx)

            # Tratare noduri interne
            else:
                k_c = k(x, y)
                # Coeficienți pentru diferențe finite
                A[idx, node(i, j - 1)] = -k_c / hy2
                A[idx, node(i - 1, j)] = -k_c / hx2
                A[idx, idx] = 2 * k_c * (1 / hx2 + 1 / hy2)
                A[idx, node(i + 1, j)] = -k_c / hx2
                A[idx, node(i, j + 1)] = -k_c / hy2
                b_vec[idx] = f(x, y, hx, hy)

    return csr_matrix(A), b_vec


# =============================================================================
# INTERPOLARE SPLINE PĂTRATICĂ
# =============================================================================

def spline_patratica_1d(X, Y):
    """Construiește un spline pătratic 1D pentru datele date"""
    n = len(X) - 1
    A = np.zeros((3 * n, 3 * n))
    b = np.zeros(3 * n)
    row = 0

    # Condiții de potrivire în puncte
    for i in range(n):
        # Valoare la începutul intervalului
        A[row, 3 * i:3 * i + 3] = [0, 0, 1]
        b[row] = Y[i]
        row += 1

        # Valoare la sfârșitul intervalului
        dx = X[i + 1] - X[i]
        A[row, 3 * i:3 * i + 3] = [dx ** 2, dx, 1]
        b[row] = Y[i + 1]
        row += 1

    # Continuitatea derivatelor
    for i in range(n - 1):
        dx = X[i + 1] - X[i]
        # Derivata la sfârșitul intervalului i
        A[row, 3 * i:3 * i + 3] = [2 * dx, 1, 0]
        # Derivata la începutul intervalului i+1
        A[row, 3 * (i + 1) + 1] = -1
        row += 1

    # Condiție spline natural (derivata a doua zero la început)
    A[row, 0] = 2
    row += 1

    # Rezolvare sistem
    coef = rezolva_sistem_QR(A[:row], b[:row])

    # Funcție de evaluare spline
    def eval_spline(x):
        i = np.searchsorted(X, x) - 1
        i = max(0, min(i, n - 1))
        dx = x - X[i]
        a_coef, b_coef, c_coef = coef[3 * i:3 * i + 3].flatten()
        return a_coef * dx ** 2 + b_coef * dx + c_coef

    return eval_spline


def spline_bi2d(X, Y, Z):
    """Construiește un spline pătratic bidimensional"""
    # Spline-uri de-a lungul fiecărui rând
    row_splines = [spline_patratica_1d(X, Z[j, :]) for j in range(len(Y))]
    return row_splines

# =============================================================================
# VIZUALIZARE REZULTATE
# =============================================================================

def plot_solutions(X, Y, Z_num, Z_exact, N_val):
    """Generează grafice 3D pentru soluții și erori"""
    error = np.abs(Z_num - Z_exact)
    max_error = np.max(error)
    print(f"N = {N_val}, Eroare maximă = {max_error:.6f}")

    # Creare figură cu dimensiuni fixe
    fig = plt.figure(figsize=(21, 7))

    # Soluția numerică
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z_num, cmap='viridis', rstride=1, cstride=1)
    ax1.set_title(f'Soluția Numerică (N={N_val})', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_zlabel('u(x,y)', fontsize=12)

    # Soluția exactă
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z_exact, cmap='plasma', rstride=1, cstride=1)
    ax2.set_title('Soluția Analitică', fontsize=14)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_zlabel('u(x,y)', fontsize=12)

    # Setare limite comune pentru axa z
    z_min = min(np.min(Z_num), np.min(Z_exact))
    z_max = max(np.max(Z_num), np.max(Z_exact))
    ax1.set_zlim(z_min, z_max)
    ax2.set_zlim(z_min, z_max)

    # Eroarea absolută
    ax3 = fig.add_subplot(133, projection='3d')
    surf = ax3.plot_surface(X, Y, error, cmap='coolwarm', rstride=1, cstride=1)
    ax3.set_title('Eroare Absolută', fontsize=14)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.set_zlabel('Eroare', fontsize=12)

    # Adăugare bară de culoare
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5)
    fig.tight_layout(rect=(0., 0., 1., 0.95))  # Lasă loc pentru titluri
    plt.show()

    return max_error


# =============================================================================
# BUCLA PRINCIPALĂ - REZOLVARE PENTRU DIFERITE DIMENSIUNI DE GRILE
# =============================================================================

for N_val in n_vals:
    print(f"\n=== Rezolvare pentru N = {N_val} ===")
    start_time = time.time()

    # Construire și rezolvare sistem
    A_sparse, b_vec = construieste_sistem_sparse(N_val)
    U = spsolve(A_sparse, b_vec)

    # Pregătire grilă
    x_vals = np.linspace(0, a, N_val + 1)
    y_vals = np.linspace(0, b, N_val + 1)
    Z_vals = U.reshape((N_val + 1, N_val + 1))

    # Construire interpolant spline
    row_splines = spline_bi2d(x_vals, y_vals, Z_vals)

    # Evaluare pe grilă fină bazată pe dimensiunea problemei
    if N_val <= 64:
        num_dense = 50
    elif N_val <= 128:
        num_dense = 30
    else:
        num_dense = 20

    # Evaluare pe grilă densă
    x_dense = np.linspace(0, a, num_dense)
    y_dense = np.linspace(0, b, num_dense)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)

    # Evaluare spline pe grilă densă
    Z_row = np.zeros((len(y_vals), len(x_dense)))
    for j, spl in enumerate(row_splines):
        Z_row[j, :] = np.array([spl(x) for x in x_dense])

    # Interpolare coloană cu spline pătratic
    Z_dense = np.zeros((len(y_dense), len(x_dense)))
    for i in range(len(x_dense)):
        col_spline = spline_patratica_1d(y_vals, Z_row[:, i])
        Z_dense[:, i] = np.array([col_spline(y) for y in y_dense])

    # Calcul soluție exactă
    Z_exact = u(X_dense, Y_dense)

    # Vizualizare și înregistrare eroare
    max_error = plot_solutions(X_dense, Y_dense, Z_dense, Z_exact, N_val)
    max_errors.append(max_error)

    print(f"Timp execuție: {time.time() - start_time:.2f} secunde")

# =============================================================================
# ANALIZA CONVERGENȚEI
# =============================================================================

# Scalare logaritmică
plt.figure(figsize=(10, 6))
plt.plot(n_vals, max_errors, 'o-', linewidth=2)
plt.xlabel('Număr de intervale (N)')
plt.ylabel('Eroare Maximă Absolută')
plt.title('Analiză Rată de Convergență')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.tight_layout()
plt.show()

# Calcul rata convergență
log_h = np.log(np.array(h_vals))
log_e = np.log(np.array(max_errors))
A = np.vstack([log_h, np.ones(len(log_h))]).T
slope, intercept = np.linalg.lstsq(A, log_e, rcond=None)[0]


# Rezumat analiză convergență
print("\n=== REZUMAT ANALIZĂ CONVERGENȚĂ ===")
print(f"Rata globală de convergență: {slope:.4f}")
print("Rate locale de convergență între grile consecutive:")
for i in range(1, len(h_vals)):
    local_rate = (np.log(max_errors[i]) - np.log(max_errors[i - 1])) / (np.log(h_vals[i]) - np.log(h_vals[i - 1]))
    print(f"h={h_vals[i - 1]:.4f} la h={h_vals[i]:.4f}: {local_rate:.4f}")