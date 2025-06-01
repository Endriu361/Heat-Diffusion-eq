import numpy as np
import matplotlib.pyplot as plt


# === Dimensiuni domeniu dreptunghiular ===
a = 3 # lungimea pe axa x
b = 2 # lungimea pe axa y
N = 3  # număr de subîmpărțiri pe fiecare axă
hx = a / N
hy = b / N
hx2 = hx ** 2
hy2 = hy ** 2
n = (N + 1) ** 2

def este_in_GammaD(x, y):
    return y == 0 or y == b

def este_in_GammaN(x, y):
    return x == 0 or x == a

def gD(x, y):
    return u(x, y)  # sau orice altă funcție

def gN(x, y):
    k_val = k(x, y)
    # calculul derivatelor normale, depinde de locație
    if x == 0:
        du_dn = (u(x + hx, y) - u(x, y)) / hx  # derivata spre interior
    elif x == a:
        du_dn = (u(x, y) - u(x - hx, y)) / hx
    else:
        raise ValueError("gN definit doar pe x=0 sau x=a")
    return -k_val * du_dn

def rezolva_sistem_QR(A, b):
    # 1. Factorizare QR cu Gram-Schmidt modificat
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
            raise ValueError("Coloane liniare dependent – QR eșuează.")
        Q[:, j] = v / R[j, j]

    # 2. Rezolvăm Qt * b = c
    c = Q.T @ b

    # 3. Rezolvăm R * x = c prin substituție descendentă
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i].item() - np.dot(R[i, i+1:], x[i+1:])) / R[i, i].item()

    return x.reshape(-1, 1)

# Funcția exactă
def u(x, y):
    return np.sin(2 * np.pi * x / 3) * np.cos(np.pi * y / 2)

def k(x, y):
    return 2 + np.cos(2 * np.pi * x / 3) * np.sin(np.pi * y / 2)

def f(x, y):
    # Derivate centrate de ordin 2
    du_dx = (u(x + hx, y) - u(x - hx, y)) / (2 * hx)
    du_dy = (u(x, y + hy) - u(x, y - hy)) / (2 * hy)

    dk_dx = (k(x + hx, y) - k(x - hx, y)) / (2 * hx)
    dk_dy = (k(x, y + hy) - k(x, y - hy)) / (2 * hy)

    d2u_dx2 = (u(x + hx, y) - 2 * u(x, y) + u(x - hx, y)) / hx**2
    d2u_dy2 = (u(x, y + hy) - 2 * u(x, y) + u(x, y - hy)) / hy**2

    term_x = dk_dx * du_dx + k(x, y) * d2u_dx2
    term_y = dk_dy * du_dy + k(x, y) * d2u_dy2

    return -(term_x + term_y)

# Mapare index 2D în vector 1D
def node(i, j):
    return i + j * (N + 1)

def construieste_sistem(gD, gN, este_in_GammaD, este_in_GammaN):
    A = np.zeros((n, n))
    b_vec = np.zeros((n, 1))
    for j in range(N + 1):
        for i in range(N + 1):
            idx = node(i, j)
            x = i * hx
            y = j * hy

            if este_in_GammaD(x, y):
                A[idx, idx] = 1
                b_vec[idx] = gD(x, y)

            elif este_in_GammaN(x, y):
                if i ==  0:
                    k_avg = 0.5 * (k(x, y) + k(x + hx, y))
                    A[idx, node(i, j)] = -k_avg / hx
                    A[idx, node(i + 1, j)] = k_avg / hx
                    b_vec[idx] = gN(x, y)
                elif i == N:
                    k_avg = 0.5 * (k(x, y) + k(x - hx, y))
                    A[idx, node(i, j)] = k_avg / hx
                    A[idx, node(i - 1, j)] = -k_avg / hx
                    b_vec[idx] = gN(x, y)
            else:
                # Nod interior – ecuație diferențială
                k_c = k(x, y)
                A[idx, node(i, j - 1)] = -k_c / hy2
                A[idx, node(i - 1, j)] = -k_c / hx2
                A[idx, idx] = 2 * k_c * (1 / hx2 + 1 / hy2)
                A[idx, node(i + 1, j)] = -k_c / hx2
                A[idx, node(i, j + 1)] = -k_c / hy2
                b_vec[idx] = f(x, y)
    return A, b_vec

array, b_vec = construieste_sistem(gD, gN, este_in_GammaD, este_in_GammaN)
# Rezolvăm sistemul
U = rezolva_sistem_QR(array, b_vec)

# Interpolare spline pătratică 1D
def spline_patratica_1d(X, Y):
    n = len(X) - 1
    A = np.zeros((3 * n, 3 * n))
    b = np.zeros(3 * n)
    row = 0
    for i in range(n):
        A[row, 3 * i:3 * i + 3] = [0, 0, 1]
        b[row] = Y[i]
        row += 1
        dx = X[i+1] - X[i]
        A[row, 3 * i:3 * i + 3] = [dx ** 2, dx, 1]
        b[row] = Y[i+1]
        row += 1
    for i in range(n - 1):
        dx = X[i+1] - X[i]
        A[row, 3 * i:3 * i + 2] = [2 * dx, 1]
        A[row, 3 * (i + 1) + 1] = -1
        row += 1
    A[row, 1] = 1
    b[row] = (Y[1].item() - Y[0].item()) / (X[1] - X[0])
    coef = rezolva_sistem_QR(A, b)

    def eval_spline(x):
        for i in range(n):
            if X[i] <= x <= X[i+1]:
                a, b_, c = coef[3 * i:3 * i + 3]
                dx = x - X[i]
                return a * dx ** 2 + b_ * dx + c
        return None
    return eval_spline

# Setăm grila de valori pentru domeniul dreptunghiular
x_vals = np.linspace(0, a, N + 1)
y_vals = np.linspace(0, b, N + 1)

Z_vals = U.reshape((N + 1, N + 1))
print(Z_vals)

# Interpolare 2D spline pătratic
def spline_bi2d(X, Y, Z):
    row_splines = []
    for j in range(len(Y)):
        row_interp = spline_patratica_1d(X, Z[j, :])
        row_splines.append(row_interp)

    def eval_bi2d(x, y):
        z_temp = np.array([row_splines[j](x).item() for j in range(len(Y))])
        final_spline = spline_patratica_1d(Y, z_temp)
        return final_spline(y).item()

    return eval_bi2d

# Construim spline-ul 2D
spline2d = spline_bi2d(x_vals, y_vals, Z_vals)

# Grilă densă pentru grafic
x_dense = np.linspace(0, a, 100)
y_dense = np.linspace(0, b, 100)
X_dense, Y_dense = np.meshgrid(x_dense, y_dense)

Z_dense = np.zeros_like(X_dense)
for j in range(Y_dense.shape[0]):
    for i in range(X_dense.shape[1]):
        Z_dense[j, i] = spline2d(X_dense[j, i], Y_dense[j, i])

# Funcția reală pentru comparație
Z_true = u(X_dense, Y_dense)

error = np.abs(Z_dense - Z_true)
print("Diferenta majora:", np.max(error))
plt.figure(figsize=(6, 5))
cp = plt.contourf(X_dense, Y_dense, error, levels=20, cmap='Reds')
plt.colorbar(cp)
plt.title("Eroare absolută")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# === GRAFIC Spline vs Funcție reală ===
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X_dense, Y_dense, Z_dense, cmap='coolwarm', alpha=0.8)
ax1.set_title("Spline pătratică 2D")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X_dense, Y_dense, Z_true, cmap='viridis', alpha=0.8)
ax2.set_title("Funcția reală")
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

plt.tight_layout()
plt.show()

def calculeaza_eroare(N):
    global hx, hy, hx2, hy2, n
    hx = a / N
    hy = b / N
    hx2 = hx ** 2
    hy2 = hy ** 2
    n = (N + 1) ** 2
    def node(i, j): return i + j * (N + 1)
    array, b_vec = construieste_sistem(gD, gN, este_in_GammaD, este_in_GammaN)
    U, *_ = np.linalg.lstsq(array, b_vec, rcond=None)
    x_vals = np.linspace(0, a, N + 1)
    y_vals = np.linspace(0, b, N + 1)
    Z_vals = U.reshape((N + 1, N + 1))
    spline2d = spline_bi2d(x_vals, y_vals, Z_vals)
    x_dense = np.linspace(0, a, 100)
    y_dense = np.linspace(0, b, 100)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
    Z_dense = np.zeros_like(X_dense)
    for j in range(Y_dense.shape[0]):
        for i in range(X_dense.shape[1]):
            Z_dense[j, i] = spline2d(X_dense[j, i], Y_dense[j, i])
    Z_exact = u(X_dense, Y_dense)
    eroare_relativa = np.linalg.norm(Z_dense - Z_exact) / np.linalg.norm(Z_exact)
    return hx, eroare_relativa

Ns = [4, 8, 16, 32]
Hs = []
Erori = []
for N_val in Ns:
    h, err = calculeaza_eroare(N_val)
    Hs.append(h)
    Erori.append(err)
    print(f"N={N_val:2d}, h={h:.4f}, eroare relativă={err:.4e}")

# === Grafic log-log ===
plt.figure(figsize=(6, 5))
plt.loglog(Hs, Erori, 'o-', label='Eroare relativă')
plt.xlabel("Pas de discretizare h (log)")
plt.ylabel("Eroare relativă (log)")
plt.title("Ordinea de convergență (log-log)")
plt.grid(True, which='both')
plt.legend()
panta, _ = np.polyfit(np.log(Hs), np.log(Erori), 1)
print(f"\nOrdin estimat de convergență:{abs(panta):.2f}")
plt.show()