import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# === Dimensiuni domeniu dreptunghiular ===
a = 3  # lungimea pe axa x
b = 2  # lungimea pe axa y
N = 3  # număr de subîmpărțiri pe fiecare axă
hx = a / N
hy = b / N
hx2 = hx ** 2
hy2 = hy ** 2
n = (N + 1) ** 2

U = np.zeros((n, 1))
array = np.zeros((n, n))
b_vec = np.zeros((n, 1))

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

    # 2. Rezolvăm Qᵗ * b = c
    c = Q.T @ b

    # 3. Rezolvăm R * x = c prin substituție descendentă
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i].item() - np.dot(R[i, i+1:], x[i+1:])) / R[i, i].item()

    return x.reshape(-1, 1)

# Funcția exactă
def u(x, y):
    return np.exp(-((x - 1.5)**2 + (y - 1.0)**2)) * np.cos(2 * x) * np.sin(2 * y)

# Mapare index 2D în vector 1D
def node(i, j):
    return i + j * (N + 1)

# Umplem matricea
for j in range(N + 1):
    for i in range(N + 1):
        idx = node(i, j)
        if i == 0 or i == N or j == 0 or j == N:
            array[idx, idx] = 1
        else:
            array[idx, node(i, j - 1)] = -1 / hy2
            array[idx, node(i - 1, j)] = -1 / hx2
            array[idx, idx] = 2 / hx2 + 2 / hy2
            array[idx, node(i + 1, j)] = -1 / hx2
            array[idx, node(i, j + 1)] = -1 / hy2

# Funcție f(x, y)
def f(x, y):
    return (-1 / hx2) * (u(x - hx, y) + u(x + hx, y)) + \
           (-1 / hy2) * (u(x, y - hy) + u(x, y + hy)) + \
           (2 / hx2 + 2 / hy2) * u(x, y)

# Populăm vectorul b
for j in range(N + 1):
    for i in range(N + 1):
        idx = node(i, j)
        x = i * hx
        y = j * hy
        if i == 0 or i == N or j == 0 or j == N:
            b_vec[idx] = u(x, y)
        else:
            b_vec[idx] = f(x, y)

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

Z_vals = U.reshape((N + 1, N + 1)).T

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

# === GRAFIC EROARE ===
error = np.abs(Z_dense - Z_true)
print("Max error:", np.max(error))
plt.figure(figsize=(6, 5))
cp = plt.contourf(X_dense, Y_dense, error, levels=20, cmap='Reds')
plt.colorbar(cp)
plt.title("Eroare absolută")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
