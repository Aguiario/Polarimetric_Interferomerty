import sympy as sp

def symbolic_intensity(values = None):
    alpha, chi, mu, k, x, phi_is, phi_r = sp.symbols('alpha chi mu k x phi_is phi_r', real=True)
    A_isx, A_isy, A_rx, A_ry = sp.symbols('A_isx A_isy A_rx A_ry', real=True)

    A_r = sp.Matrix([[A_rx], [A_ry * sp.exp(1j * phi_r)]])

    # Definir los coeficientes de la matriz de la lámina de onda
    S_xx = sp.cos(alpha)**2 + sp.exp(-1j * chi) * sp.sin(alpha)**2
    S_xy = (1 - sp.exp(-1j * chi)) * sp.cos(alpha) * sp.sin(alpha)
    S_yx = S_xy
    S_yy = sp.sin(alpha)**2 + sp.exp(-1j * chi) * sp.cos(alpha)**2

    # Transformar el vector A_is con la lámina de onda
    E_sx = S_xx * A_isx + S_xy * A_isy * sp.exp(1j * phi_is)
    E_sy = S_yx * A_isx + S_yy * A_isy * sp.exp(1j * phi_is)
    E_s = sp.Matrix([[E_sx], [E_sy]])

    # Expresiones para la intensidad
    Omega = A_r.norm()**2 + E_s.norm()**2
    Psi = sp.Abs(A_r.dot(E_s))

    # Cálculo de la fase relativa (vartheta)
    numerator = -sp.Abs(A_r[1,0]) * sp.Abs(E_s[1]) * sp.sin(sp.arg(A_r[1,0]) - sp.arg(E_s[1]))
    denominator = sp.Abs(A_r[0,0]) * sp.Abs(E_s[0]) + sp.Abs(A_r[1,0]) * sp.Abs(E_s[1]) * sp.cos(sp.arg(A_r[1,0]) - sp.arg(E_s[1]))
    vartheta = sp.atan(numerator/ denominator)

    # Modulación de fase
    zeta = 2 * (k * x - vartheta)
    cos_term = sp.cos(mu + zeta)

    I = Omega + Psi * cos_term

    if values:
        I = I.subs(values).evalf()

    return I
