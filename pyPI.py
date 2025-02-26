import numpy as np
import matplotlib.pyplot as plt
import cv2
import sympy as sp


def field_notation(E, p=False):
    """
    Converts electric field vector into field notation.
    """
    E_x = np.abs(E[0,0])
    phi_x = np.angle(E[0,0])
    E_y = np.abs(E[1,0])
    phi_y = np.angle(E[1,0])
    delta_phi = np.abs(phi_y - phi_x)

    if p:
        print(E_x)
        print(E_y)
        print(f"{delta_phi/np.pi}π")

    return np.array([[E_x], [E_y * np.exp(1j*delta_phi)]])

def I(E_r, E_s, mu=0, plot=False):
    """
    Computes the interference intensity pattern for given reference and sample electric field vectors.

    Parameters:
    E_r (numpy array): Reference electric field vector (2x1 complex array).
    E_s (numpy array): Sample electric field vector (2x1 complex array).
    mu (float, optional): Global phase shift (default=0).
    plot (bool, optional): If True, displays the interference pattern (default=False).

    Returns:
    tuple: (info, I) where:
        - info: List containing Omega, Psi, and vartheta parameters.
        - I: Computed intensity array.
    """
    lambda_ = 532e-9  # Wavelength in meters
    k = 2 * np.pi / lambda_  # Wave number
    x = np.linspace(-k, k, 500)  # Spatial positions along x-axis

    # Compute intensity components
    Omega = np.linalg.norm(E_r)**2 + np.linalg.norm(E_s)**2
    Psi = np.abs(np.dot(E_r.T, E_s))[0, 0]

    # Compute relative phase (vartheta)
    numerator = -np.abs(E_r[1]) * np.abs(E_s[1]) * np.sin(np.angle(E_r[1]) - np.angle(E_s[1]))
    denominator = np.abs(E_r[0]) * np.abs(E_s[0]) + np.abs(E_r[1]) * np.abs(E_s[1]) * np.cos(np.angle(E_r[1]) - np.angle(E_s[1]))
    vartheta = np.arctan(numerator/denominator)[0]

    # Phase modulation
    zeta = k * x - vartheta
    cos_term = np.cos(mu + zeta)
    I = Omega + Psi * cos_term
    info = [Omega, Psi, vartheta]

    # Normalize intensity to scale between 0 and 255 (grayscale image)
    I_norm = ((I - I.min()) / (I.max() - I.min()) * 255).astype(np.uint8)

    # Create grayscale interference pattern image
    height = 500  # Image height in pixels
    pattern_image = np.tile(I_norm, (height, 1))

    # Apply Gaussian blur to smooth fringes
    pattern_image = cv2.GaussianBlur(pattern_image, (5, 5), 0)
    
    if plot:
        # Display the interferogram
        plt.figure(figsize=(8, 6))
        plt.imshow(pattern_image, cmap='gray', aspect='auto', extent=[x.min(), x.max(), 0, height])
        plt.xlabel("x")
        plt.ylabel("Intensity")
        plt.title("Interferogram")
        plt.show()
    
    return info, I

def Es_parameters(Er_1, Er_2, info_1, info_2, p=False):
    """
    Computes the parameters Esx, Esy, and Delta_phi_s for a given electric field vector Es.

    Parameters:
    Es (numpy array): 2x1 complex electric field vector.
    p (bool, optional): If True, prints the calculated parameters. Default is False.

    Returns:
    tuple: (Esx, Esy, Delta_phi_s) where:
        - Esx: x-component of the estimated electric field.
        - Esy: y-component of the estimated electric field.
        - Delta_phi_s: Phase shift between components.
    """

    # Calculate Esx and Esy
    Esx = info_1[1] / np.abs(Er_1[0])
    Esy = np.sqrt(info_1[0] - Esx**2 - Er_1[0]**2)
    Es_c = np.array([[Esx], [Esy]])  # Construct estimated field vector

    # Calculating the numerator and denominator to calculate delta phi
    numerador = Es_c[1, 0] * Er_2[1, 0] - np.sqrt(-Er_2[0, 0]**2 * Es_c[0, 0]**2 * np.tan(info_2[2])**2 + Er_2[0, 0]**2 * Es_c[1, 0]**2 * np.tan(info_2[2])**2 + Er_2[1, 0]**2 * Es_c[1, 0]**2)
    denominador = (Er_2[0, 0] * Es_c[0, 0] - Er_2[1, 0] * Es_c[1, 0]) * np.tan(info_2[2])

    # Calculation of delta_phi_s
    if denominador == 0:
        delta_phi_s = 0
    else:
        delta_phi_s = 2 * np.arctan(numerador / denominador)[0]

    # Convert Esx and Esy to scalar values
    Esx = np.abs(Esx[0])
    Esy = np.abs(Esy[0])

    E = np.array([[Esx], [Esy * np.exp(1j*delta_phi_s)]])

    # Print results if required
    if p:
        print("Calculated Parameters:")
        print(f"Esx: {np.abs(E[0,0])}")
        print(f"Esy: {np.abs(E[1,0])}")
        print(f"Delta_phi_s: {np.angle(E[1,0])/np.pi}π")
    # Return computed values
    return E

def jones_matrix(delta, alpha):
    """
    Computes the Jones matrix for a birefringent optical element.

    Parameters:
    delta : float
        The phase retardation introduced by the birefringent material (in radians).
    alpha : float
        The angle (in radians) of the fast axis with respect to the reference axis.

    Returns:
    np.ndarray
        A 2x2 complex-valued numpy array representing the Jones matrix.
    """

    # Compute the elements of the Jones matrix
    m_11 = (np.cos(alpha) ** 2 + np.exp(-1j * delta) * np.sin(alpha) ** 2)  # First row, first column
    m_12 = ((1 - np.exp(-1j * delta)) * np.cos(alpha) * np.sin(alpha))      # First row, second column
    m_21 = ((1 - np.exp(-1j * delta)) * np.cos(alpha) * np.sin(alpha))      # Second row, first column
    m_22 = (np.sin(alpha) ** 2 + np.exp(-1j * delta) * np.cos(alpha) ** 2)  # Second row, second column

    # Construct the 2x2 Jones matrix
    M = np.array([[m_11, m_12], 
                  [m_21, m_22]])

    return M  # Return the computed Jones matrix

def plot_alpha_variation(Eis, delta_chi):
    """
    Plots the variation of |Esx|, |Esy|, and Delta_phi_s as a function of alpha.
    
    Parameters:
    Eis : ndarray
        Incident electric field vector.
    delta_chi : float
        Phase shift introduced by the birefringent material.
    """
    alpha_values = np.linspace(0, 2 * np.pi, 100)
    Esx_values = []
    Esy_values = []
    delta_phi_s_values = []
    
    key_alphas = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    key_values = []
    
    for alpha in alpha_values:

        S = jones_matrix(delta_chi, alpha)
        
        Es = S @ Eis
        delta_phi_s = np.angle(Es[1,0]) - np.angle(Es[0,0])
        
        Esx_values.append(np.abs(Es[0])[0])
        Esy_values.append(np.abs(Es[1])[0])
        delta_phi_s_values.append(delta_phi_s)
        
        if alpha in key_alphas:
            key_values.append((alpha, np.abs(Es[0])[0], np.abs(Es[1])[0], delta_phi_s))
    
    # Líneas de referencia
    def plot_reference_lines():
        for alpha in key_alphas:
            plt.axvline(x=alpha, color='gray', linestyle='--', alpha=0.6)
    
    # Gráfica de Esx
    plt.figure()
    plt.plot(alpha_values, Esx_values, label='|Esx|', color='b')
    plot_reference_lines()
    Esx_0 = key_values[0][1]  # Valor de Esx cuando alpha = 0
    plt.axhline(y=Esx_0, color='b', linestyle='dotted')
    for alpha, Esx, _, _ in key_values:
        plt.scatter(alpha, Esx, color='b')
    plt.xlabel('Alpha (n π)')
    plt.xticks(key_alphas, ['0', 'π/2', 'π', '3π/2', '2π'])
    plt.ylabel('|Esx|')
    plt.title('Variación de Esx con Alpha')
    plt.grid()
    plt.legend()
    plt.show()
    
    # Gráfica de Esy
    plt.figure()
    plt.plot(alpha_values, Esy_values, label='|Esy|', color='r')
    plot_reference_lines()
    Esy_0 = key_values[0][2]  # Valor de Esy cuando alpha = 0
    plt.axhline(y=Esy_0, color='r', linestyle='dotted')
    for alpha, _, Esy, _ in key_values:
        plt.scatter(alpha, Esy, color='r')
    plt.xlabel('Alpha (n π)')
    plt.xticks(key_alphas, ['0', 'π/2', 'π', '3π/2', '2π'])
    plt.ylabel('|Esy|')
    plt.title('Variación de Esy con Alpha')
    plt.grid()
    plt.legend()
    plt.show()
    
    # Gráfica de delta_phi_s
    plt.figure()
    plt.plot(alpha_values, delta_phi_s_values, label='Delta_phi_s', color='g')
    plot_reference_lines()
    delta_phi_0 = key_values[0][3]  # Valor de Delta_phi_s cuando alpha = 0
    plt.axhline(y=delta_phi_0, color='g', linestyle='dotted')
    for alpha, _, _, delta_phi_s in key_values:
        plt.scatter(alpha, delta_phi_s, color='g')
    plt.xlabel('Alpha (n π)')
    plt.xticks(key_alphas, ['0', 'π/2', 'π', '3π/2', '2π'])
    plt.ylabel('Delta_phi_s (rad)')
    plt.title('Variación de Delta_phi_s con Alpha')
    plt.grid()
    plt.legend()
    plt.show()
    
    # Crear tabla con valores clave
    #df = pd.DataFrame(key_values, columns=['Alpha (rad)', '|Esx|', '|Esy|', 'Delta_phi_s'])
    #print(df)




