import numpy as np
import matplotlib.pyplot as plt
import cv2
import sympy as sp

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
    vartheta = np.arctan2(numerator, denominator)

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

def Es_parameters(info_1, info_2, p=False):
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

    # Define reference fields
    Er_1 = np.array([[1], [0]])  # Reference vector 1
    Er_2 = np.array([[1], [1]])  # Reference vector 2

    # Calculate Esx and Esy
    Esx = info_1[1] / np.abs(Er_1[0])
    Esy = np.sqrt(info_1[0] - Esx**2 - Er_1[0]**2)
    Es_c = np.array([[Esx], [Esy]])  # Construct estimated field vector

    # Compute squared norms of reference and estimated fields
    norm_Er2_sq = np.linalg.norm(Er_2)**2
    norm_Es_c_sq = np.linalg.norm(Es_c)**2
    
    # this inot working by coding but by handmade
    # Check later
    # Compute numerator and denominator for phase calculation
    # numerator = (norm_Er2_sq + norm_Es_c_sq) * np.tan(info_2[2])[0]
    # denominator = norm_Er2_sq + norm_Es_c_sq + 2 * np.abs(Er_2[0])[0] * np.abs(Es_c[0])[0] * np.abs(Er_2[1])[0] * np.abs(Es_c[1])[0]
    # delta_phi_s = np.arctan2(numerator, denominator)

    # Define symbolic variable for phase shift
    phi_s = sp.Symbol('Delta_phi_s')

    # Define symbolic matrices for field components
    Es_c_sp = sp.Matrix([Esx[0], Esy[0]])  # Symbolic estimated field vector
    Er_2_sp = sp.Matrix([Er_2[0,0], Er_2[1,0]])  # Symbolic reference vector

    # Expression for phase shift calculation
    expr = sp.atan2(-Er_2_sp[1] * Es_c_sp[1] * sp.sin(-phi_s), 
                    Er_2_sp[0] * Es_c_sp[0] + Er_2_sp[1] * Es_c_sp[1] * sp.cos(-phi_s)) - info_2[2]

    # Solve for Delta_phi_s
    solution = sp.solve(expr, phi_s)
    delta_phi_s = solution[0] if solution else "No symbolic solution found."

    # Convert Esx and Esy to scalar values
    Esx = Esx[0]
    Esy = Esy[0]
    delta_phi_s = delta_phi_s[0] if isinstance(delta_phi_s, (list, tuple)) else delta_phi_s

    # Print results if required
    if p:
        print("Calculated Parameters:")
        print(f"Esx: {Esx}")
        print(f"Esy: {Esy}")
        print(f"Delta_phi_s: {delta_phi_s}")
    # Return computed values
    return Esx, Esy, delta_phi_s




