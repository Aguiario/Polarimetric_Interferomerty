import numpy as np
import matplotlib.pyplot as plt
import cv2

def I(E_r, E_s, S=0, mu=0, plot=False):
    lambda_ = 532e-9  # Longitud de onda en metros
    k = 2 * np.pi / lambda_  # Número de onda
    x = np.linspace(-k, k, 500)  # Rango de posiciones en el eje x
    
    # Cálculo de términos de intensidad
    Omega = np.linalg.norm(E_r)**2 + np.linalg.norm(E_s)**2
    Psi = np.abs(np.dot(E_r.T, E_s))[0, 0]
    
    # Cálculo de la fase relativa vartheta
    numerador = -np.abs(E_r[1]) * np.abs(E_s[1]) * np.sin(np.angle(E_r[1]) - np.angle(E_s[1]))
    denominador = np.abs(E_r[0]) * np.abs(E_s[0]) + np.abs(E_r[1]) * np.abs(E_s[1]) * np.cos(np.angle(E_r[1]) - np.angle(E_s[1]))
    vartheta = np.arctan2(numerador, denominador)
    
    # Modulación de fase
    zeta = k * x - vartheta
    cos_term = np.cos(mu + zeta)
    I = Omega + Psi * cos_term
    info = [Omega, Psi, vartheta]

    # Normalización de la intensidad para escalar entre 0 y 255 (imagen en escala de grises)
    I_norm = (I - I.min()) / (I.max() - I.min()) * 255
    I_norm = np.uint8(I_norm)

    # Crear imagen de patrón de interferencia en escala de grises
    height = 500  # Altura de la imagen en píxeles
    pattern_image = np.tile(I_norm, (height, 1))

    # Aplicar un filtro Gaussiano para suavizar las franjas
    pattern_image = cv2.GaussianBlur(pattern_image, (5, 5), 0)
    
    if plot:
        # Graficar el interferograma
        plt.figure(figsize=(8, 6))
        plt.imshow(pattern_image, cmap='gray', aspect='auto', extent=[x.min(), x.max(), 0, height])
        plt.xlabel("x")
        plt.ylabel("Intensidad")
        plt.title("Interferograma")
        #plt.colorbar(label='Intensidad Normalizada')
        plt.show()
    
    return info ,I
