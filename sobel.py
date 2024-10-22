import cv2
import numpy as np

# Carregar a imagem original
image = cv2.imread('imagens/lena.png', cv2.IMREAD_GRAYSCALE)

# Aplicar o filtro Sobel para detecção de bordas
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Bordas horizontais
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Bordas verticais

# Combinar as bordas
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Normalizar a imagem para o intervalo [0, 255]
sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))
#sobel_combined = cv2.bitwise_not(sobel_combined)

# Salvar a imagem de referência gerada com Sobel
cv2.imwrite('imagem_referencia_sobel_brain.jpg', sobel_combined)

print("Imagem de referência gerada com Sobel salva com sucesso.")
