import cv2 as cv #Biblioteca para carregar a imagem
import sys #Utilizar funções do sistema
import numpy as np #Utilizar funções da biblioteca numpy
from scipy.ndimage import distance_transform_edt #Usado na função da metrica FOM
from skimage.metrics import structural_similarity as ssim #Importa a metrica SSIM
from skimage.metrics import mean_squared_error as mse #Importa a metrica MSE
from skimage.metrics import peak_signal_noise_ratio as psnr #Importa a metrica PSNR

imagem = cv.imread('imagens/lena.png') #Carregando imagem
imagem_ori = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY) #Converte em uma imagem monocromática (Escala de cinza)
imagem_ref = cv.imread('imagens/imagem_referencia_sobel.jpg', cv.COLOR_BGR2GRAY) #Carregando imagem de referencia 

def fom(bordas_ref, bordas_detectadas, alpha=1/9):
    # Certifique-se de que as bordas sejam binárias (0 e 1)
    bordas_ref_bin = (bordas_ref > 0).astype(np.float32)
    bordas_detectadas_bin = (bordas_detectadas > 0).astype(np.float32)

    # Calcular a transformada de distância na imagem de borda detectada
    dist_transform = distance_transform_edt(1 - bordas_detectadas_bin)

    # Número de pixels na borda de referência
    num_bordas_ref = np.sum(bordas_ref_bin)

    # Somatório de 1 / (1 + alpha * distância)
    fom_score = 0.0
    for i in range(bordas_ref.shape[0]):
        for j in range(bordas_ref.shape[1]):
            if bordas_ref_bin[i, j] > 0:  # Pixel na borda de referência (garantir valor escalar)
                fom_score += 1 / (1 + alpha * dist_transform[i, j])

    # Normalizar o FOM
    fom_score /= max(num_bordas_ref, np.sum(bordas_detectadas_bin))

    return fom_score * 100

def features_extraction(kernel):
    IP = kernel.size #Pegar o tamano do kernel 3x3 = 9
    histograma = cv.calcHist([kernel], [0], None, [256], [0,256]) #Calcular histograma
    probabilidade = histograma/IP #Calcular a probabilidade
    niveis = np.arange(256) #Niveis de cinza
    media = np.sum(niveis * probabilidade.flatten()) #Calcula a média
    desvioPD = np.sqrt(np.sum((niveis - media) ** 2 * probabilidade.flatten())) #Calcular desvio Padrão

    return desvioPD

def apply_kernel(imagem_ori):
    nova_imagem = np.zeros_like(imagem_ori, dtype=np.float32) #Criando nova imagem com a extração de caracteristicas

    #Pegando o desvio padrão dos pixels em um kernel 3x3 e adicionando na nova imagem(Extração de caracteristicas usando histograma)
    for i in range(1, imagem_ori.shape[0] - 1):
        for j in range(1, imagem_ori.shape[1] - 1):

            kernel = imagem_ori[i-1:i+2, j-1:j+2]
            kernel_desvioPD = features_extraction(kernel)
            nova_imagem[i, j] = kernel_desvioPD

    #Deixa a imagem no formato que possamos usar o OpenCV para visualizar 
    nova_imagem_show = cv.normalize(nova_imagem, None, 0, 255, cv.NORM_MINMAX)
    nova_imagem_show = nova_imagem_show.astype('uint8')

    return nova_imagem_show

#Gera a nova imagem com a extração de caracteristicas 
nova_imagem = apply_kernel(imagem_ori)

#Imagem original com o detector de bordas canny
imagem_ori_canny = cv.Canny(imagem_ori, 100, 200)

#Imagem com a extração de caracteristicas com o detector de bordas canny
imagem_mfi_canny = cv.Canny(nova_imagem, 60, 150)

fom_result_ori = fom(imagem_ref, imagem_ori_canny)
fom_result_mfi = fom(imagem_ref, imagem_mfi_canny)
ssim_result_ori = ssim(imagem_ori_canny, imagem_ref)
ssim_result_mfi = ssim(imagem_mfi_canny, imagem_ref)
psnr_result_ori = psnr(imagem_ori_canny, imagem_ref)
psnr_result_mfi = psnr(imagem_mfi_canny, imagem_ref)
mse_result_ori = mse(imagem_ori_canny, imagem_ref)
mse_result_mfi = mse(imagem_mfi_canny, imagem_ref)

#Inverte as cores da imagem
nova_imagem = cv.bitwise_not(nova_imagem)
imagem_mfi_canny = cv.bitwise_not(imagem_mfi_canny)
imagem_ori_canny = cv.bitwise_not(imagem_ori_canny)

#Mostrar metricas de avaliação
print(f"FOM_original: {fom_result_ori}")
print(f"FOM_mfi: {fom_result_mfi}")
print(f"SSIM_original: {ssim_result_ori}")
print(f"SSIM_mfi: {ssim_result_mfi}")
print(f"PSNR_ori: {psnr_result_ori}")
print(f"PSNR_mfi: {psnr_result_mfi}")
print(f"MSE_ori: {mse_result_ori}")
print(f"MSE_mfi: {mse_result_mfi}")

#Mantem a imagem aparecendo até digitar a tecla esc
while True:
    ch = cv.waitKey()
    if ch == 27:
        sys.exit()
    else:
        cv.imshow('Original',imagem_ori)
        cv.imshow('MFI', nova_imagem)
        cv.imshow('Original_canny', imagem_ori_canny)
        cv.imshow('MFI_canny', imagem_mfi_canny)