import cv2

# Função responsável por calcular a similaridade entre duas imagens
# utilizando o algoritmo ORB para detecção de características.
def orb_sim(img1, img2):
    
    # Cria o detector ORB (Oriented FAST and Rotated BRIEF)
    # Esse algoritmo identifica pontos-chave (keypoints) nas imagens
    orb = cv2.ORB_create()
    
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img1, None)
    
    # Cria o objeto BFMatcher (Brute Force Matcher)
    # Esse método compara cada descritor da imagem A com todos da imagem B
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    matches = bf.match(desc_a, desc_b)
    
    # Filtra apenas as correspondências consideradas "boas"
    # Quanto menor a distância, maior a similaridade entre os descritores
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)

img00 = cv2.imread('sign1.jpg', 0)
img01 = cv2.imread('sign2.jpg', 0)

img00Res = cv2.resize(img00, (1200, 600))
img01Res = cv2.resize(img01, (1200, 600))

orb_similarity = orb_sim(img00Res, img01Res)
print("0 Percentual de Similaridade Utilizando o ORB é: ", orb_similarity)