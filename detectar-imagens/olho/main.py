import cv2

# carregar direto da lib do opencv, não precisa baixar o xml
CarregaFace = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

CarregaOlho = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

image = cv2.imread('fotos/image-1.jpg')

# imagens cinzas são mais fáceis de processar, por isso convertemos a imagem para cinza
imagemCinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ajustar parametros dependendo da imagem
detectaFace = CarregaFace.detectMultiScale(imagemCinza)

# (x para horizontal e y para vertical) (l para largura, a para altura)
for(x, y, l, a) in detectaFace:
    espFace = cv2.rectangle(image, (x, y), (x+l, y+a), (0, 255, 0), 2)
    
    
    localOlho = image[y:y + a, x:x + l]
    localOlhoCinza = cv2.cvtColor(localOlho, cv2.COLOR_BGR2GRAY)
    detectaOlho = CarregaOlho.detectMultiScale(localOlhoCinza, scaleFactor=1.08, minNeighbors=4)
    
    # fazer igual anteriormente mas com 'o' para olho
    for(ox, oy, ol, oa) in detectaOlho:
        cv2.rectangle(localOlho, (ox, oy), (ox + ol, oy + oa), (255, 0, 0), 2)
        
cv2.imshow('Face e Olhos', image)
cv2.waitKey()