import cv2

# carregar direto da lib do opencv, não precisa baixar o xml
carregaAlgoritmo = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

image = cv2.imread('fotos/image-5.jpg')

# imagens cinzas são mais fáceis de processar, por isso convertemos a imagem para cinza
imagemCinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ajustar parametros dependendo da imagem
faces = carregaAlgoritmo.detectMultiScale(imagemCinza, scaleFactor=1.07, minNeighbors=3)

print(faces)

# (x para horizontal e y para vertical) (l para largura, a para altura)
for(x, y, l, a) in faces:
    cv2.rectangle(image, (x, y), (x+l, y+a), (0, 255, 0), 2)

cv2.imshow('Rosto', image)
cv2.waitKey()