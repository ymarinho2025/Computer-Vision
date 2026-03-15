import cv2

carregaAlgoritmo = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

image = cv2.imread('fotos/image-4.png')

imagemCinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = carregaAlgoritmo.detectMultiScale(imagemCinza)

print(faces)

for(x, y, l, a) in faces:
    cv2.rectangle(image, (x, y), (x+l, y+a), (0, 255, 0), 2)

cv2.imshow('Rosto', image)
cv2.waitKey()