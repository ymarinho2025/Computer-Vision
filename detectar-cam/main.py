import cv2

webcam = cv2.VideoCapture(0)

classificarFace = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

classificarOlho = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

while True:
    camera, frame = webcam.read()
    # imagens cinzas são mais fáceis de processar, por isso convertemos a imagem para cinza
    cameraCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ajustar parametros dependendo da imagem
    detectaFace = classificarFace.detectMultiScale(cameraCinza)
    
    # (x para horizontal e y para vertical) (l para largura, a para altura)
    for(x, y, l, a) in detectaFace:
        espFace = cv2.rectangle(frame, (x, y), (x+l, y+a), (255, 0, 0), 2)
    
        localOlho = frame[y:y + a, x:x + l]
        localOlhoCinza = cv2.cvtColor(localOlho, cv2.COLOR_BGR2GRAY)
        detectaOlho = classificarOlho.detectMultiScale(localOlhoCinza, scaleFactor=1.2, minNeighbors=4)
    
        # fazer igual anteriormente mas com 'o' para olho
        for(ox, oy, ol, oa) in detectaOlho:
            cv2.rectangle(localOlho, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)
        
    cv2.imshow('Video WebCam', frame)
    
    if cv2.waitKey(1) == ord('q'):
        print('Fechando a webcam')
        break
    
webcam.release()
cv2.destroyAllWindows()