import cv2

# Mapeamento de IDs para nomes.
nomes = {
    1: "Yuri",
    2: "Isa",
}

webcam = cv2.VideoCapture(0)

classificarFace = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

identificador = cv2.face.LBPHFaceRecognizer_create()
identificador.read('classificadorTreinado.yml')

classificarOlho = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

while True:
    camera, frame = webcam.read()
    # imagens cinzas são mais fáceis de processar, por isso convertemos a imagem para cinza
    cameraCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ajustar parametros dependendo da imagem
    detectaFace = classificarFace.detectMultiScale(cameraCinza, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))
    
    # (x para horizontal e y para vertical) (l para largura, a para altura)
    for(x, y, l, a) in detectaFace:
        imagemFace = cv2.resize(cameraCinza[y:y + a, x:x + l], (220, 220))
        espFace = cv2.rectangle(frame, (x, y), (x+l, y+a), (255, 0, 0), 2)
        
        id, acuracia = identificador.predict(imagemFace)
            
        # No LBPH, quanto menor a confiança, melhor o reconhecimento.
        if acuracia <= 55:
            nome = nomes.get(id, f"ID {id}")
            texto = f"{nome} ({acuracia:.1f})"
            cor = (0, 255, 0)
        else:
            texto = f"Desconhecido ({acuracia:.1f})"
            cor = (0, 0, 255)
        
        contador = str(detectaFace.shape[0])
        
        localOlho = frame[y:y + a, x:x + l]
        localOlhoCinza = cv2.cvtColor(localOlho, cv2.COLOR_BGR2GRAY)
        detectaOlho = classificarOlho.detectMultiScale(localOlhoCinza)
    
        # fazer igual anteriormente mas com 'o' para olho
        for(ox, oy, ol, oa) in detectaOlho:
            cv2.rectangle(localOlho, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)
            
        cv2.putText(frame, texto, (x - 7, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.putText(frame, 'Quantidade de faces detectadas:' + contador, (0, frame.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        
    cv2.imshow('Video WebCam', frame)
    
    if cv2.waitKey(1) == ord('q'):
        print('Fechando a webcam')
        break
    
webcam.release()
cv2.destroyAllWindows()