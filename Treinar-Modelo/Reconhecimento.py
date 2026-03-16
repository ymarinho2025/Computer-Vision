import os
import cv2

# Mapeamento de IDs para nomes.
NOMES = {
    1: "Yuri",
    2: "Isa",
}

classificador = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if classificador.empty():
    raise RuntimeError("Não foi possível carregar o classificador de faces do OpenCV.")

if not hasattr(cv2, "face"):
    raise RuntimeError(
        "O módulo cv2.face não está disponível. Instale opencv-contrib-python."
    )

modelo_path = "classificadorLBPH_V1.yml"
if not os.path.exists(modelo_path):
    raise FileNotFoundError(
        f"Arquivo de modelo não encontrado: {modelo_path}. Execute o treinamento primeiro."
    )

reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read(modelo_path)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Não foi possível acessar a webcam.")

while True:
    conectado, imagem = camera.read()
    if not conectado:
        print("Erro ao capturar imagem da webcam.")
        break

    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces_detectadas = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),)

    for (x, y, l, a) in faces_detectadas:
        imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (220, 220))
        id_previsto, confianca = reconhecedor.predict(imagem_face)

        # No LBPH, quanto menor a confiança, melhor o reconhecimento.
        if confianca <= 55:
            nome = NOMES.get(id_previsto, f"ID {id_previsto}")
            texto = f"{nome} ({confianca:.1f})"
            cor = (0, 255, 0)
        else:
            texto = f"Desconhecido ({confianca:.1f})"
            cor = (0, 0, 255)

        cv2.rectangle(imagem, (x, y), (x + l, y + a), cor, 2)
        cv2.putText(imagem, texto, (x, y - 10 if y > 20 else y + a + 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, cor, 2)
    cv2.putText(imagem, "Pressione Q para sair", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Reconhecimento Facial", imagem)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
