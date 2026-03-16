import os
import cv2

# Carrega o classificador Haar Cascade usando o diretório interno do OpenCV.
classificador = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if classificador.empty():
    raise RuntimeError("Não foi possível carregar o classificador de faces do OpenCV.")

webcamera = cv2.VideoCapture(0)
if not webcamera.isOpened():
    raise RuntimeError("Não foi possível acessar a webcam.")

# Garante que a pasta de saída exista.
os.makedirs("Fotos", exist_ok=True)

amostra = 1
numero_amostras = 25
id_pessoa = input("Digite seu identificador numérico: ").strip()

if not id_pessoa.isdigit():
    webcamera.release()
    cv2.destroyAllWindows()
    raise ValueError("O identificador deve ser numérico para funcionar com o treinamento LBPH.")

print("Pressione 'c' para capturar a foto da face detectada.")
print("Pressione 'q' para encerrar.")

while True:
    conectado, imagem = webcamera.read()
    if not conectado:
        print("Erro ao capturar imagem da webcam.")
        break

    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces_detectadas = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.2, minNeighbors=5, minSize=(150, 150))

    for (x, y, l, a) in faces_detectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

    cv2.putText(imagem, f"Fotos: {amostra - 1}/{numero_amostras}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,)
    cv2.putText(imagem, "C = capturar | Q = sair", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,)

    cv2.imshow("Captura de Face", imagem)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        print("Captura encerrada pelo usuário.")
        break

    if tecla == ord("c"):
        if len(faces_detectadas) == 0:
            print("Nenhuma face detectada no momento.")
        else:
            # Captura somente a primeira face detectada no frame.
            x, y, l, a = faces_detectadas[0]
            imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (220, 220))
            caminho = os.path.join("Fotos", f"pessoa.{id_pessoa}.{amostra}.jpg")
            sucesso = cv2.imwrite(caminho, imagem_face)
            if sucesso:
                print(f"[Foto {amostra} capturada com sucesso]")
                amostra += 1
            else:
                print(f"Erro ao salvar a foto {amostra}.")

    if amostra > numero_amostras:
        break

print("Faces capturadas com sucesso.")
webcamera.release()
cv2.destroyAllWindows()