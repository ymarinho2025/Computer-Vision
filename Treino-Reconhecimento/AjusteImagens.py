import os
import cv2
import shutil
from imutils import paths

pasta = "Fotos"
id_pessoa = 2


def ajustar_imagens(pasta_origem=pasta, pasta_destino="Fotos", id_usuario=id_pessoa):
    if not os.path.isdir(pasta_origem):
        raise FileNotFoundError(f"A pasta de origem '{pasta_origem}' não foi encontrada.")

    imagens_path = list(paths.list_images(pasta_origem))
    if not imagens_path:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em '{pasta_origem}'.")

    os.makedirs(pasta_destino, exist_ok=True)

    numero = 1
    for caminho_imagem in imagens_path:
        extensao = os.path.splitext(caminho_imagem)[1].lower() or ".jpg"
        nome_destino = f"pessoa.{id_usuario}.{numero}{extensao}"
        caminho_destino = os.path.join(pasta_destino, nome_destino)

        shutil.copy2(caminho_imagem, caminho_destino)

        img = cv2.imread(caminho_destino, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Erro ao abrir a imagem copiada: {caminho_destino}")
            numero += 1
            continue

        resized_image = cv2.resize(img, (220, 220))
        cv2.imwrite(caminho_destino, resized_image)
        print(nome_destino)
        numero += 1


ajustar_imagens()
