import os
import cv2
import numpy as np

if not hasattr(cv2, "face"):
    raise RuntimeError(
        "O módulo cv2.face não está disponível. Instale opencv-contrib-python."
    )

lbph = cv2.face.LBPHFaceRecognizer_create()


def get_imagens_com_id(pasta_fotos="Fotos"):
    if not os.path.isdir(pasta_fotos):
        raise FileNotFoundError(
            f"A pasta '{pasta_fotos}' não foi encontrada. Capture ou copie as fotos antes de treinar."
        )

    caminhos = [
        os.path.join(pasta_fotos, f)
        for f in os.listdir(pasta_fotos)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not caminhos:
        raise FileNotFoundError("Nenhuma imagem encontrada na pasta 'Fotos'.")

    faces = []
    ids = []

    for caminho_imagem in caminhos:
        imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
        if imagem is None:
            print(f"Ignorando arquivo inválido: {caminho_imagem}")
            continue

        nome_arquivo = os.path.split(caminho_imagem)[-1]
        partes = nome_arquivo.split(".")

        # Formato esperado: pessoa.<id>.<numero>.jpg
        if len(partes) < 4:
            print(f"Nome de arquivo fora do padrão, ignorado: {nome_arquivo}")
            continue

        try:
            id_pessoa = int(partes[1])
        except ValueError:
            print(f"ID inválido no arquivo, ignorado: {nome_arquivo}")
            continue

        faces.append(imagem)
        ids.append(id_pessoa)

    if not faces or not ids:
        raise RuntimeError("Nenhuma face válida foi encontrada para treinamento.")

    return np.array(ids, dtype=np.int32), faces


ids, faces = get_imagens_com_id()

print("Treinando...")
lbph.train(faces, ids)
lbph.write("classificadorLBPH_V1.yml")
print("Treinamento concluído.")
