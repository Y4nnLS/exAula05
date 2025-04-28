import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Lista de imagens de teste
imagens_entrada = [
    "imagem1note.jpg",
    "imagem2sala.jpg",
    "imagem3cubo.png"
]

shi_tomasi_configs = [
    {"maxCorners": 100, "qualityLevel": 0.01, "minDistance": 10},  # Detecta tudo
    {"maxCorners": 50, "qualityLevel": 0.05, "minDistance": 20},   # Médio
    {"maxCorners": 20, "qualityLevel": 0.1, "minDistance": 30},    # Só cantos fortes
]

# Pasta para salvar os resultados
output_dir = "resultados_tarefa_pratica_v2/shi_tomasi"
os.makedirs(output_dir, exist_ok=True)

def carregar_imagem(caminho):
    imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        raise FileNotFoundError(f"Imagem '{caminho}' não encontrada.")
    return imagem

def aplicar_shi_tomasi(imagem, maxCorners, qualityLevel, minDistance, nome_saida):
    corners = cv2.goodFeaturesToTrack(
        imagem,
        maxCorners=maxCorners,
        qualityLevel=qualityLevel,
        minDistance=minDistance
    )
    imagem_cantos = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        for i in corners:
            x, y = i.ravel()
            cv2.circle(imagem_cantos, (int(x), int(y)), 4, (0, 255, 0), -1)
    cv2.imwrite(nome_saida, imagem_cantos)

def aplicar_sift(imagem, nome_saida):
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(imagem, None)
    imagem_sift = cv2.drawKeypoints(imagem, keypoints, None)
    cv2.imwrite(nome_saida, imagem_sift)

def aplicar_orb(imagem, nome_saida):
    orb = cv2.ORB_create()
    keypoints, _ = orb.detectAndCompute(imagem, None)
    imagem_orb = cv2.drawKeypoints(imagem, keypoints, None, color=(0, 255, 0))
    cv2.imwrite(nome_saida, imagem_orb)

def processar_imagem(imagem_path):
    imagem_nome = os.path.splitext(os.path.basename(imagem_path))[0]
    imagem = carregar_imagem(imagem_path)

    # Shi-Tomasi para cada configuração
    for idx, config in enumerate(shi_tomasi_configs):
        nome_saida = os.path.join(output_dir, f"{imagem_nome}_shi_tomasi_cfg{idx}.jpg")
        aplicar_shi_tomasi(imagem, **config, nome_saida=nome_saida)

    # SIFT
    nome_sift = os.path.join(output_dir, f"{imagem_nome}_sift.jpg")
    aplicar_sift(imagem, nome_sift)

    # ORB
    nome_orb = os.path.join(output_dir, f"{imagem_nome}_orb.jpg")
    aplicar_orb(imagem, nome_orb)

    print(f"[OK] Processado {imagem_nome}")

def mostrar_resultados_shi_tomasi_em_blocos():
    imagens_shi = []

    # Coleta todos os caminhos das imagens Shi-Tomasi
    for imagem_path in imagens_entrada:
        imagem_nome = os.path.splitext(os.path.basename(imagem_path))[0]
        for idx in range(3):
            caminho_imagem = os.path.join(output_dir, f"{imagem_nome}_shi_tomasi_cfg{idx}.jpg")
            imagens_shi.append((imagem_nome, idx, caminho_imagem))

    # Mostra de 3 em 3
    for i in range(0, len(imagens_shi), 3):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Detecção de Cantos - Shi-Tomasi (Bloco)', fontsize=16)

        bloco = imagens_shi[i:i+3]
        for ax, (nome, cfg_idx, caminho) in zip(axes, bloco):
            imagem = cv2.imread(caminho)
            imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
            ax.imshow(imagem_rgb)
            ax.set_title(f"{nome} - cfg{cfg_idx}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()  # Espera você fechar a janela para mostrar o próximo bloco

# Execução principal
if __name__ == "__main__":
    # Primeiro, processa todas as imagens
    for imagem_path in imagens_entrada:
        processar_imagem(imagem_path)

    # Depois, exibe as imagens em blocos de 3
    mostrar_resultados_shi_tomasi_em_blocos()

    print("\nTodos os resultados foram salvos em 'resultados_tarefa_pratica_v2'.")
