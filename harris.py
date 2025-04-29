import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


imagens_entrada = [
    "imagem1note.jpg",
    "imagem2sala.jpg",
    "imagem3cubo.png"
]

harris_configs = [
    {"blockSize": 2, "ksize": 3, "k": 0.04},   
    {"blockSize": 5, "ksize": 5, "k": 0.05},   
    {"blockSize": 9, "ksize": 7, "k": 0.06},   
]


output_dir = "resultados_tarefa_pratica_v2/harris"
os.makedirs(output_dir, exist_ok=True)

def carregar_imagem(caminho):
    imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        raise FileNotFoundError(f"Imagem '{caminho}' não encontrada.")
    return imagem

def aplicar_harris(imagem, blockSize, ksize, k, nome_saida):
    harris = cv2.cornerHarris(imagem, blockSize=blockSize, ksize=ksize, k=k)
    harris = cv2.dilate(harris, None)
    imagem_harris = imagem.copy()
    imagem_harris[harris > 0.01 * harris.max()] = 255
    cv2.imwrite(nome_saida, imagem_harris)

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

    
    for idx, config in enumerate(harris_configs):
        nome_saida = os.path.join(output_dir, f"{imagem_nome}_harris_cfg{idx}.jpg")
        aplicar_harris(imagem, **config, nome_saida=nome_saida)

    
    nome_sift = os.path.join(output_dir, f"{imagem_nome}_sift.jpg")
    aplicar_sift(imagem, nome_sift)

    
    nome_orb = os.path.join(output_dir, f"{imagem_nome}_orb.jpg")
    aplicar_orb(imagem, nome_orb)

    print(f"[OK] Processado {imagem_nome}")

def mostrar_resultados_harris_em_blocos():
    imagens_harris = []

    
    for imagem_path in imagens_entrada:
        imagem_nome = os.path.splitext(os.path.basename(imagem_path))[0]
        for idx in range(3):
            caminho_imagem = os.path.join(output_dir, f"{imagem_nome}_harris_cfg{idx}.jpg")
            imagens_harris.append((imagem_nome, idx, caminho_imagem))

    
    for i in range(0, len(imagens_harris), 3):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Detecção de Cantos - Harris (Bloco)', fontsize=16)

        bloco = imagens_harris[i:i+3]
        for ax, (nome, cfg_idx, caminho) in zip(axes, bloco):
            imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
            ax.imshow(imagem, cmap='gray')
            ax.set_title(f"{nome} - cfg{cfg_idx}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()  


if __name__ == "__main__":
    
    for imagem_path in imagens_entrada:
        processar_imagem(imagem_path)

    
    mostrar_resultados_harris_em_blocos()

    print("\n✅ Todos os resultados foram salvos em 'resultados_tarefa_pratica_v2'.")
