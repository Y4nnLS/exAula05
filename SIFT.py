import cv2
import numpy as np
import os


output_dir = "resultados_tarefa_pratica_v2/SIFT"
os.makedirs(output_dir, exist_ok=True)

def carregar_imagem(caminho):
    imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        raise FileNotFoundError(f"Imagem '{caminho}' não encontrada.")
    return imagem

def aplicar_sift(imagem, nome_saida="sift_imagem2sala_result.jpg"):
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(imagem, None)

    imagem_colorida = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

    for kp in keypoints:
        x, y = map(int, kp.pt)  
        cv2.drawMarker(
            imagem_colorida,
            (x, y),
            (0, 255, 0),                
            markerType=cv2.MARKER_CROSS,
            markerSize=5,
            thickness=1,
            line_type=cv2.LINE_AA
        )

    
    caminho = os.path.join(output_dir, nome_saida)
    cv2.imwrite(caminho, imagem_colorida)
    print(f"[OK] Resultado SIFT salvo em {caminho}")

    escala = 2.0
    largura = int(imagem_colorida.shape[1] * escala)
    altura = int(imagem_colorida.shape[0] * escala)
    imagem_redimensionada = cv2.resize(imagem_colorida, (largura, altura), interpolation=cv2.INTER_LINEAR)

    
    cv2.imshow('Resultado SIFT - imagem2sala', imagem_redimensionada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imagem_colorida


if __name__ == "__main__":
    caminho_imagem = "imagem2sala.jpg"  
    imagem = carregar_imagem(caminho_imagem)

    aplicar_sift(imagem, nome_saida="sift_imagem2sala_result.jpg")
