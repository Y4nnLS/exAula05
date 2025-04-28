def aplicar_harris(imagem, blockSize, ksize, k, nome_saida):
    harris = cv2.cornerHarris(imagem, blockSize=blockSize, ksize=ksize, k=k)
    harris = cv2.dilate(harris, None)
    imagem_harris = imagem.copy()
    imagem_harris[harris > 0.01 * harris.max()] = 255
    cv2.imwrite(nome_saida, imagem_harris)