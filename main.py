import cv2
import copy

img = cv2.imread('images/image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
_ , contours, _ = cv2.findContours(copy.copy(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_contours = cv2.drawContours(copy.copy(img), contours, -1, (0, 255, 0), 3)

cv2.imshow('Imagem Original', img)
cv2.imshow('Imagem Binarizada', thresh)
cv2.imshow('Imagem com Contornos', img_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()