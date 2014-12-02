import cv2
import copy
import numpy as np


# Returns the cosine of the Angle
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


# Check if the contour is rectangular
def contour_is_rectangular(contour):
    cnt_len = cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, 0.02*cnt_len, True)
    if len(contour) == 4 and cv2.contourArea(contour) > 1000 and cv2.isContourConvex(contour):
        contour = contour.reshape(-1, 2)
        max_cos = np.max([angle_cos( contour[i], contour[(i+1) % 4], contour[(i+2) % 4] ) for i in xrange(4)])
        if max_cos < 0.1:
            return True
    return False


# Get rectangular contours
def get_rectangular_contours(binary_image):
    _, contours, _ = cv2.findContours(copy.copy(binary_image), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if contour_is_rectangular(contour)]


# Binarize an image
def binarize_image(original_image):
    gray_scale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_scale_image, 127, 255, cv2.THRESH_BINARY)
    return binary_image


def process_from_file():
    original_image = cv2.imread('images/image.jpg')
    binary_image = binarize_image(original_image)
    rectangular_contours = get_rectangular_contours(binary_image)
    img_contours = cv2.drawContours(copy.copy(original_image), rectangular_contours, -1, (0, 255, 0), 3)
    cv2.imshow('Imagem com bordos quadrados', img_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_from_cam():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        binary_image = binarize_image(frame)
        rectangular_contours = get_rectangular_contours(binary_image)
        img_contours = cv2.drawContours(copy.copy(frame), rectangular_contours, -1, (0, 255, 0), 3)
        cv2.imshow('Imagem com bordos quadrados', img_contours)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()