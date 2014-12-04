import cv2
import copy
import numpy as np
import uuid

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


# Get marker borders
def get_marker_borders(marker_image):
    contours = get_rectangular_contours(marker_image)
    contours_areas_sorted = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    return contours_areas_sorted[0]


def convert_to_list(ndarray_structure):
    return map(lambda x: x[0], ndarray_structure.tolist())


def get_dimensions(points):
    x_min = points[0][0]
    x_max = points[0][0]
    y_min = points[0][1]
    y_max = points[0][1]

    for point in points:
        x = point[0]
        y = point[1]
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return width, height


# http://opencvpython.blogspot.pt/2012/06/contours-2-brotherhood.html
def get_corners(contour):
    return cv2.approxPolyDP(contour, 0.1*cv2.arcLength(contour, True), True)



def compare_contours(original_image_binary, original_image_contours, marker_image_binary, marker_borders):
    # Get marker corners and dimensions
    marker_borders = convert_to_list(marker_borders)
    width, height = get_dimensions(marker_borders)

    for contour in original_image_contours[1:]:
        # Get corners
        corners = get_corners(contour)

        # Compute transformation matrix
        src = np.array(corners, np.float32)
        dst = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]], np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)

        # Generate frontal view from perspective
        image_frontal_perspective = cv2.warpPerspective(original_image_binary, matrix, (width, height))
        cv2.imshow(str(uuid.uuid4()), image_frontal_perspective)

        # Compare
        if images_are_equal(image_frontal_perspective, marker_image_binary):
            return corners

    return False


def images_are_equal(img1, img2):
    #TODO
    return True

def process_from_file():
    # Image sources
    original_image = cv2.imread('images/image.jpg')
    marker_image = cv2.imread('images/pattern_hiro_crop.jpg')

    # Binarize images
    binary_image = binarize_image(original_image)
    binary_marker_image = binarize_image(marker_image)

    # Get contours
    image_rectangular_contours = get_rectangular_contours(binary_image)
    marker_borders = get_marker_borders(binary_marker_image)

    # Compare contours
    corners = compare_contours(binary_image, image_rectangular_contours, binary_marker_image, marker_borders)

    # Compute homography
    img_contours = cv2.drawContours(copy.copy(original_image), image_rectangular_contours, -1, (0, 255, 0), 3)
    #img_contours = cv2.drawContours(img_contours, [corners], -1, (0, 0, 255), 3)

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