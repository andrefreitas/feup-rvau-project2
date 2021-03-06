import cv2
import copy
import numpy as np
import glob


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
        if max_cos < 0.5:
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


# http://opencvpython.blogspot.pt/2012/06/contours-2-brotherhood.html
def get_corners(contour):
    return cv2.approxPolyDP(contour, 0.1*cv2.arcLength(contour, True), True)


def compare_contours(original_image_binary, original_image_contours, marker_image_binary):
    # Get marker corners and dimensions
    height, width = marker_image_binary.shape

    for contour in original_image_contours[1:]:

        # Get corners
        corners = get_corners(contour)
        a = corners[0]
        b = corners[1]
        c = corners[2]
        d = corners[3]

        # Initial src and dst corners
        src = np.array(corners, np.float32)
        dst = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]], np.float32)

        # Rotate and compare
        for i in range(4):

            # Rotate real image
            if i == 1:
                src = np.array([b, c, d, a], np.float32)
            elif i == 2:
                src = np.array([c, d, a, b], np.float32)
            elif i == 3:
                src = np.array([d, a, b, c], np.float32)

            # Compute homography
            matrix = cv2.getPerspectiveTransform(src, dst)

            # Generate frontal view from perspective
            image_frontal_perspective = cv2.warpPerspective(original_image_binary, matrix, (width, height))

            # Compare images
            similarity = compute_similarity(image_frontal_perspective, marker_image_binary)
            if similarity > 0.8:
                return src

    return False


# Compare 2 images of the same size
def compute_similarity(img1, img2):
    rest = cv2.bitwise_xor(img1, img2)
    height, width = img1.shape
    pixels = height * width
    diffs = 0
    for x in range(width):
        for y in range(height):
            if rest[y, x] != 0:
                diffs += 1

    similarity = (pixels - diffs) / float(pixels)

    return similarity


#http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
def calibrate_camera(img):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('images/chess_calibration/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        if ret:
            objpoints.append(objp)

            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist


#http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_pose/py_pose.html#pose-estimation
def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (255, 0, 0), 3)

    return img


def process_from_file():
    # Image sources
    original_image = cv2.imread('images/image201412170001.jpg')
    marker_image = cv2.imread('images/pattern_hiro_crop.jpg')
    sec_marker_image = cv2.imread('images/pattern_kanji_crop.jpg')

    # Calibrate camera
    ret, mtx, dist = calibrate_camera(original_image)

    # Binarize images
    sec_binary_marker_image = binarize_image(sec_marker_image)
    binary_image = binarize_image(original_image)
    binary_marker_image = binarize_image(marker_image)

    # Get contours
    image_rectangular_contours = get_rectangular_contours(binary_image)

    # Compute similarity and return src
    src = compare_contours(binary_image, image_rectangular_contours, binary_marker_image)
    if type(src) is bool:
        src = compare_contours(binary_image, image_rectangular_contours, sec_binary_marker_image)

    axis = np.float32([[0,0,0], [30,0,0], [30,30,0], [0,30,0],
                       [0,0,30],[30,0,30],[30,30,30],[0,30,30] ])

    if type(src) is not bool:
        # Convert src to the right type for solvePnP method
        src = src.reshape(4,2)
        objp = np.array([[0,0,0],[30,0,0],[30,30,0],[0,30,0]], dtype=np.float32)

        # Get rotation and translation vectors and Project Points
        ret, rvec, tvec = cv2.solvePnP(objp, src, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

        img = draw(original_image, imgpts)
        cv2.imshow('Virtual Reality FEUP', img)
    else:
        cv2.imshow('Virtual Reality FEUP', original_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_from_cam():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        marker_image = cv2.imread('images/pattern_hiro_crop.jpg')
        sec_marker_image = cv2.imread('images/pattern_kanji_crop.jpg')

        # Calibrate camera
        ret, mtx, dist = calibrate_camera(frame)

        # Binarize images
        sec_binary_marker_image = binarize_image(sec_marker_image)
        binary_marker_image = binarize_image(marker_image)
        binary_image = binarize_image(frame)

        image_rectangular_contours = get_rectangular_contours(binary_image)

        # Compute similarity and return src
        src = compare_contours(binary_image, image_rectangular_contours, binary_marker_image)
        if type(src) is bool:
            src = compare_contours(binary_image, image_rectangular_contours, sec_binary_marker_image)

        axis = np.float32([[0,0,0], [30,0,0], [30,30,0], [0,30,0],
                       [0,0,30],[30,0,30],[30,30,30],[0,30,30] ])

        # Check if similar or not before drawing
        if type(src) is not bool:
            src = src.reshape(4,2)
            objp = np.array([[0,0,0],[30,0,0],[30,30,0],[0,30,0]], dtype=np.float32)

            ret, rvec, tvec = cv2.solvePnP(objp, src, mtx, dist)
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

            img = draw(frame, imgpts)
            cv2.imshow('Virtual Reality FEUP', img)
        else:
            cv2.imshow('Virtual Reality FEUP', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()