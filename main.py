import numpy as np
import cv2 as cv
import requests


def mainFunc():
    url = "https://github.com/denis14082000/pyOpenCv/blob/master/wheelOnBlack.jpg?raw=true"

    resp = requests.get(url, stream=True).raw
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv.imdecode(img, cv.IMREAD_COLOR)

    blurred = cv.medianBlur(img, 3)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), 'uint8')

    erode_img = cv.erode(gray, kernel, cv.BORDER_REFLECT, iterations=1)

    circles = cv.HoughCircles(erode_img, cv.HOUGH_GRADIENT, 1.4, 25, param1=350, param2=55, maxRadius=75, minRadius=25)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[1] + 10, i[0] + 5)
            # circle center
            cv.circle(erode_img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2] + 245
            cv.circle(erode_img, center, radius, (0, 0, 0), cv.FILLED, 3)

    corners = cv.goodFeaturesToTrack(erode_img, 70, 0.1, 20)  # 0.5 works

    for corner in corners:
        x, y = corner.ravel()
        cv.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    print("count angle", int(len(corners) / 2))

    cv.imshow('thresh_img', img)
    cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == '__main__':
    mainFunc()
