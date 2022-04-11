import numpy as np
import cv2 as cv


def mainFunc():
    img = cv.imread('wheelOnBlack.jpg')  # 36

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(gray, 3)
    kernel = np.ones((2, 2), 'uint8')
    # T, thresh_img = cv.threshold(blurred, 240, 255, cv.THRESH_BINARY)

    erode_img = cv.erode(gray, kernel, cv.BORDER_REFLECT, iterations=1)

    circles = cv.HoughCircles(erode_img , cv.HOUGH_GRADIENT, 1.4, 25,param1=350,param2=55,maxRadius=75,minRadius=25)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[1] + 10, i[0] + 5)
            # circle center
            cv.circle(erode_img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2] + 245
            cv.circle(erode_img, center, radius, (0, 0, 0), cv.FILLED, 3)

    corners = cv.goodFeaturesToTrack(erode_img, 70, 0.1, 20)  #0.5 works

    for corner in corners:
        x, y = corner.ravel()
        cv.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    print("count angle", len(corners))

    cv.imshow('thresh_img', img)
    cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == '__main__':
    mainFunc()
