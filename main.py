import numpy as np
import cv2 as cv

def mainFunc():
    img = cv.imread('image2.jpg') #36
    output = img.copy()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(gray, 51)
    T, thresh_img = cv.threshold(blurred, 240, 255, cv.THRESH_BINARY)

    rows = blurred.shape[0]
    cv.imshow('img', thresh_img)
    cv.waitKey(0)
    circles = cv.HoughCircles(thresh_img, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=1, maxRadius=30)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(img, center, 1, (0, 100, 100), cv.FILLED, 3)
            # circle outline
            radius = i[2]
            cv.circle(img, center, radius, (255, 255, 255), cv.FILLED, 3)

    cv.imshow('img', img)
    cv.waitKey(0)

    # corners = cv.goodFeaturesToTrack(gray, 70, 0.5, 20) #0.5 works
    #
    # for corner in corners:
    #     x, y = corner.ravel()
    #     cv.circle(img, (int(x), int(y)), 5, (36, 255, 12), -1)
    #
    # print("count angle", len(corners))
    #
    # cv.imshow('thresh_img', img)
    # cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == '__main__':
    mainFunc()