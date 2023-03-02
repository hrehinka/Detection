from PIL import Image
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
import math
import collections


def ORB_descriptor():
    img1 = cv2.imread('D:/Diploma thesis/Maps/UPJS/Dvere/door2.png',cv2.IMREAD_GRAYSCALE)          # queryImage

    img2 = cv2.imread('D:/Diploma thesis/Maps/UPJS/IIa_AP3601_separated.png',cv2.IMREAD_GRAYSCALE) # trainImage
    img2 = cv2.resize(img2, (650, int(650 * img2.shape[0] / img2.shape[1])))
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()

# ORB_descriptor()

def SIFT_descriptor():
    img1 = cv2.imread('D:/Diploma thesis/Maps/UPJS/Dvere/door1.png', cv2.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv2.imread('D:/Diploma thesis/Maps/UPJS/IIa_AP3601_separated.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.resize(img2, (650, int(650 * img2.shape[0] / img2.shape[1])))# trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()


def descriptor():
    query_img = cv2.imread('D:/Diploma thesis/Maps/UPJS/Dvere/door1.png')
    train_img = cv2.imread('D:/Diploma thesis/Maps/UPJS/IIa_AP3601_separated.png')
    train_img = cv2.resize(train_img, (650, int(650 * train_img.shape[0] / train_img.shape[1])))  # trainImage

    # Convert it to grayscale
    query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create()

    # Now detect the keypoints and compute
    # the descriptors for the query image
    # and train image
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

    # Initialize the Matcher for matching
    # the keypoints and then match the
    # keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.match(queryDescriptors, trainDescriptors)

    # draw the matches to the final image
    # containing both the images the drawMatches()
    # function takes both images and keypoints
    # and outputs the matched query image with
    # its train image
    final_img = cv2.drawMatches(query_img, queryKeypoints,
                                train_img, trainKeypoints, matches[:20], None)

    # final_img = cv2.resize(final_img, (1000, 650))

    # Show the final image
    cv2.imshow("Matches", final_img)
    cv2.waitKey(3000)

# descriptor()

# def generalized_hough():
#     img = cv2.imread('D:/Diploma thesis/Maps/UPJS/Dvere/door1.png')
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     template = cv2.imread('D:/Diploma thesis/Maps/UPJS/IIa_AP3601_separated.png')
#     height, width = template.shape[:2]
#
#     edges = cv2.Canny(template, 200, 250)
#     ght = cv2.createGeneralizedHoughGuil()
#     ght.setTemplate(edges)
#
#     ght.setMinDist(100)
#     ght.setMinAngle(0)
#     ght.setMaxAngle(360)
#     ght.setAngleStep(1)
#     ght.setLevels(360)
#     ght.setMinScale(1)
#     ght.setMaxScale(1.3)
#     ght.setScaleStep(0.05)
#     ght.setAngleThresh(100)
#     ght.setScaleThresh(100)
#     ght.setPosThresh(100)
#     ght.setAngleEpsilon(1)
#     ght.setLevels(360)
#     ght.setXi(90)
#
#     positions = ght.detect(img_gray)[0][0]
#
#     for position in positions:
#         center_col = int(position[0])
#         center_row = int(position[1])
#         scale = position[2]
#         angle = int(position[3])
#
#         found_height = int(height * scale)
#         found_width = int(width * scale)
#
#         rectangle = ((center_col, center_row),
#                      (found_width, found_height),
#                      angle)
#
#         box = cv2.boxPoints(rectangle)
#         box = np.int0(box)
#         cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
#
#         for i in range(-2, 3):
#             for j in range(-2, 3):
#                 img[center_row + i, center_col + j] = 0, 0, 255
#
#     cv2.imshow("Matches", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#

def generalized_hough():
    image = cv2.imread('D:/Diploma thesis/Maps/UPJS/Dvere/door1.png')
    image = cv2.resize(image, (650, int(650 * image.shape[0] / image.shape[1])))  # trainImage
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ght = cv2.createGeneralizedHoughGuil()
    template = cv2.imread('D:/Diploma thesis/Maps/UPJS/IIa_AP3601_separated.png')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w = 133
    h = 128

    ght.setMinDist(10)
    ght.setDp(3)
    ght.setMaxBufferSize(1000)
    ght.setMinAngle(0)
    ght.setMaxAngle(360)
    ght.setAngleStep(1)
    ght.setLevels(360)
    ght.setMinScale(1)
    ght.setMaxScale(2.0)
    ght.setScaleStep(0.05)
    ght.setAngleThresh(1500)
    ght.setScaleThresh(50)
    ght.setPosThresh(10)
    ght.setCannyLowThresh(30)
    ght.setCannyHighThresh(110)
    ght.setAngleEpsilon(1)
    ght.setLevels(360)
    ght.setXi(90)

    ght.setTemplate(template)
    guil = ght.detect(template)


    positions, votes = houghTransformer.detect(template)

    listOfPosition = positions[0]

    outputImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y, scale, orientation in listOfPosition:
        halfHeight = 133 / 2. * scale
        halfWidth = 128 / 2. * scale
        p1 = (int(x - halfWidth), int(y - halfHeight))
        p2 = (int(x + halfWidth), int(y + halfHeight))
        print("x = {}, y = {}, scale = {}, orientation = {}, p1 = {}, p2 = {}".format(x, y, scale, orientation, p1, p2))
        cv2.rectangle(outputImage, p1, p2, (0, 0, 255))

    cv2.imshow("Matches", outputImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

generalized_hough()
