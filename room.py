import cv2
import numpy as np

def image_preprocess(gray_img):
    ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('removed', thresh)
    cv2.waitKey(0)

    thresh = 127
    im_bw = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow('removed', im_bw)
    cv2.waitKey(0)



def find_rooms(img, noise_removal_threshold=25, corners_threshold=0.1,
               room_closing_max_length=50, gap_in_wall_threshold=500):
    """

    :param img: grey scale image of rooms, already eroded and doors removed etc.
    :param noise_removal_threshold: Minimal area of blobs to be kept.
    :param corners_threshold: Threshold to allow corners. Higher removes more of the house.
    :param room_closing_max_length: Maximum line length to add to close off open doors.
    :param gap_in_wall_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
    :return: rooms: list of numpy arrays containing boolean masks for each detected room
             colored_house: A colored version of the input image, where each room has a random color.
    """
    # assert 0 <= corners_threshold <= 1
    # # Remove noise left from door removal

    img[img < 128] = 0
    img[img > 128] = 255
    contours, hierarchy = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > noise_removal_threshold:
            cv2.fillPoly(mask, [contour], 255)

    img = ~mask


    # Detect corners (you can play with the parameters here)
    dst = cv2.cornerHarris(img,2,3,0.04)

    dst = cv2.dilate(dst,None)

    corners = dst > corners_threshold * dst.max()


    # Draw lines to close the rooms off by adding a line between corners on the same x or y coordinate
    # This gets some false positives.
    # You could try to disallow drawing through other existing lines for example.
    for y,row in enumerate(corners):
        x_same_y = np.argwhere(row)
        for x1, x2 in zip(x_same_y[:-1], x_same_y[1:]):

            if x2[0] - x1[0] < room_closing_max_length:
                color = 0

                cv2.line(img, (int(x1), int(y)), (int(x2), int(y)), color, 1)

    for x,col in enumerate(corners.T):
        y_same_x = np.argwhere(col)
        for y1, y2 in zip(y_same_x[:-1], y_same_x[1:]):
            if y2[0] - y1[0] < room_closing_max_length:
                color = 0
                cv2.line(img, (int(x), int(y1)), (int(x), int(y2)), color, 2)


    # Mark the outside of the house as black
    contours, hierarchy = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [biggest_contour], 255)
    img[mask == 0] = 0

    # Find the connected components in the house
    ret, labels = cv2.connectedComponents(img)
    print(ret)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    unique = np.unique(labels)
    rooms = []
    for label in unique:
        # print(label)
        component = labels == label
        if img[component].sum() == 0 or np.count_nonzero(component) < gap_in_wall_threshold:
            color = 0
        else:

            rooms.append(component)
            color = np.random.randint(0, 255, size=3)
        img[component] = color

    # print(rooms)
    return rooms, img



#Read gray image
img = cv2.imread("D:/Diploma thesis/Maps/UPJS/IIa_AP3601_separated.png")
img = cv2.resize(img, (650, int(650 * img.shape[0] / img.shape[1])))
cv2.imshow('input', img)
cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', img)
cv2.waitKey(0)
image_preprocess(img)

# rooms, colored_house = find_rooms(img.copy())
#
# cv2.imwrite('D:/Diploma thesis/Maps/UPJS/Room_detection_results/50&500.jpg', colored_house)
# cv2.imshow('result', colored_house)
# cv2.waitKey()
# cv2.destroyAllWindows()