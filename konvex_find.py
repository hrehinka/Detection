
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


def create_final_list_of_walls(df):
    list_of_lines = []
    for k in range(0, len(df)):
        x_s = df.at[k, 'Ax']
        y_s = df.at[k, 'Ay']
        start_point = (int(x_s), int(y_s))

        x_e = df.at[k, 'Bx']
        y_e = df.at[k, 'By']
        end_point = (int(x_e), int(y_e))

        list_of_lines.append((start_point, end_point))

    return list_of_lines

def LSD_computing_info_lines(lsdresults, df_lines):
    for i in range(0, len(lsdresults)):
        # take data from results of lsd for given picture
        res = lsdresults[i]
        res_lines = res[0].reshape(-1, 4)
        # print(res_lines.shape)
        res_width = res[1].reshape(-1)
        width_col = res_width.reshape(-1, 1)

        full_lines = np.column_stack((res_lines, res_width))

        # create a data frame from full_lines, with defines columns
        df_line = pd.DataFrame(full_lines, columns=['Ax', 'Ay', 'Bx', 'By', 'width'], dtype=np.float32)

        # calculation of length and angle of lines and adding label based on number of picture in queue
        df_line['length'] = np.sqrt((df_line['Ax'] - df_line['Bx']) ** 2 + (df_line['Ay'] - df_line['By']) ** 2)
        df_line['angle'] = np.abs(
            np.arctan2(df_line['Ay'] - df_line['By'], df_line['Ax'] - df_line['Bx']) * 180 / np.pi)



    return df_line

def LSD(image, lsd):
    # take photo from a folder and convert to greyscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # create an empty photo for visualisation of detection
    empty = np.zeros_like(img)

    # use LSD detector
    lines = lsd.detect(img)[0]

    # draw detected lines on empty image
    drawn_img = lsd.drawSegments(empty, lines)
    cv2.imwrite('D:/Diploma thesis/Maps/UPJS/LSDresult/01.jpg',
                drawn_img)


def LSD_lines_info(image_black, lsdresults, lsd):
    image_black = cv2.cvtColor(image_black, cv2.COLOR_BGR2GRAY)
    lsdresult = lsd.detect(image_black)
    lsdresults.append(lsdresult)


# DEFINING ROOMS

# get zones from image by clicking somewhere in the room
# https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture
clicked = []

def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x, y))

# get clicked points
def get_zone_points():
    img = cv2.imread('D:/Diploma thesis/Maps/UPJS/1.png')
    cv2.imshow('Get zones points', img)
    cv2.setMouseCallback('Get zones points', on_mouse)
    cv2.waitKey(0)
    return clicked

# find zones(rooms) from clicked points and detected walls
def perform_operation(list_of_zone_points, list_of_wall_lines, dictionary_of_zones):
    for zone_point in list_of_zone_points:
        zone = get_zone(zone_point, list_of_wall_lines)

        # dictionary_of_zones = defaultdict(list)
        tuple_point = tuple(zone_point)
        # if tuple_point not in dictionary_of_zones:
        #
        #     dictionary_of_zones[tuple_point] = zone

        dictionary_of_zones[tuple_point].extend(zone)


    return dictionary_of_zones


# get all lines(walls) that create zone(room) based on zone point
def get_zone(zone_point, list_of_wall_lines):
    first_connection = get_first_connection(zone_point, list_of_wall_lines)
    if (first_connection[0] is None) or (first_connection[1] is None):
        return False;

    zone = []
    potential_connection = []

    is_first = True
    start = first_connection[0]
    previous = start
    current = first_connection[1]
    while (not (previous is start)) or is_first:
        is_first = False
        zone.append(previous)
        next_point = next_zone_point(previous, current, list_of_wall_lines, potential_connection)
        # if next_point is None:
        #     print("cannot find another point")
        #     return None
        # if ccw(previous, current, next_point) < 0:
        #     print("angle is not less than 180")
        #     return None

        previous = current
        current = next_point

    return zone


# for given points A, B it finds point C that is connected with point B and also angle between AC and AB is the smallest
def next_zone_point(point_a, point_b, list_of_wall_lines, potential_connection):
    result = []
    best_angle = 1.7976931348623157e+308

    for line in list_of_wall_lines:
        if line[0] == point_b:
            potential_connection = line[1]
        elif line[1] == point_b:
            potential_connection = line[0]
        else:
            continue

        if potential_connection == point_a:
            continue

        print(potential_connection)
        angle = get_angle(point_a, point_b, potential_connection)
        if angle < best_angle:
            result = potential_connection
            best_angle = angle

    return result


# compute angle for given points that they have with node B
def get_angle(point_a, point_b, point_c):
    print(point_c)
    a = math.dist(point_b, point_c)
    b = math.dist(point_a, point_c)
    c = math.dist(point_a, point_b)
    cos_beta = (a * a + c * c - b * b) / (2 * a * c)
    if ccw(point_a, point_b, point_c) < 0:
        computation = 2 * math.pi - math.acos(cos_beta)
        return computation

    computation = math.acos(cos_beta)
    return computation


# counter clock wise computation - check if point c is on the right or left from line  a-b
def ccw(a, b, c):
    print(c)
    result = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if result > 0:
        return 1
    else:
        if result == 0:
            return 0

    return -1


# for zone_point find connection that is closest to line from that point vertically
def get_first_connection(zone_point, list_of_wall_lines):
    zone_point_x = zone_point[0]
    zone_point_y = zone_point[1]
    best_y = 0

    result = []

    for line in list_of_wall_lines:
        map_point_a = line[0]
        map_point_b = line[1]

        if is_between(zone_point[0], map_point_a[0], map_point_b[0]):
            y = get_y(line, zone_point_x)
            if y > zone_point_y:
                continue

            if y > best_y:
                best_y = y
                if map_point_a[0] < map_point_b[0]:
                    result.clear()
                    result.append(map_point_a)
                    result.append(map_point_b)
                    # result[0] = map_point_a
                    # result[1] = map_point_b
                else:
                    result.clear()
                    result.append(map_point_b)
                    result.append(map_point_a)

                    # result[0] = map_point_b
                    # result[1] = map_point_a

    return result


# check if value is between a and b
def is_between(x, a, b):
    if (x > a) and (x > b):
        return False

    if (x < a) and (x < b):
        return False

    return True


# for specified line and coordination of x, compute y coordinate
# in other words compute y coordinate of point [x, y] on line
def get_y(line, zone_point_x):
    ax = line[0][0]
    ay = line[0][1]
    bx = line[1][0]
    by = line[1][1]
    t = (zone_point_x - ax) / (bx - ax)
    connection_coordinate = ay + t * (by - ay)
    return connection_coordinate

def draw(lines, img):
    lines_new = []
    for i in range(0, len(lines)):
        if(i == 5 or i == 1 or i == 2 or i == 6):
            lines_new.append(lines[i])

    print(lines_new)

    for line in lines_new:
        start_point = line[0]
        end_point = line[1]
        color = (0, 255, 0)

        thickness = 2

        image = cv2.line(img, start_point, end_point, color, thickness)

    cv2.imshow('window_name', image)
    cv2.waitKey(0)
    return lines_new


img = cv2.imread('D:/Diploma thesis/Maps/UPJS/1.png')
img = cv2.resize(img, (650, int(650 * img.shape[0] / img.shape[1])))
lsd = cv2.createLineSegmentDetector(0)

LSD(img, lsd)

image_black = cv2.imread('D:/Diploma thesis/Maps/UPJS/LSDresult/01.jpg')
lsdresults = []


LSD_lines_info(image_black, lsdresults, lsd)
# print(lsdresults)

df_lines = pd.DataFrame(columns=['Ax', 'Ay', 'Bx', 'By', 'width', 'length', 'angle'], dtype=np.float32)
empty = np.zeros_like(img)

df_lines = LSD_computing_info_lines(lsdresults, df_lines)

list_of_wall_lines = create_final_list_of_walls(df_lines)
list_of_wall_lines = draw(list_of_wall_lines, img)
list_of_points = get_zone_points()
list_of_zone_points = np.array(list_of_points)



dictionary_of_zones = collections.defaultdict(list)
list_of_zones = perform_operation(list_of_zone_points, list_of_wall_lines, dictionary_of_zones)
print(list_of_zones)