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

# TOOLS
# generating number labels for pictures
list_of_numbers = []
for i in range(0, 170):
    number = f"{i:02d}"
    list_of_numbers.append(number)


# directory listing of color Dataset
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


# convert dataframe into required list format
def create_final_list_of_walls(df):
    list_of_lines = []
    for k in range(0, len(df)):

        x_s = df.at[k, 'Ax']
        y_s = df.at[k, 'Ay']
        start_point = (int(x_s), int(y_s))

        x_e = df.at[k, 'Bx']
        y_e = df.at[k, 'By']
        end_point = (int(x_e), int(y_e))
        if (start_point[0] != end_point[0] or start_point[1] != end_point[1]):
            list_of_lines.append((start_point, end_point))

    return list_of_lines


# ------------------------------------------------------------------------------------------------------------------

# WALLS DETECTION
# LSD - LINE SEGMENT DETECTOR
def LSD(image, lsd):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    empty = np.zeros_like(img)

    # use LSD detector
    lsdresult = lsd.detect(img)

    for dline in lsdresult[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv2.line(empty, (x0, y0), (x1, y1), (118, 238, 0), 1, cv2.LINE_AA)

    cv2.imwrite('D:/Diploma thesis/Maps/UPJS/ResultsThesis/BLACK_LSD.png', empty)

    for dline in lsdresult[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv2.line(img, (x0, y0), (x1, y1), (118, 238, 0), 1, cv2.LINE_AA)

    cv2.imwrite('D:/Diploma thesis/Maps/UPJS/ResultsThesis/LSD.png', img)

    return lsdresult


def LSD_lines_info(image_black, lsdresults, lsdresult, lsd):
    lsdresults.append(lsdresult)


def LSD_computing_info_points(lsdresults, image, df_lines, empty):
    for i in range(0, len(lsdresults)):
        res = lsdresults[i]
        res_lines = res[0].reshape(-1, 4)
        res_width = res[1].reshape(-1)
        res_points = res[0].reshape(-1, 2)
        blank_picture = np.zeros_like(empty)

        # mean shift clusteriasation
        df_final = mean_shift(res_lines, res_points, blank_picture, image)
        width_col = res_width.reshape(-1, 1)

        full_lines = np.column_stack((res_lines, res_width))

        # create a data frame from full_lines, with defines columns
        df_line = pd.DataFrame(full_lines, columns=['Ax', 'Ay', 'Bx', 'By', 'width'], dtype=np.float32)

        # calculation of lenght and angle of lines and adding label based on number of picture in queue
        df_line['length'] = np.sqrt((df_line['Ax'] - df_line['Bx']) ** 2 + (df_line['Ay'] - df_line['By']) ** 2)
        df_line['angle'] = np.abs(
            np.arctan2(df_line['Ay'] - df_line['By'], df_line['Ax'] - df_line['Bx']) * 180 / np.pi)

    return df_lines, df_final


def mean_shift_x(points, img):
    points_x = np.empty(shape=(len(points), 2))
    points_x.fill(0)
    for i in range(0, len(points)):
        point = points[i, 0]
        if (i == 0) or (i % 2 == 0):
            np.put(points_x, i, [point, 0])

    #  5
    cluster_x = MeanShift(bandwidth=5)
    cluster_x.fit(points_x)
    cluster_centers_x = cluster_x.cluster_centers_
    labels = cluster_x.labels_

    return cluster_centers_x, labels, points_x


def mean_shift_y(points, img):
    points_y = np.empty(shape=(len(points), 2))
    points_y.fill(0)
    y_axis_points = []

    for i in range(0, len(points)):
        point = points[i, 1]
        y_axis_points.append(point)

    index = 1
    for i in range(0, len(y_axis_points)):
        np.put(points_y, i + index, y_axis_points[i])
        index = index + 1

    cluster_y = MeanShift(bandwidth=5)
    cluster_y.fit(points_y)
    cluster_centers_y = cluster_y.cluster_centers_
    labels = cluster_y.labels_

    return cluster_centers_y, labels, points_y


def mean_shift(lsd_lines, points, blank_picture, img):
    clustering = MeanShift(bandwidth=5)
    clustering.fit(points)
    labels = clustering.labels_
    cluster_centers = clustering.cluster_centers_
    n_clusters_ = len(np.unique(labels))

    # adding resulting cluster centers in plot but represented as circle (point) - first clusterization visualisation
    # for i in range(0, len(cluster_centers)):
    #     img = cv2.circle(img, (int(cluster_centers[i, 0]), int(cluster_centers[i, 1])), radius=0, color=(0, 0, 255),
    #                      thickness=8)
    #
    # cv2.imwrite('D:/Diploma thesis/Maps/UPJS/ResultsThesis/cluster_first.jpg', img)
    # cv2.imshow('clusters', img)
    # cv2.waitKey(0)

    tuple_list = []
    for k in range(0, len(labels) - 1):
        if k == 0:
            tuple = (labels[k], labels[k + 1])
            tuple_list.append(tuple)

        if (k != 0) and (k % 2 == 0):
            tuple = (labels[k], labels[k + 1])
            tuple_list.append(tuple)

    tuples = np.array(tuple_list)
    full_data = np.column_stack((lsd_lines, tuples))
    df_clustering = pd.DataFrame(full_data, columns=['Ax', 'Ay', 'Bx', 'By', 'cluster_index1', 'cluster_index2'],
                                 dtype=np.float32)

    for s in range(0, len(full_data)):
        cluster1 = df_clustering.at[s, 'cluster_index1']
        cluster2 = df_clustering.at[s, 'cluster_index2']
        centroid1 = cluster_centers[int(cluster1)]
        centroid2 = cluster_centers[int(cluster2)]
        df_clustering.at[s, 'Ax'] = centroid1[0]
        df_clustering.at[s, 'Ay'] = centroid1[1]
        df_clustering.at[s, 'Bx'] = centroid2[0]
        df_clustering.at[s, 'By'] = centroid2[1]

    df = df_clustering.drop_duplicates(subset=['Ax', 'Ay', 'Bx', 'By'], ignore_index=True, keep='first')

    # clustering for x and y-axis separately
    cluster_centers_x, labels_x, points_x = mean_shift_x(cluster_centers, img)
    cluster_centers_y, labels_y, points_y = mean_shift_y(cluster_centers, img)

    # wall reconstruction
    for i in range(0, len(points_x)):
        replaced = points_x[i]
        replaced_x = replaced[0]
        label_x = labels_x[i]
        cluster_center = cluster_centers_x[label_x]
        cluster_x = cluster_center[0]
        df['Ax'] = df['Ax'].replace(to_replace=replaced_x, value=cluster_x)
        df['Bx'] = df['Bx'].replace(to_replace=replaced_x, value=cluster_x)

    for i in range(0, len(points_y)):
        replaced = points_y[i]
        replaced_y = replaced[1]
        label_y = labels_y[i]
        cluster_center = cluster_centers_y[label_y]
        cluster_y = cluster_center[1]
        df['Ay'] = df['Ay'].replace(to_replace=replaced_y, value=cluster_y)
        df['By'] = df['By'].replace(to_replace=replaced_y, value=cluster_y)

    df_final = df.drop_duplicates(subset=['Ax', 'Ay', 'Bx', 'By'], ignore_index=True, keep='first')

    for l in range(0, len(df_final)):
        x_s = df_final.at[l, 'Ax']
        y_s = df_final.at[l, 'Ay']
        start_point = (int(x_s), int(y_s))

        x_e = df_final.at[l, 'Bx']
        y_e = df_final.at[l, 'By']
        end_point = (int(x_e), int(y_e))

        color = (0, 255, 0)

        thickness = 2
        if (start_point[0] != end_point[0] or start_point[1] != end_point[1]):
            img = cv2.line(img, start_point, end_point, color, thickness)

    for l in range(0, len(df_final)):
        x_s = df_final.at[l, 'Ax']
        y_s = df_final.at[l, 'Ay']
        start_point = (int(x_s), int(y_s))
        x_e = df_final.at[l, 'Bx']
        y_e = df_final.at[l, 'By']
        if (start_point[0] != end_point[0] or start_point[1] != end_point[1]):
            img = cv2.circle(img, (int(x_s), int(y_s)), radius=0, color=(0, 0, 255), thickness=5)
            img = cv2.circle(img, (int(x_e), int(y_e)), radius=0, color=(0, 0, 255), thickness=5)

    cv2.imwrite('D:/Diploma thesis/Maps/UPJS/LSDresult/points1.jpg', img)
    cv2.imshow('results', img)
    cv2.waitKey(0)

    cv2.imshow('results', img)
    cv2.waitKey(0)
    img = cv2.resize(img, (650, int(650 * img.shape[0] / img.shape[1])))
    cv2.imwrite('D:/Diploma thesis/Maps/UPJS/ResultsThesis/mean_shift_altered12.png', img)

    return df_final


# ------------------------------------------------------------------------------------------------------------------


# DEFINING ROOMS

# get zones from image by clicking somewhere in the room
# https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture
clicked = []


def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x, y))


# get clicked points from wall detection image
def get_zone_points():
    img = cv2.imread('D:/Diploma thesis/Maps/UPJS/ResultsThesis/mean_shift_altered12.png')
    cv2.imshow('Get zones points', img)
    cv2.setMouseCallback('Get zones points', on_mouse)
    cv2.waitKey(0)
    return clicked


# find zones(rooms) from clicked points and detected walls
def perform_operation(list_of_zone_points, list_of_wall_lines, dictionary_of_zones):
    for zone_point in list_of_zone_points:
        zone = get_zone(zone_point, list_of_wall_lines)

        tuple_point = tuple(zone_point)
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
    while (not (previous == start)) or is_first:
        is_first = False
        zone.append(previous)
        next_point = next_zone_point(previous, current, list_of_wall_lines, potential_connection)
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

        angle = get_angle(point_a, point_b, potential_connection)
        if angle < best_angle:
            result = potential_connection
            best_angle = angle

    return result


# compute angle for given points that they have with node B
def get_angle(point_a, point_b, point_c):
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

                else:
                    result.clear()
                    result.append(map_point_b)
                    result.append(map_point_a)

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


# ------------------------------------------------------------------------------------------------------------------
# MAIN

# load and resize image
img = cv2.imread('D:/Diploma thesis/Maps/UPJS/4.jpg')
# img = cv2.resize(img, (650, int(650 * img.shape[0] / img.shape[1])))

# initialise LSD
lsd = cv2.createLineSegmentDetector(0)

# detect lines with LSD
lsdresult = LSD(img, lsd)

# load detected black image + create result list
image_black = cv2.imread('D:/Diploma thesis/Maps/UPJS/ResultsThesis/BLACK_LSD.png')

# get info from the LSD results about detected walls
lsdresults = []
LSD_lines_info(image_black, lsdresults, lsdresult, lsd)

# create global dataframe for lines and blank picture like the original one
df_lines = pd.DataFrame(columns=['Ax', 'Ay', 'Bx', 'By', 'width', 'length', 'angle', 'img_label'], dtype=np.float64)
empty = np.zeros_like(img)

# wall detection result
lines, df_final = LSD_computing_info_points(lsdresults, img, df_lines, empty)

list_of_wall_lines = create_final_list_of_walls(df_final)
list_of_points = get_zone_points()
list_of_zone_points = np.array(list_of_points)

# find convex regions
dictionary_of_zones = collections.defaultdict(list)
list_of_zones = perform_operation(list_of_zone_points, list_of_wall_lines, dictionary_of_zones)
print(list_of_zones)
