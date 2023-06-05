import cv2
import numpy as np
import os

from glob import glob


img_1_init_points = [(376, 140), (367, 179), (363, 229), (356, 257), 
                     (341, 287), (321, 326), (309, 368), (293, 417), 
                     (275, 492), (280, 573), (290, 616), (300, 652), 
                     (304, 678), (321, 732), (340, 776), (360, 812), 
                     (378, 831), (417, 846), (468, 855), (517, 857), 
                     (577, 855), (616, 850), (633, 834), (649, 809), 
                     (652, 787), (667, 756), (686, 723), (703, 688), 
                     (715, 660), (724, 627), (742, 556), (739, 504), 
                     (737, 440), (720, 390), (696, 355), (667, 325), 
                     (665, 300), (671, 264), (678, 219), (678, 193), 
                     (672, 175), (659, 163), (637, 154), (605, 143), 
                     (576, 137), (545, 134), (513, 131), (478, 129), 
                     (438, 129), (409, 130), (395, 132)]

img_2_init_points = [(274, 365), (264, 383), (242, 388), (213, 377), 
                     (167, 350), (130, 341), (106, 348), (84, 361), 
                     (50, 381), (10, 425), (21, 473), (38, 509), 
                     (58, 565), (91, 643), (127, 702), (162, 746), 
                     (194, 785), (248, 819), (256, 834), (313, 867), 
                     (346, 881), (433, 892), (520, 894), (574, 882), 
                     (648, 855), (700, 834), (735, 792), (757, 751), 
                     (794, 721), (840, 677), (878, 635), (893, 606), 
                     (905, 564), (909, 515), (902, 461), (888, 412), (868, 368), (823, 336), (769, 316), (709, 321), (659, 333), (622, 331), (601, 309), (572, 302), (525, 299), (491, 298), (455, 297), (418, 297), (384, 304), (355, 318), (335, 332), (309, 340), (307, 340)]

img_3_init_points = [(233, 73), (221, 99), (211, 133), (199, 173), 
                     (187, 219), (185, 286), (181, 337), (180, 407), 
                     (183, 468), (189, 518), (194, 573), (200, 633), 
                     (213, 691), (245, 765), (282, 855), (325, 919), 
                     (352, 931), (417, 944), (507, 949), (575, 952), 
                     (628, 952), (663, 930), (703, 886), (737, 814), 
                     (757, 764), (787, 706), (803, 644), (816, 581), 
                     (827, 511), (841, 430), (845, 356), (848, 245), 
                     (834, 167), (810, 112), (765, 75), (702, 53), 
                     (628, 38), (545, 30), (477, 20), (421, 20), 
                     (360, 30), (323, 37), (284, 55)]

def mouse_callback(event, x, y, flags, init_points) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        # print('mouse clicked')
        # print(x, y)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('img', img)

        init_points.append((x, y))


def set_init_point(image: np.ndarray) -> tuple:
    cv2.imshow('img', image)
    init_points = []
    cv2.setMouseCallback('img', mouse_callback, init_points)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()

    return init_points
    

def activate_contour(grad_img: np.ndarray, points: list, alpha: float, beta: float, gamma: float) -> np.ndarray:

    for i in range(len(points)):
        energy_min = float('inf')

        prev_point = points[i-1]
        next_point = points[(i+1) % len(points)]

        # set search region
        search_param = 4

        for search_x in range(points[i][0] - search_param, points[i][0] + search_param + 1):
            for search_y in range(points[i][1] - search_param, points[i][1] + search_param + 1):

                energy_cont = pow(search_x - prev_point[0], 2) + pow(search_y - prev_point[1], 2)
                energy_curv = pow((prev_point[0]) - 2*(search_x) + (next_point[0]), 2) + pow((prev_point[1]) - 2*(search_y) + next_point[1], 2)
                energy_img = -1 * abs(grad_img[search_y][search_x])
                energy_total = alpha * energy_cont + beta * energy_curv + gamma * energy_img

                if energy_total < energy_min:
                    energy_min = energy_total
                    points[i] = (search_x, search_y)
                    
    return points

def gradient(img: np.ndarray, kernal_size: int) -> np.ndarray:

    grad_img_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernal_size)
    grad_img_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernal_size)
    grad_img = cv2.addWeighted(cv2.convertScaleAbs(grad_img_x), 0.5, cv2.convertScaleAbs(grad_img_y), 0.5, 0)

    # grad_img = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)

    return grad_img

def draw_points(img: np.ndarray, points: list) -> None:
    for i in range(len(points)):
        cv2.circle(img, points[i], 3, (0, 0, 255), -1)
        if i > 0:
            cv2.line(img, points[i-1], points[i], (0, 0, 255), 1)
        if i == len(points) - 1:
            cv2.line(img, points[i], points[0], (0, 0, 255), 1)

def set_rectangle_points(img: np.ndarray, each_edge_points_num: int) -> tuple:
    height, width = img.shape[:2]
    init_x, init_y = 50, 50
    end_x, end_y = width - 50, height - 50

    points = []
    for i in range(each_edge_points_num):
        points.append((init_x + (end_x - init_x) // each_edge_points_num * i, init_y))
    for i in range(each_edge_points_num):
        points.append((end_x, init_y + (end_y - init_y) // each_edge_points_num * i))
    for i in range(each_edge_points_num):
        points.append((end_x - (end_x - init_x) // each_edge_points_num * i, end_y))
    for i in range(each_edge_points_num):
        points.append((init_x, end_y - (end_y - init_y) // each_edge_points_num * i))

    return points

if __name__ == '__main__':
    use_set_up_init_point_mode = False
    use_saved_init_points_mode = False
    use_rectangle_points_mode = True
    rectangle_points_num = 20

    Max_iter = 500
    alpha = 2
    beta = 4
    gamma = 100
    kernal_size = 3

    img_list = glob('./test_img/*.jpg')
    img_list.sort()
    print(img_list)

    for i, img_path in enumerate(img_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if i == 0:
            contour_points = img_1_init_points if use_saved_init_points_mode else None
        elif i == 1:
            contour_points = img_2_init_points if use_saved_init_points_mode else None
        elif i == 2:
            contour_points = img_3_init_points if use_saved_init_points_mode else None

        if use_set_up_init_point_mode:
            points_img = img.copy()
            contour_points = set_init_point(img)
            print(contour_points)
            img = points_img.copy()

        if use_rectangle_points_mode:
            contour_points = set_rectangle_points(img, rectangle_points_num)

        img = cv2.GaussianBlur(img, (5, 5), 0)
        grad_img = gradient(img, kernal_size)
        cv2.imshow('grad_img', grad_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        video_writer = cv2.VideoWriter('result/{}.mp4'.format(os.path.basename(img_path).split('.')[0]), 
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       10,
                                       (img.shape[1], img.shape[0]),
                                       False)

        for iter in range(Max_iter):
            print('iter: ', iter)

            img_copy = img.copy()
            draw_points(img_copy, contour_points)
            
            
            if iter == 0:
                cv2.imwrite('result/{}_init_points.jpg'.format(os.path.basename(img_path).split('.')[0]), img_copy)
                cv2.imshow('init_points', img_copy)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            key = cv2.waitKey(1)
            cv2.imshow('result', img_copy)
            video_writer.write(img_copy)

            if key == ord('q'):
                break

            contour_points = activate_contour(grad_img, contour_points, alpha, beta, gamma)

        video_writer.release()
        video_writer = None
        
        cv2.imwrite('result/{}.jpg'.format(os.path.basename(img_path).split('.')[0]), img_copy)
        cv2.imshow('img', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
            




        
 


