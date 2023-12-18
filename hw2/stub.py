"""
Stub for homework 2
"""
import time
import random
import math
import cv2
import numpy as np
import mujoco
from mujoco import viewer
import PIL

model = mujoco.MjModel.from_xml_path("car.xml")
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)


def sim_step(forward, turn, steps=1000, view=False):
    data.actuator("forward").ctrl = forward
    data.actuator("turn").ctrl = turn
    for _ in range(steps):
        step_start = time.time()
        mujoco.mj_step(model, data)
        if view:
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step / 10)

    renderer.update_scene(data, camera="camera1")
    img = renderer.render()

    return img


def task_1_step(turn):
    return sim_step(0.1, turn, steps=200, view=True)


def task_1():
    steps = random.randint(0, 2000)
    img = sim_step(0, 0.1, steps, view=False)

    # TODO: change the lines below,
    # for car control, you should use task_1_step(turn) function
    # you can change anything below this line

    def get_limits(color):
        """
        This function is used to get the limits of the color in HSV color space.
        """

        c = np.uint8([[color]])
        hsv_color = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)

        lowwer = np.array([hsv_color[0][0][0] - 10, 100, 100])
        upper = np.array([hsv_color[0][0][0] + 10, 255, 255])

        return lowwer, upper

    def find_color(img, lowwer, upper):
        """
        This function is used to find the color in the image and returns the bounding box of the color.
        """
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask = cv2.inRange(hsv_img, lowwer, upper)
        # res = cv2.bitwise_and(img, img, mask=mask)
        mask_ = PIL.Image.fromarray(mask)
        bbox = mask_.getbbox()
        # print(bbox, bbox_area(bbox))
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # pil.Image.fromarray(img).show()

        return bbox

    def bbox_area(bbox):
        """
        This function is used to calculate the area of the bounding box.
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            return (x2 - x1) * (y2 - y1)
        else:
            return 0

    # main loop
    for i in range(1000):
        # image for starting position
        if i == 0:
            img = task_1_step(0)

        # center position on x axis
        img_center_x = img.shape[1] / 2

        # get the limits of the red color
        lowwer, upper = get_limits([255, 0, 0])

        # find the bounding box of the red color (red ball)
        bbox = find_color(img, lowwer, upper)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            middle_x, middle_y = ((x1 + x2) / 2, (y1 + y2) / 2)

            # stop condition is based on the area of the bounding box
            if bbox_area(bbox) < 8000:
                # turn the car based on the position of the red ball
                if middle_x < img_center_x:
                    takeoff = math.fabs(middle_x - img_center_x) / img_center_x / 2
                    turn = min(0.1, takeoff)
                    img = task_1_step(turn)
                elif middle_x > img_center_x:
                    takeoff = math.fabs(middle_x - img_center_x) / img_center_x / 2
                    turn = min(0.1, takeoff)
                    img = task_1_step(-turn)
                elif middle_x == img_center_x:
                    img = task_1_step(0)
            else:
                break
        else:
            img = task_1_step(0.1)
    return img
    # at the end, your car should be close to the red ball (0.2 distance is fine)
    # data.body("car").xpos) is the position of the car


def task_2():
    sim_step(0.5, 0, 1000, view=True)
    speed = random.uniform(0.3, 0.5)
    turn = random.uniform(-0.2, 0.2)
    img = sim_step(speed, turn, 1000, view=True)

    # TODO: change the lines below,
    # you should use sim_step(forward, turn) function
    # you can change the speed and turn as you want
    # do not change the number of steps (1000)

    def find_red_ball(img, view=True):
        for i in range(1000):
            # image for starting position
            if i == 0:
                img = sim_step(0.1, 0.1, 200, view=view)

            # center position on x axis
            img_center_x = img.shape[1] / 2

            # get the limits of the red color
            lowwer, upper = get_limits([255, 0, 0])

            # find the bounding box of the red color (red ball)
            bbox = find_color(img, lowwer, upper)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                middle_x, middle_y = ((x1 + x2) / 2, (y1 + y2) / 2)
                red_ball_area = bbox_area(bbox)
                # print("red ball area", red_ball_area)

                # stop condition is based on the area of the bounding box
                if red_ball_area < 5500:
                    # turn the car based on the position of the red ball
                    if middle_x < img_center_x:
                        takeoff = math.fabs(middle_x - img_center_x) / img_center_x / 2
                        turn = min(0.1, takeoff)
                        img = sim_step(0.1, turn, 200, view=view)
                    elif middle_x > img_center_x:
                        takeoff = math.fabs(middle_x - img_center_x) / img_center_x / 2
                        turn = min(0.1, takeoff)
                        img = sim_step(0.1, -turn, 200, view=view)
                    elif middle_x == img_center_x:
                        if red_ball_area < 500:
                            img = sim_step(1, 0, 1000, view=view)
                        else:
                            img = sim_step(0.1, 0, 200, view=view)
                else:
                    break
            else:
                img = sim_step(0.1, 0.1, 200, view=view)
        return img

    def get_limits(color):
        """
        This function is used to get the limits of the color in HSV color space.
        """
        c = np.uint8([[color]])
        hsv_color = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)

        lowwer = np.array([hsv_color[0][0][0] - 10, 100, 100])
        upper = np.array([hsv_color[0][0][0] + 10, 255, 255])

        return lowwer, upper

    def find_color(img, lowwer, upper):
        """
        This function is used to find the color in the image and returns the bounding box of the color.
        """
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask = cv2.inRange(hsv_img, lowwer, upper)
        mask_ = PIL.Image.fromarray(mask)
        bbox = mask_.getbbox()

        return bbox

    def bbox_area(bbox):
        """
        This function is used to calculate the area of the bounding box.
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            return (x2 - x1) * (y2 - y1)
        else:
            return 0

    def circle_travel(forward, i=1, steps=1000, view=True):
        """
        This function is used to make the car travel in a circle. It assume that car is looking into circle central point
        """
        sim_step(0, 0.12, 1000, view)
        if i == 0:
            for _ in range(12):
                sim_step(forward, -0.02, 1000, view)
        else:
            sim_step(forward, -0.02, steps, view)
        img = sim_step(0, -0.12, 1000, view)
        return img

    def check_bbox_collision(bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        if x2_1 < x1_2 or x2_2 < x1_1:
            return False

        if y2_1 < y1_2 or y2_2 < y1_1:
            return False

        return True

    def allign_red_ball_with_green_box(img, view=True):
        was_greenbox_in_last_frame = False
        for i in range(1000):
            red_bbox = find_color(img, *get_limits([255, 0, 0]))
            green_bbox = find_color(img[:400, :], *get_limits([0, 255, 0]))

            # if red ball is lost find it again
            if red_bbox is None:
                img = find_red_ball(img, view)
            # if there is no green box, circle around or prepeare to push the ball
            elif red_bbox is not None and green_bbox is None:
                if was_greenbox_in_last_frame:
                    img = find_red_ball(img, view)
                    return img
                else:
                    img = circle_travel(0.1, i, view=view)
            # if there is redball and green box, allign them
            elif red_bbox is not None and green_bbox is not None:
                if check_bbox_collision(red_bbox, green_bbox):
                    img = circle_travel(0.1, i, steps=400, view=view)
                else:
                    img = circle_travel(0.1, i, view=view)
            else:
                img = circle_travel(0.1, i, view=view)

            if green_bbox:
                was_greenbox_in_last_frame = True
            else:
                was_greenbox_in_last_frame = False

    def push_ball_to_box(img, view=True):
        for i in range(1000):
            green_bbox = find_color(img[:400, :], *get_limits([0, 255, 0]))

            if green_bbox is None:
                img = sim_step(0.1, 0, 1000, view=view)
            else:
                green_area = bbox_area(green_bbox)
                if green_area < 7000:
                    img = sim_step(0.1 + 0.01 * i, 0, 1000, view=view)
                else:
                    img = sim_step(0, 0, 1000, view=True)
                    return img

    img = find_red_ball(img, True)
    img = allign_red_ball_with_green_box(img, True)
    push_ball_to_box(img, True)
    # at the end, red ball should be close to the green box (0.25 distance is fine)


drift = 0


def task3_step(forward, turn, steps=1000, view=False):
    return sim_step(forward, turn + drift, steps=steps, view=view)


def task_3():
    global drift
    drift = np.random.uniform(-0.1, 0.1)

    # TODO: change the lines below
    # you should use task3_step(forward, turn, steps) function
    def calibrate(img, view=True):
        drift_correction = 0
        sign = 0
        while True:
            lowwer_green, upper_green = get_limits([0, 255, 0])
            lowwer_red, upper_red = get_limits([255, 0, 0])

            img2 = task3_step(0, drift_correction, 1000, view=view)
            if np.array_equal(img, img2):
                return img2, drift_correction

            bbox1_green = find_color(img[:400, :], lowwer_green, upper_green)
            bbox2_green = find_color(img2[:400, :], lowwer_green, upper_green)
            bbox1_red = find_color(img, lowwer_red, upper_red)
            bbox2_red = find_color(img2, lowwer_red, upper_red)

            if bbox1_green is not None and bbox2_green is not None:
                bbox1, bbox2 = bbox1_green, bbox2_green
            elif bbox1_red is not None and bbox2_red is not None:
                bbox1, bbox2 = bbox1_red, bbox2_red
            else:
                bbox1, bbox2 = None, None

            if bbox1 is not None and bbox2 is not None:
                x11, y11, x12, y12 = bbox1
                middle_x1, middle_y1 = ((x11 + x12) / 2, (y11 + y12) / 2)
                x21, y21, x22, y22 = bbox2
                middle_x2, middle_y2 = ((x21 + x22) / 2, (y21 + y22) / 2)

                if middle_x1 < middle_x2:
                    if sign == 1:
                        return img, drift_correction
                    drift_correction -= 0.001
                    sign = -1
                else:
                    if sign == -1:
                        return img, drift_correction
                    drift_correction += 0.001
                    sign = 1

            img = img2

    def find_green_box(img, dc, view=True):
        for i in range(1000):
            # center position on x axis
            img_center_x = img.shape[1] / 2

            # get the limits of the red color
            lowwer, upper = get_limits([0, 255, 0])

            # find the bounding box of the red color (red ball)
            bbox = find_color(img[:400, :], lowwer, upper)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                middle_x, middle_y = ((x1 + x2) / 2, (y1 + y2) / 2)
                green_box_area = bbox_area(bbox)

                # stop condition is based on the area of the bounding box
                if green_box_area < 10000:
                    # turn the car based on the position of the red ball
                    if middle_x < img_center_x:
                        img = task3_step(0.1, 0.05 + dc, view=view)
                    elif middle_x > img_center_x:
                        img = task3_step(0.1, -0.05 + dc, 200, view=view)
                    elif middle_x == img_center_x:
                        img = task3_step(0.1, dc, view=view)
                else:
                    break
            else:
                img = task3_step(0, 0.01 + dc, view=view)
        return img

    def get_limits(color):
        """
        This function is used to get the limits of the color in HSV color space.
        """
        c = np.uint8([[color]])
        hsv_color = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)

        lowwer = np.array([hsv_color[0][0][0] - 10, 100, 100])
        upper = np.array([hsv_color[0][0][0] + 10, 255, 255])

        return lowwer, upper

    def find_color(img, lowwer, upper):
        """
        This function is used to find the color in the image and returns the bounding box of the color.
        """
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask = cv2.inRange(hsv_img, lowwer, upper)
        # res = cv2.bitwise_and(img, img, mask=mask)
        mask_ = PIL.Image.fromarray(mask)
        bbox = mask_.getbbox()
        # print(bbox, bbox_area(bbox))
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # pil.Image.fromarray(img).show()

        return bbox

    def bbox_area(bbox):
        """
        This function is used to calculate the area of the bounding box.
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            return (x2 - x1) * (y2 - y1)
        else:
            return 0

    def get_middle_point_of_two_bboxes(bbox1, bbox2):
        x11, y11, x12, y12 = bbox1
        middle_x1, middle_y1 = ((x11 + x12) / 2, (y11 + y12) / 2)
        x21, y21, x22, y22 = bbox2
        middle_x2, middle_y2 = ((x21 + x22) / 2, (y21 + y22) / 2)

        return (middle_x1 + middle_x2) / 2, (middle_y1 + middle_y2) / 2

    def drive_to_get_perspective(img, dc, view=True):
        task3_step(0, 0.09 + dc, view=view)
        task3_step(1, dc, view=view)
        task3_step(1, dc, view=view)
        img = task3_step(0, 0.2 + dc, view=view)

        img_middle_x = img.shape[1] / 2
        target_seen = False

        for i in range(1000):
            bbox_green = find_color(img[:400, :], *get_limits([0, 255, 0]))
            bbox_blue = find_color(img[:400, :], *get_limits([0, 0, 255]))

            if bbox_green is not None and bbox_blue is not None:
                target_seen = True
                target_x, _ = get_middle_point_of_two_bboxes(bbox_green, bbox_blue)

                if target_x < img_middle_x:
                    img = task3_step(0.05, 0.005 + dc, view=view)
                elif target_x > img_middle_x:
                    img = task3_step(0.05, -0.005 + dc, view=view)
                else:
                    img = task3_step(0.05, dc, view=view)

            elif target_seen:
                img = task3_step(0.15, dc, view=view)
                return img
            elif not target_seen:
                img = task3_step(0, 0.01 + dc, view=view)

        return img

    img = task3_step(0, 0)
    img, drift_correction = calibrate(img, view=True)
    img = find_green_box(img, drift_correction, view=True)
    img = drive_to_get_perspective(img, drift_correction, view=True)
    # at the end, car should be between the two boxes


# task_1()
task_2()
# task_3()
