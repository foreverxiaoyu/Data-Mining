import numpy as np
from numpy import ndarray
import PIL
from PIL import ImageDraw, ImageFont
from PIL.Image import Image

# COCO 17 points
point_name = ["nose", "left_eye", "right_eye",
              "left_ear", "right_ear",
              "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow",
              "left_wrist", "right_wrist",
              "left_hip", "right_hip",
              "left_knee", "right_knee",
              "left_ankle", "right_ankle"]

point_color = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
               (240, 2, 127), (240, 2, 127),
               (255, 255, 51), (255, 255, 51),
               (254, 153, 41), (44, 127, 184),
               (217, 95, 14), (0, 0, 255),
               (255, 255, 51), (255, 255, 51), (228, 26, 28),
               (49, 163, 84), (252, 176, 243), (0, 176, 240),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142)]

connections = [(0, 1), (0, 2), (1, 3), (2, 4),
               (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
               (5, 11), (6, 12), (11, 12),
               (11, 13), (13, 15), (12, 14), (14, 16)]


def draw_keypoints(img: Image,
                   keypoints: ndarray,
                   scores: ndarray = None,
                   thresh: float = 0.2,
                   r: int = 2,
                   draw_text: bool = False,
                   font: str = 'arial.ttf',
                   font_size: int = 10):
    if isinstance(img, ndarray):
        img = PIL.Image.fromarray(img)

    if scores is None:
        scores = np.ones(keypoints.shape[0])

    if draw_text:
        try:
            font = ImageFont.truetype(font, font_size)
        except IOError:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)

    # 绘制连接线
    for connection in connections:
        pt1, pt2 = connection
        if scores[pt1] > thresh and scores[pt2] > thresh:
            point1, point2 = keypoints[pt1], keypoints[pt2]
            if np.max(point1) > 0 and np.max(point2) > 0:
                draw.line([point1[0], point1[1], point2[0], point2[1]],
                          fill=point_color[pt1], width=2)

    # 计算左右肩膀的中点并与鼻子相连
    if scores[0] > thresh and scores[5] > thresh and scores[6] > thresh:
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        midpoint = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
        if np.max(nose) > 0 and np.max(midpoint) > 0:
            draw.line([nose[0], nose[1], midpoint[0], midpoint[1]], fill=point_color[0], width=2)

    for i, (point, score) in enumerate(zip(keypoints, scores)):
        if score > thresh and np.max(point) > 0:
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                         fill=point_color[i],
                         outline=(255, 255, 255))
            if draw_text:
                draw.text((point[0] + r, point[1] + r), text=point_name[i], font=font)

    return img
