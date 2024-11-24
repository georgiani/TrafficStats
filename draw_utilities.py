import cv2

def draw_line(img, x1, y1, x2, y2):
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)