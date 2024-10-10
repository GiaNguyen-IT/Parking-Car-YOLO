import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import torch

# Kiểm tra xem GPU có khả dụng không
if not torch.cuda.is_available():
    print("CUDA is not available. Please check your GPU configuration.")
else:
    print("CUDA is available.")
    print("Current GPU device:", torch.cuda.current_device())
    print("GPU device name:", torch.cuda.get_device_name(0))

# Khởi tạo mô hình YOLO và chuyển sang GPU
model = YOLO('yolov8m.pt').to('cuda')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Mở video
cap = cv2.VideoCapture('parking1.mp4')

# Đọc danh sách lớp từ file coco.txt
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Định nghĩa các khu vực
areas = [
    [(298, 40), (350, 38), (348, 92), (283, 90)],
    [(353, 38), (415, 40), (420, 93), (349, 92)],
    [(415, 39), (479, 41), (498, 96), (421, 92)],
    [(480, 41), (550, 46), (574, 102), (499, 96)],
    [(550, 45), (617, 47), (660, 110), (582, 101)],
    [(618, 50), (690, 55), (738, 118), (660, 110)],
    [(697, 56), (758, 63), (815, 128), (742, 118)],
    [(283, 94), (344, 95), (340, 169), (265, 165)],
    [(348, 94), (420, 96), (430, 176), (344, 167)],
    [(422, 97), (501, 99), (522, 180), (432, 174)],
    [(500, 99), (578, 105), (619, 191), (525, 182)],
    [(578, 107), (659, 111), (714, 199), (621, 192)],
    [(665, 113), (741, 120), (806, 210), (720, 199)],
    [(745, 123), (820, 131), (890, 222), (806, 208)],
    [(246, 280), (337, 285), (366, 437), (247, 433)],
    [(337, 285), (435, 286), (500, 438), (366, 436)],
    [(500, 290), (600, 293), (711, 423), (579, 432)]
]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    time.sleep(0.01)
    frame = cv2.resize(frame, (1020, 500))

    # Dự đoán trên GPU
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a.cpu()).astype("float")

    occupied = [0] * len(areas)  # Danh sách lưu trạng thái chiếm chỗ của từng khu vực

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)

            for i, area in enumerate(areas):
                results = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if results >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    occupied[i] = 1
                    cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    # Tính số chỗ trống
    total_spaces = len(areas)
    occupied_spaces = sum(occupied)
    available_spaces = total_spaces - occupied_spaces
    print("Available spaces:", available_spaces)

    # Hiển thị trạng thái của từng khu vực
    for i, area in enumerate(areas):
        color = (0, 0, 255) if occupied[i] == 1 else (0, 255, 0)
        cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
        cv2.putText(frame, str(i + 1), (area[0][0], area[0][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    # Hiển thị số chỗ trống trên khung hình
    cv2.putText(frame, f"Available: {available_spaces}", (23, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    # Hiển thị khung hình
    cv2.imshow("RGB", frame)

    key = cv2.waitKey(1)
    if key == 32:  # Tua qua 10 phút nếu nhấn phím Space
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        new_time = current_time + 600000
        cap.set(cv2.CAP_PROP_POS_MSEC, new_time)

    if key & 0xFF == 27:  # Nhấn phím Esc để thoát
        break

cap.release()
cv2.destroyAllWindows()
