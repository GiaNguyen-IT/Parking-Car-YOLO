import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

model = YOLO('yolov5m.pt')

# Hàm callback để lấy tọa độ RGB khi di chuột trên cửa sổ video
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

# Tạo cửa sổ để hiển thị video
cv2.namedWindow('Parking Car')
cv2.setMouseCallback('Parking Car', RGB)

# Đọc video
cap = cv2.VideoCapture('parking1.mp4')

# Đọc danh sách các lớp đối tượng từ file coco.txt
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Định nghĩa các khu vực đỗ xe (tọa độ các đa giác)
areas = [
    [(298,40),(350,38),(348,92),(283,90)],  # area1
    [(353,38),(415,40),(420,93),(349,92)],  # area2
    [(415,39),(479,41),(498,96),(421,92)],  # area3
    [(480,41),(550,46),(574,102),(499,96)],  # area4
    [(550,45),(617,47),(660,110),(582,101)], # area5
    [(618,50),(690,55),(738,118),(660,110)], # area6
    [(697,56),(758,63),(815,128),(742,118)], # area7
    [(283,94),(344,95),(340,169),(265,165)], # area8
    [(348,94),(420,96),(430,176),(344,167)], # area9
    [(422,97),(501,99),(522,180),(432,174)], # area10
    [(500,99),(578,105),(619,191),(525,182)], # area11
    [(578,107),(659,111),(714,199),(621,192)], # area12
    [(665,113),(741,120),(806,210),(720,199)], # area13
    [(745,123),(820,131),(890,222),(806,208)], # area14
    [(246,280),(337,285),(366,437),(247,433)], # area15
    [(337,285),(435,286),(500,438),(366,436)], # area16
    [(500,290),(600,293),(711,423),(579,432)]  # area17
]

# Tốc độ khung hình mục tiêu (chỉ xử lý 15 khung hình mỗi giây)
fps_target = 15
frame_interval = int(cap.get(cv2.CAP_PROP_FPS) // fps_target)

# Đếm số lượng khung hình
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chỉ xử lý khung hình sau mỗi 'frame_interval' khung
    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    time.sleep(0.01)
    frame = cv2.resize(frame, (1020, 500))

    # Dự đoán đối tượng trong khung hình bằng YOLO
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Khởi tạo số lượng xe trong từng khu vực
    a1 = a2 = a3 = a4 = a5 = a6 = a7 = a8 = a9 = a10 = a11 = a12 = a13 = a14 = a15 = a16 = a17 = 0

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        if 'car' in c:
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            # Kiểm tra xe có nằm trong khu vực nào không
            if cv2.pointPolygonTest(np.array(areas[0], np.int32), (cx, cy), False) >= 0: a1 += 1
            if cv2.pointPolygonTest(np.array(areas[1], np.int32), (cx, cy), False) >= 0: a2 += 1
            if cv2.pointPolygonTest(np.array(areas[2], np.int32), (cx, cy), False) >= 0: a3 += 1
            if cv2.pointPolygonTest(np.array(areas[3], np.int32), (cx, cy), False) >= 0: a4 += 1
            if cv2.pointPolygonTest(np.array(areas[4], np.int32), (cx, cy), False) >= 0: a5 += 1
            if cv2.pointPolygonTest(np.array(areas[5], np.int32), (cx, cy), False) >= 0: a6 += 1
            if cv2.pointPolygonTest(np.array(areas[6], np.int32), (cx, cy), False) >= 0: a7 += 1
            if cv2.pointPolygonTest(np.array(areas[7], np.int32), (cx, cy), False) >= 0: a8 += 1
            if cv2.pointPolygonTest(np.array(areas[8], np.int32), (cx, cy), False) >= 0: a9 += 1
            if cv2.pointPolygonTest(np.array(areas[9], np.int32), (cx, cy), False) >= 0: a10 += 1
            if cv2.pointPolygonTest(np.array(areas[10], np.int32), (cx, cy), False) >= 0: a11 += 1
            if cv2.pointPolygonTest(np.array(areas[11], np.int32), (cx, cy), False) >= 0: a12 += 1
            if cv2.pointPolygonTest(np.array(areas[12], np.int32), (cx, cy), False) >= 0: a13 += 1
            if cv2.pointPolygonTest(np.array(areas[13], np.int32), (cx, cy), False) >= 0: a14 += 1
            if cv2.pointPolygonTest(np.array(areas[14], np.int32), (cx, cy), False) >= 0: a15 += 1
            if cv2.pointPolygonTest(np.array(areas[15], np.int32), (cx, cy), False) >= 0: a16 += 1
            if cv2.pointPolygonTest(np.array(areas[16], np.int32), (cx, cy), False) >= 0: a17 += 1

    # Số chỗ trống
    space = 17 - (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17)

    # Vẽ các khu vực và hiển thị số lượng khu vực (slot)
    if a1 == 1:
        cv2.polylines(frame, [np.array(areas[0], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('1'), (323, 29), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[0], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('1'), (323, 29), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a2 == 1:
        cv2.polylines(frame, [np.array(areas[1], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('2'), (380, 29), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[1], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('2'), (380, 29), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a3 == 1:
        cv2.polylines(frame, [np.array(areas[2], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('3'), (446, 29), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[2], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('3'), (446, 29), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a4 == 1:
        cv2.polylines(frame, [np.array(areas[3], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('4'), (510, 29), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[3], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('4'), (510, 29), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a5 == 1:
        cv2.polylines(frame, [np.array(areas[4], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('5'), (580, 29), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[4], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('5'), (580, 29), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a6 == 1:
        cv2.polylines(frame, [np.array(areas[5], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('6'), (644, 29), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[5], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('6'), (644, 29), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a7 == 1:
        cv2.polylines(frame, [np.array(areas[6], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('7'), (720, 39), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[6], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('7'), (720, 39), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a8 == 1:
        cv2.polylines(frame, [np.array(areas[7], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('8'), (300,191), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[7], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('8'), (300,191), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a9 == 1:
        cv2.polylines(frame, [np.array(areas[8], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('9'), (398,198), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[8], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('9'), (398,198), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a10 == 1:
        cv2.polylines(frame, [np.array(areas[9], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('10'), (478,200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[9], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('10'), (478,200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a11 == 1:
        cv2.polylines(frame, [np.array(areas[10], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('11'), (572,206), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[10], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('11'), (572,206), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a12 == 1:
        cv2.polylines(frame, [np.array(areas[11], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('12'), (666,210), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[11], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('12'), (666,210), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if a13 == 1:
        cv2.polylines(frame, [np.array(areas[12], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('13'), (769,228), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[12], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('13'), (769,228), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)        
    if a14 == 1:
        cv2.polylines(frame, [np.array(areas[13], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('14'), (854,237), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[13], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('14'), (854,237), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)          
    if a15 == 1:
        cv2.polylines(frame, [np.array(areas[14], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('15'), (290,258), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[14], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('15'), (290,258), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)         
    if a16 == 1:
        cv2.polylines(frame, [np.array(areas[15], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('16'), (382,259), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[15], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('16'), (382,259), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)        
    if a17 == 1:
        cv2.polylines(frame, [np.array(areas[16], np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, str('17'), (544,267), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame, [np.array(areas[16], np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str('17'), (544,267), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)           


    # Thêm nền màu đen
    cv2.rectangle(frame, (15, 48), (265, 95), (0, 0, 0), -1)

    # Hiển thị văn bản
    cv2.putText(frame, f"emty slot: {space}", (28, 78), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


    
    cv2.imshow("Parking Car", frame)

    key = cv2.waitKey(1)

    if key == 32:  # Nhấn phím Space tua video qua 10 phút
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        new_time = current_time + 600000 
        cap.set(cv2.CAP_PROP_POS_MSEC, new_time)

    if key & 0xFF == 27:  # Nhấn phím Esc để thoát
        break

 
cap.release()
cv2.destroyAllWindows()
