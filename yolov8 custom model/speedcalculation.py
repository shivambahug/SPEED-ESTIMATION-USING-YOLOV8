import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import time
from math import dist

model = YOLO('best.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('veh2.mp4')

my_file = open("popo.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()
cy1 = 322
cy2 = 370
offset = 6

vh_down = {}
counter = []
vh_up = {}
counter1 = []

# Create an empty list to hold speed data
speed_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x4, y4, x5, y5, id = bbox
        cx = int(x4 + x5) // 2
        cy = int(y4 + y5) // 2

        cv2.rectangle(frame, (x4, y4), (x5, y5), (0, 255, 0), 2)
        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

        if cy1 < (cy + offset) and cy1 > (cy - offset):
            vh_down[id] = time.time()
        if id in vh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                elapsed_time = time.time() - vh_down[id]
                if counter.count(id) == 0:
                    counter.append(id)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x5, y5), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)

                    # Append speed data to the list
                    speed_data.append({"Vehicle ID": id, "Speed": int(a_speed_kh)})

        if cy2 < (cy + offset) and cy2 > (cy - offset):
            vh_up[id] = time.time()
        if id in vh_up:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                elapsed1_time = time.time() - vh_up[id]
                if counter1.count(id) == 0:
                    counter1.append(id)
                    distance1 = 10  # meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x5, y5), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)

                    # Append speed data to the list
                    speed_data.append({"Vehicle ID": id, "Speed": int(a_speed_kh1)})

    cv2.line(frame, (0, 322), (1019, 322), (255, 255, 255), 1)
    cv2.putText(frame, 'Line 1', (0, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (0, 370), (1019, 370), (255, 255, 255), 1)
    cv2.putText(frame, 'Line 2', (0, 368), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    d = len(counter)
    u = len(counter1)
    cv2.putText(frame, 'goingdown: ' + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, 'goingup: ' + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Create a DataFrame from the speed_data list
speed_df = pd.DataFrame(speed_data)

# Save the DataFrame to an Excel file
speed_df.to_excel("vehicle_speed.xlsx", index=False)