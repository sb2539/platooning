import numpy as np
import cv2
import IPython
import time



min_confidence = 0.1
weight_file = 'rc-yolo_final.weights'
cfg_file = 'rc-yolo.cfg'
name_file = 'classes.names'



# file_name = 'cabc30fc-e7726578.mp4'

# Load Yolo
net = cv2.dnn.readNet(weight_file, cfg_file)

classes = []
with open(name_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]  # coco.names 파일의 개행 문자 제거 후 classes 리스트에 넣음
print(classes)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def writeFrame(img):
    # use global variable, writer
    global writer
    height, width = img.shape[:2]
    print(height, width)
    if writer is None and output_name is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')              # 영상 코덱 mjpg 형태로 변경
        writer = cv2.VideoWriter(output_name, fourcc, 24, (width, height), True)  # 영상 저장(영상 이름, 코덱, 프래임, 사이즈(width, height))
    if writer is not None:
        writer.write(img)


frame_count = 0
# initialize the video writer
writer = None
output_name = 'output_car_tracking.avi'

detected = False
frame_mode = 'Detection'
elapsed_time = 0
margin = 120
tracker = cv2.TrackerKCF_create()
trackers = cv2.MultiTracker_create()

detected_width = 0
vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

while True:
    start_time = time.time()
    frame_count += 1
    ret, frame = vs.read()
    if frame is None:
        print('### No more frame ###')
        break
    IPython.display.clear_output(wait=True)
    height, width, channedls = frame.shape

    class_ids = []
    confidences = []
    boxes = []
    rr = []
    # Region of Interest
    roi_left = int(0.2 * width)
    roi_right = int(0.8 * width)

    if frame_mode == 'Tracking':
        #frame_mode = 'Tracking'
        (success, tracking_boxes) = trackers.update(frame)
        tracking_box = tracking_boxes[0]
        tx = int(tracking_box[0])
        ty = int(tracking_box[1])
        tw = int(tracking_box[2])
        th = int(tracking_box[3])
        roi = frame[ty-margin:ty+th+margin, tx-margin:tx+tw+margin]
        roi_width, roi_height = roi.shape[:2]
        try :
            blob = cv2.dnn.blobFromImage(roi, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        except Exception as e :
            print(str(e))

        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if (confidence > min_confidence) and (class_id == 0):
                    # Object detected
                    center_x = int(detection[0] * roi_width)
                    center_y = int(detection[1] * roi_height)
                    w = int(detection[2] * roi_width)
                    h = int(detection[3] * roi_height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
        if len(boxes):
            boxes.sort(key=lambda x: x[2], reverse=True)
            box = boxes[0]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            roi_x = tx-margin + x
            roi_y = ty-margin + y
            distance_width = w - detected_width
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + w, roi_y + h), (0, 255, 255), 1)
            label = 'Initial Width : ' + str(detected_width) + ', Current Width : ' + str(w) + ', Distance : ' + str(w-detected_width)
            print(box, label)
            if abs(distance_width) > 5:
                if distance_width < 40:
                    cv2.putText(frame, 'Speed up', (30, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 5)
                elif distance_width < 70:
                    cv2.putText(frame, 'Slow down', (30, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 128, 255), 5)
                else:
                    cv2.putText(frame, 'Be Careful', (30, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)
            cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), (255, 255, 0), 1)
        cv2.rectangle(frame, (tx-margin, ty-margin), (tx+tw+margin, ty+th+margin), (255, 0, 0), 1)

    elif frame_mode == 'Detection':               # detection 모드
        # Detecting objects
        # https://docs.opencv.org/master/d6/d0f/group__dnn.html
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if (confidence > 0.1) and (class_id == 0):
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)




        indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
        font = cv2.FONT_HERSHEY_COMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                # Eliminate Small object(<50)
                if (w > 50) and (x > roi_left) and (x < roi_right):            # 너비가 50 이상, 정의한 범위 내에 속하는 경우 boxes의 좌표 전달 및 detect 사각형 출력
                    selected = boxes[i]
                    detected_width = w
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 5)
                    rr = selected
                    print(rr)
        if len(rr) != 0:                        # 감지된 차량이 없으면 tracking 모드로 넘어가지 않음
            trackers.add(tracker, frame, tuple(rr))
            frame_mode = 'Tracking'


    cv2.imshow('FRAME',frame)
    cv2.waitKey(1)
    writeFrame(frame)
    frame_time = time.time() - start_time
    elapsed_time += frame_time
    print("[{}] Frame {} time {}".format(frame_mode, frame_count, frame_time))

print("Elapsed time {}".format(elapsed_time))
vs.release()
