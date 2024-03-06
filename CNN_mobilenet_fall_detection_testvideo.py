import cv2
import numpy as np
from keras.models import load_model

# file chính thức


def process_frame(frame, net, model, emotion_labels):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (224, 224), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_ITALIC

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            roi = frame[y : y + h, x : x + w]
            if roi.size > 0.5:
                roi = cv2.resize(roi, (WIDTH, HEIGHT))
                roi = roi.astype("float") / 255.0
                frame_prediction = model.predict(np.expand_dims(roi, axis=0))
                predicted_label_index = np.argmax(frame_prediction)
                predicted_label = emotion_labels[predicted_label_index]
                predicted_prob = frame_prediction[0][predicted_label_index] * 100
                text = f"{predicted_label} ({predicted_prob:.2f}%)"
                if predicted_label == "nga" and w > h:
                    cv2.putText(frame, text, (x, y - 5), font, 1, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                if predicted_label == "dung":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


yolo_config_path = "D:/NLCN-cuoicung/code/yolo/yolov4-tiny.cfg"
yolo_weights_path = "D:/NLCN-cuoicung/code/yolo/yolov4-tiny.weights"
# yolo_config_path = "D:/NLCN/yolo/yolov4.cfg"
# yolo_weights_path = "D:/NLCN/yolo/yolov4.weights"
model_path = "D:/NLCN-cuoicung/code/mobilenet_xy.h5"
emotion_labels = ["dung", "nga"]
WIDTH = 224
HEIGHT = 224

video_paths = [
    # "D:/NLCN/tenga/data/tenga_test_cut.mp4",
    # "D:/NLCN/tenga/data/te1.mp4",
    # "D:/NLCN/tenga/data/te_nga_cut_1.mp4",
    # "D:/NLCN/tenga/data/tenga1.mp4",

    "D:/NLCN-cuoicung/code/data_test/tenga.mp4"
]

net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)
model = load_model(model_path)

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    # fps = 24

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, net, model, emotion_labels)

        cv2.imshow("Video", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

cv2.destroyAllWindows()
