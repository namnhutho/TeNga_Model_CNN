import json
import os
import cv2
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Convolution2D,
    Activation,
    Dropout,
    GlobalAveragePooling2D,
)
from keras.applications import (
    InceptionV3,
    MobileNet,
    VGG16,
    DenseNet121,
    EfficientNetB7,
)
from numpy import argmax
from sklearn.metrics import confusion_matrix, accuracy_score

HEIGHT = 224
WIDTH = 224
data = []
# D:/Báo cáo niên luận/NLCN-cuoicung/code/data_test/data_framecut/
json_dir = "D:/NLCN-cuoicung/code/data_test/data_framecut/"
image_dir = "D:/NLCN-cuoicung/code/data_test/data_framecut/"
for json_file in os.listdir(json_dir):
    if json_file.endswith(".json"):
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, "r") as file:
            json_data = json.load(file)
            shapes = json_data.get("shapes", [])
            for shape_info in shapes:
                label = shape_info.get("label", "")
                if label:
                    image_filename = json_data["imagePath"]
                    image_path = os.path.join(image_dir, image_filename)
                    # Tiền xử lý hình ảnh
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (WIDTH, HEIGHT))
                    image = image.astype("float") / 255.0
                    data.append({"image": image, "label": label})

# Chuyển đổi dữ liệu thành dạng mảng NumPy
images = np.array([item["image"] for item in data])
labels = [item["label"] for item in data]

count1 = labels.count("dung")
print("Số lần xuất hiện của nhãn đứng:", count1)
count2 = labels.count("nga")
print("Số lần xuất hiện của nhãn ngã:", count2)

# Chuyển nhãn thành dạng số nguyên
label_encoder = LabelEncoder()
indexed_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
y = to_categorical(indexed_labels, num_classes)
trainX, testX, trainY, testY = train_test_split(
    images, y, test_size=0.2, random_state=42
)
base_model = MobileNet(
    input_shape=(HEIGHT, WIDTH, 3), include_top=False, weights="imagenet"
)
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(
    Dense(128, activation="relu")
)  # 128 đơn vị, có 128 neuron hoạt động trong quá trình tính toán.
model.add(Dropout(0.5))  # 0.5 thường cho phép mô hình duy trì hiệu suất tốt
model.add(Flatten(name="flatten"))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Huấn luyện mô hình
EPOCHS = 10 
BATCH_SIZE = 32 
history = model.fit(
    trainX,
    trainY,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=1,
)

test_predictions = model.predict(testX)
predicted_labels = np.argmax(test_predictions, axis=1)
predicted_one_hot = to_categorical(predicted_labels, 2)
accuracy = accuracy_score(testY, predicted_one_hot)
print(f"Accuracy: {accuracy * 100:.2f}%")
