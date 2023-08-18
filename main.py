import cv2
import os
import base64
import requests
import pickle
import numpy as np

url = "http://localhost:8080/api/gethog"

classB = [
    "Audi",
    "Hyundai Creta",
    "Mahindra Scorpio",
    "Rolls Royce",
    "Swift",
    "Tata Safari",
    "Toyota Innova",
]


def img2HOG(img):
    v, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer)
    data = "image data," + str.split(str(img_str), "'")[1]
    response = requests.get(url, json={"img": data})

    return response.json()


def readData(path):
    response = []
    for sub in os.listdir(path):
        for fn in os.listdir(path + "/" + sub):
            img_file_name = path + "/" + sub + "/" + fn
            img = cv2.imread(img_file_name)
            res = img2HOG(img)
            hog = list(res["hog"])
            hog.append(classB.index(sub))
            response.append(hog)
    return response


def savePkl(filename, path):
    cars = readData(path)

    write_path = filename + ".pkl"
    pickle.dump(cars, open(write_path, "wb"))
    print("data preparation is done")


def loadPkl(filename):
    dataset = pickle.load(open(filename + ".pkl", "rb"))
    return dataset


train_dir = r"Cars Dataset\train"
savePkl("train_cars", train_dir)

test_dir = r"Cars Dataset\test"
savePkl("test_cars", test_dir)

dataset_train = loadPkl("train_cars")
print("Data train : ", len(dataset_train))
dataset_test = loadPkl("test_cars")
print("Data test : ", len(dataset_test))

train_arr = np.array(dataset_train)
x_train = train_arr[:, 0:-1]
y_train = train_arr[:, -1]

test_arr = np.array(dataset_test)
x_test = test_arr[:, 0:-1]
y_test = test_arr[:, -1]

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))

path_model = "carbrandmodel.pkl"
pickle.dump(clf, open(path_model, "wb"))
