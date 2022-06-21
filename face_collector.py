import argparse
import cv2
import os
import sys
import numpy as np
from PIL import Image

from bing_images import bing
from mtcnn import MTCNN

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
                help="string to be searched")
ap.add_argument("-l", "--limit", required=False, type=int, default=100,
                help="(optional, default 100) number of images to download")
ap.add_argument("-a", "--alignment", required=False, type=str, default='n',
                help="(optional, default 'n') face alignment (y/n)")
ap.add_argument("-g", "--gray", required=False, type=str, default='n',
                help="(optional, default 'n') convert images to gray (y/n)")
ap.add_argument("-s", "--size", required=False,
                help="(optional) size to convert 'w, h'")
args = vars(ap.parse_args())


output_dir = "downloaded/" + args["query"]
extracted_dir = "extracted/" + args["query"]
images_list = []
resizing_lengths = (int(args["size"].split(",")[0]), int(args["size"].split(",")[1]))
detector = MTCNN()


def check_arguments():
    if args["alignment"] == "y" or args["alignment"] == "n":
        pass
    else:
        print("[ERROR] Face alignment argument should be 'y' or 'n'\n[INFO] Exiting...")
        sys.exit(0)

    if args["gray"] == "y" or args["gray"] == "n":
        pass
    else:
        print("[ERROR] Convert to gray argument should be 'y' or 'n' [INFO] Exiting...")
        sys.exit(0)


def delete_image(path):
    try:
        os.remove(path)
        print(path + " is deleted")
    except OSError as error:
        print("[ERROR] Image deleting error! " + str(error))


def delete_blurry_images():
    global output_dir
    output_list = os.listdir(output_dir)
    for image in output_list:
        image_path = output_dir + "/" + image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 100:
            delete_image(image_path)

    print("[INFO] Blurry images detection is finished")


def rotate_image(path, angle):
    try:
        img = Image.open(path)
        rotated = img.rotate(angle, expand=True)
        rotated.save(path)
    except OSError as error:
        print("[ERROR] Face rotating error! " + str(error))


def convert_to_gray(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(image_path, gray)
    except OSError as error:
        print("[ERROR] Color converting error! " + str(error))


def resize_image(image_path):
    # resizing_lengths = (96, 96)
    try:
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, resizing_lengths, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(image_path, resized_image)
    except OSError as error:
        print("[ERROR] Image resizing error! " + str(error))


def download_images():
    global images_list, output_dir

    try:
        print("Initializing to download..")
        bing.download_images(args["query"], int(args["limit"]), output_dir, pool_size=10)
        print("[INFO] Downloading is finished")

    except OSError as error:
        print("[ERROR] Download error! " + str(error))


def detect_faces(image):
    global extracted_dir
    faces_dictionary = {}  # {'path1':{'image': img, 'angle': 25}, 'path2':{'image': img, 'angle': 50}}
    image_full_name = image.split("/")[-1]
    image_name = image_full_name.split(".")[0]
    image_extension = image_full_name.split(".")[1]
    img = cv2.imread(image)
    i_h, i_w, i_c = img.shape
    if i_h >= 50 and i_w >= 50:
        detections = detector.detect_faces(img)
        if detections:
            i = 0
            for detection in detections:
                if detection["confidence"] >= 0.90:
                    x, y, w, h = detection["box"]
                    if w >= 50 and h >= 50:
                        try:
                            i += 1
                            x = x * 0.8 if x > 0 else x
                            x = int(x)
                            y = y * 0.8 if y > 0 else y
                            y = int(y)
                            width = (x + w) * 1.2 if (x + w) < i_w else (x + w)
                            width = int(width)
                            height = (y + h) * 1.2 if (y + h) < i_h else (y + h)
                            height = int(height)
                            detected_face = img[y:height, x:width]

                            keypoints = detection["keypoints"]
                            left_eye = keypoints["left_eye"]
                            right_eye = keypoints["right_eye"]
                            if right_eye[1] > left_eye[1]:
                                higher_eye = right_eye
                                lower_eye = left_eye
                            else:
                                higher_eye = left_eye
                                lower_eye = right_eye

                            third_point = (higher_eye[0], lower_eye[1])
                            angle = -100 * (higher_eye[1] - third_point[1]) / (lower_eye[0] - third_point[0])
                            face_path = f"{extracted_dir}/{image_name}_{str(i)}.{image_extension}"

                            faces_dictionary[face_path] = {'image': detected_face, 'angle': angle}
                        except OSError as error:
                            print("[ERROR] Face detection error! " + str(error))

    return faces_dictionary


def extract_faces():
    global images_list, output_dir, extracted_dir

    images_list = os.listdir(output_dir)

    if not os.path.exists(extracted_dir):
        try:
            os.mkdir(extracted_dir)
            print(extracted_dir + " is created")
        except OSError as error:
            print("[ERROR] Directory creation error! " + str(error))

    for image in images_list:
        image_path = output_dir + "/" + image

        if image_path.lower().endswith(".png") or image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
            faces_dictionary = detect_faces(image_path)
            faces_list = faces_dictionary.keys()
            if len(faces_list) > 0:
                for face in faces_list:
                    cv2.imwrite(face, faces_dictionary[face]["image"])
                    if args["alignment"] == "y":
                        angle = faces_dictionary[face]['angle']
                        rotate_image(face, angle)

                    if args["gray"] == "y":
                        convert_to_gray(face)

                    if args["size"] != "":
                        resize_image(face)

        elif image_path.lower().endswith(".gif"):
            delete_image(image_path)

        delete_image(image_path)

    print("[INFO] Face image extraction is finished")


def verify_faces():
    global output_dir
    extracted_list = os.listdir(output_dir)
    for face in extracted_list:
        face_path = extracted_dir + "/" + face
        img = cv2.imread(face_path)
        detections = detector.detect_faces(face_path)
        if detections:
            for detection in detections:
                if detection["confidence"] >= 0.90:
                    x, y, w, h = detection["box"]
                    width = x + w
                    height = y + h
                    detected_face = img[y:height, x:width]
                    cv2.imwrite(face_path, detected_face)


check_arguments()
download_images()
extract_faces()
delete_blurry_images()
verify_faces()
