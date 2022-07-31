import numpy as np
import cv2
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from threading import Thread
from helper import draw_text, SpecialBox, Depth
start = True
dataset = None
cfg = None
weight = None
video_source_index = None
# Constant
base_line = 120
focal_length = 3
mapMMToPixel = 375
listPort = []
constant = base_line * focal_length * mapMMToPixel * 0.48
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self, cameraIndex, dataset, cfg, weight, **kwargs):
        self.dataset = dataset
        self.cfg = cfg
        self.weight = weight
        self.cameraIndex = cameraIndex
        super(Worker1, self).__init__(**kwargs)

    def run(self):
        self.ThreadActive = True
        self.camera = cv2.VideoCapture(self.cameraIndex)

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        h, w = None, None
        with open(self.dataset) as f:
            # Getting labels reading every line
            # and putting them into the list
            labels = [line.strip() for line in f]

            # print('List with labels names:')
            # print(labels)

        # Loading trained YOLO Objects Detector with the help of 'dnn' library from OpenCV
        network = cv2.dnn.readNetFromDarknet(self.cfg,
                                             self.weight)

        # Getting list with names of all layers from YOLO v3 network
        layers_names_all = network.getLayerNames()

        # print(layers_names_all)

        # Getting only output layers' names that we need from YOLO v3 algorithm
        # with function that returns indexes of layers with unconnected outputs
        layers_names_output = \
            [layers_names_all[i - 1]
                for i in network.getUnconnectedOutLayers()]

        # print()
        # print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

        # Set min prob to eliminate weak prediction
        probability_minimum = 0.5

        # Setting threshold for filtering weak bounding boxes with non-maximum suppression
        threshold = 0.3

        # Generating colours for representing every detected object
        # with function randint(low, high=None, size=None, dtype='l')
        colours = np.random.randint(
            0, 255, size=(len(labels), 3), dtype='uint8')
        while self.ThreadActive:
            # Capturing frame-by-frame from self.camera
            _, frame = self.camera.read()

            # frame = cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            # Getting spatial dimensions of the frame we do it only once from the very beginning
            # all other frames have the same dimension
            if w is None or h is None:
                # Slicing from tuple only first two elements
                h, w = frame.shape[:2]

            # Getting blob from current frame using method blobFromImage provided by opencv
            # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob from current
            # frame after mean subtraction, normalizing, and RB channels swapping
            # Resulted shape has number of frames, number of channels, width and height
            # eg.:
            # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)

            # Implementing forward pass with our blob and only through output layers
            # Calculating at the same time, needed time for forward pass
            network.setInput(blob)  # setting blob as input to the network
            start = time.time()
            output_from_network = network.forward(layers_names_output)
            

            # Preparing lists for detected bounding boxes, obtained confidences and class's number
            bounding_boxes = []
            confidences = []
            classIDs = []

            # Y_CENTER_ARR = []
            specBox = []
            # Going through all output layers after feed forward pass
            for result in output_from_network:
                # Going through all detections from current output layer
                for detected_objects in result:
                    # Getting 80 classes' probabilities for current detected object
                    scores = detected_objects[5:]
                    # Getting index of the class with the maximum value of probability
                    class_current = np.argmax(scores)
                    # Getting value of probability for defined class
                    confidence_current = scores[class_current]

                    # # Every 'detected_objects' numpy array has first 4 numbers with
                    # # bounding box coordinates and rest 80 with probabilities
                    # # for every class
                    # print(detected_objects.shape)  # (85,)

                    # Eliminating weak predictions with minimum probability
                    if confidence_current > probability_minimum:
                        # Scaling bounding box coordinates to the initial frame size
                        # YOLO data format keeps coordinates for center of bounding box
                        # and its current width and height
                        # That is why we can just multiply them elementwise
                        # to the width and height
                        # of the original frame and in this way get coordinates for center
                        # of bounding box, its width and height for original frame
                        box_current = detected_objects[0:4] * \
                            np.array([w, h, w, h])

                        # Now, from YOLO data format, we can get top left corner coordinates that are x_min and y_min
                        x_center, y_center, box_width, box_height = box_current
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))

                        bounding_boxes.append(
                            [x_min, y_min, int(box_width), int(box_height)])

                        confidences.append(float(confidence_current))
                        classIDs.append(class_current)

            # Implementing non-maximum suppression of given bounding boxes
            # With this technique we exclude some of bounding boxes if their
            # corresponding confidences are low or there is another
            # bounding box for this region with higher confidence

            # It is needed to make sure that data type of the boxes is 'int'
            # and data type of the confidences is 'float'
            # https://github.com/opencv/opencv/issues/12789
            results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                       probability_minimum, threshold)
            if len(results) > 0:
                for i in results.flatten():
                    specBox.append(SpecialBox(
                        bounding_boxes[i][0] + (bounding_boxes[i][2] / 2), bounding_boxes[i][1] + (bounding_boxes[i][3] / 2), bounding_boxes[i][2] * bounding_boxes[i][3], int(classIDs[i]), bounding_boxes[i][2], bounding_boxes[i][3]))

            # Số bounding box
            cv2.putText(frame, '{}'.format(len(specBox)), (int(w/2), int(h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, 2)
            specBox.sort(key=lambda x: (x.y_center, x.x_center))
            coupleBox = []
            for i, box in enumerate(specBox):
                id = i + 1
                while id < len(specBox):
                    if (specBox[id].y_center - box.y_center) < 20 and specBox[id].classIDs == box.classIDs \
                            and ((specBox[id].x_center >= int(FRAME_WIDTH/2) and box.x_center <= int(FRAME_WIDTH/2)) or (specBox[id].x_center <= int(FRAME_WIDTH/2) and box.x_center >= int(FRAME_WIDTH/2))):
                        if specBox[id].x_center >= int(FRAME_WIDTH/2):
                            disparity = box.x_center - \
                                (specBox[id].x_center - int(FRAME_WIDTH/2))
                            depth = (constant) / (disparity*1000)
                            coupleBox.append(
                                Depth(depth, int(box.x_center - (box.box_width / 2)), int(box.y_center - (box.box_height / 2)),
                                      box.box_width, box.box_height,
                                      box.classIDs))
                        else:
                            disparity = specBox[id].x_center - \
                                (box.x_center - int(FRAME_WIDTH/2))
                            depth = (constant) / (disparity*1000)
                            coupleBox.append(
                                Depth(depth, int(specBox[id].x_center - (specBox[id].box_width / 2)), int(specBox[id].y_center - (specBox[id].box_height / 2)),
                                      specBox[id].box_width, specBox[id].box_height,
                                      box.classIDs))
                        break
                    elif specBox[id].y_center - box.y_center >= 10:
                        break
                    id = id + 1
            coupleBox = list(set(coupleBox))
            end = time.time()

            # Showing spent time for single current frame
            print('Current frame took {:.5f} seconds'.format(end - start))
            if len(coupleBox) > 0:
                for i, box in enumerate(coupleBox):
                    draw_text(frame, '{}: {:.2f} m'.format(
                        labels[int(box.classIDs)], abs(box.depth)), font_scale=2 , pos=(box.x_min, box.y_min))
                    # boxWidth = (box.box_width * abs(box.depth) * 1.7)/(focal_length*mapMMToPixel)
                    # boxHeight = (box.box_height * abs(box.depth) * 1.7) / \
                    #     (focal_length*mapMMToPixel)

                    # cv2.putText(frame, 'W: {:.2f} m'.format(boxWidth), (int(box.x_min + (box.box_width / 2)), box.y_min + box.box_height),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # cv2.putText(frame, 'H: {:.2f} m'.format(boxHeight), (box.x_min + box.box_width, int(box.y_min + (box.box_height / 2))),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Non-maximum suppression

            # Checking if there is at least one detected object
            # after non-maximum suppression
            if len(results) > 0:
                # Going through indexes of results
                for i in results.flatten():
                    # Getting current bounding box coordinates, its width and height
                    x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                    box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                    # Preparing colour for current bounding box and converting from numpy array to list
                    colour_box_current = colours[classIDs[i]].tolist()

                    # Drawing bounding box on the original current frame
                    cv2.rectangle(frame, (x_min, y_min),
                                  (x_min + box_width, y_min + box_height),
                                  colour_box_current, 2)

                    # Preparing text with label and confidence for current bounding box
                    text_box_current = '{}: {:.4f}'.format(labels[int(classIDs[i])],
                                                           confidences[i])

                    # Putting text with label and confidence on the original image
                    cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
                    # cv2.putText(frame, 'disparity:{}'.format(disparity[i]), (x_min, y_min - 20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
                    # cv2.putText(frame, 'area:{}'.format(box_width*box_height), (x_min, y_min - 40),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
            Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # FlippedImage = cv2.flip(Image, 1)
            ConvertToQtFormat = QImage(
                Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(
                FRAME_WIDTH, FRAME_HEIGHT, Qt.KeepAspectRatio)
            self.ImageUpdate.emit(Pic)
    def stop(self):
        self.ThreadActive = False
        self.camera.release()
        self.quit()
