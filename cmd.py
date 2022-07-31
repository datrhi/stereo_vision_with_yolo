import numpy as np
import cv2
import time
print(cv2.__version__)


# Reading stream video from camera
camera = cv2.VideoCapture(0)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

h, w = None, None

# Constant
base_line = 120
focal_length = 3
mapMMToPixel = 375

constant = base_line * focal_length * mapMMToPixel * 0.48


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness)

    return text_size


class SpecialBox:
    def __init__(self, x_center, y_center, area, classIDs, box_width, box_height):
        self.x_center = x_center
        self.y_center = y_center
        self.area = area
        self.classIDs = classIDs
        # for cal width, height
        self.box_width = box_width
        self.box_height = box_height


class Depth:
    def __init__(self, depth, x_min, y_min, box_width, box_height, classIDs):
        self.depth = depth
        self.x_min = x_min
        self.y_min = y_min
        self.classIDs = classIDs
        self.box_width = box_width
        self.box_height = box_height

    def __str__(self):
        return '(\nDepth: {},\nX: {},\nY: {},\nclassIDs: {}\n)\n'.format(self.depth, self.x_min, self.y_min, self.classIDs)

    def __eq__(self, other):
        return self.classIDs == other.classIDs and abs(self.x_min - other.x_min) < 40 and \
            abs(self.y_min - other.y_min) < 40 and abs(self.depth -
                                                       other.depth) < 0.2*self.depth

    def __hash__(self):
        return hash(('classIDs', self.classIDs))


# Loading dataset (coco)
with open('yolo-coco-data/coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]

# print('List with labels names:')
# print(labels)

# Loading trained YOLO Objects Detector with the help of 'dnn' library from OpenCV
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov4.cfg',
                                     'yolo-coco-data/yolov4.weights')

# Getting list with names of all layers from YOLO v3 network
layers_names_all = network.getLayerNames()

# print(layers_names_all)

# Getting only output layers' names that we need from YOLO v3 algorithm
# with function that returns indexes of layers with unconnected outputs
layers_names_output = \
    [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

# print()
# print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Set min prob to eliminate weak prediction
probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes with non-maximum suppression
threshold = 0.3

# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


# Defining loop for catching frames
while True:
    # Capturing frame-by-frame from camera
    _, frame = camera.read()

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
    end = time.time()

    # Showing spent time for single current frame
    # print('Current frame took {:.5f} seconds'.format(end - start))

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
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

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
    if len(coupleBox) > 0:
        for i, box in enumerate(coupleBox):
            draw_text(frame, '{}: {:.2f} m'.format(
                labels[int(box.classIDs)], abs(box.depth)), font_scale=2, pos=(int(box.x_min + (box.box_width / 2)), box.y_min))
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

    cv2.namedWindow('Depth with yolo', cv2.WINDOW_NORMAL)
    # cv2.imshow('Depth with yolo', frame[0:FRAME_HEIGHT, 0:int(FRAME_WIDTH/2)])
    cv2.imshow('Depth with yolo', frame)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Releasing camera
camera.release()
# Destroying all opened OpenCV windows
cv2.destroyAllWindows()
