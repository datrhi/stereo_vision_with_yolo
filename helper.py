import cv2

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