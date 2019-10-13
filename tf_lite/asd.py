import numpy as np
import tensorflow as tf
import cv2
import sys
from PIL import Image
from skimage import img_as_ubyte
# import calass and function from TF object detection API
from object_detector_detection_api import ObjectDetectorDetectionAPI, \
    PATH_TO_LABELS, NUM_CLASSES



from utils import label_map_util

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

from utils import visualization_utils as vis_util

class ObjectDetectorLite(ObjectDetectorDetectionAPI):
    def __init__(self, model_path='detect.tflite'):
        """
            Builds Tensorflow graph, load model and labels
        """

        # Load lebel_map
        self._load_label(PATH_TO_LABELS, NUM_CLASSES, use_disp_name=True)

        # Define lite graph and Load Tensorflow Lite model into memory
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect(self, image, threshold=0.1):
        """
            Predicts person in frame with threshold level of confidence
            Returns list with top-left, bottom-right coordinates and list with labels, confidence in %
        """

        # Resize and normalize image for network input
        frame = cv2.resize(image, (300, 300))
        frame = np.expand_dims(frame, axis=0)
        frame = (2.0 / 255.0) * frame - 1.0
        # frame = frame.astype('float32')
        frame = frame.astype(np.uint8)
        # run model
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()

        # get results
        boxes = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(
            self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(
            self.output_details[2]['index'])
        num = self.interpreter.get_tensor(
            self.output_details[3]['index'])

        # Find detected boxes coordinates
        return self._boxes_coordinates(image,
                            np.squeeze(boxes[0]),
                            np.squeeze(classes[0]+1).astype(np.int32),
                            np.squeeze(scores[0]),
                            min_score_thresh=threshold)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

if __name__ == "__main__":

    cam = cv2.VideoCapture(0)

    while True:
        detector = ObjectDetectorLite()
        grabbed, frame = cam.read()
        cv2.imwrite('gieri.png', frame)
        image_path = 'gieri.png'

        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)
        image_np = img_as_ubyte(image_np)
        img = detector.detect(image_np)

        for obj in img:
            print(('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
                        format(obj[0], obj[1], obj[3], obj[2])))
            cv2.rectangle(image_np, obj[0], obj[1], (0, 255, 0), 2)
            cv2.putText(image_np, '{}: {:.2f}'.format(obj[3], obj[2]),
            (obj[0][0], obj[0][1] - 5),
            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        image = Image.fromarray(image_np);
        image.save('gieri.png')

        cv2.imshow('image', cv2.imread('gieri.png'))
        if (cv2.waitKey(1) == 27) | (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()
    sys.exit()
