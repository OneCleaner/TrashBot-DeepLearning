#!/usr/bin/python
# Matthew Dunlop, August 2018
# https://github.com/mdunlop2
#
# Contact:
# https://www.linkedin.com/in/mdunlop2/

'''
## Tflite Object Detection
Currently, `python-tflite.py` supports using Mobilenet-V1 SSD models trained using Cloud Annotations.

Note: to find a list of all models trained do:
```
cacli list
```

To use a custom model, perform
```
cacli download <model_name>
```
For example, if the downloaded files were saved to `/path/to/<model_name>` :
* Our tflite model is stored in `<model_name>/model_android/model.tflite`
* Our tflite anchors file is stored in `<model_name>/model_android/anchors.json`
* Our tflite labels file is stored in `<model_name>/model_android/labels.json`

Change directory to the root of this git.
```
cd examples/tflite_interpreter/basic/
python python-tflite.py --MODEL_DIR /path/to/<model_name>/model_android
```
This script calls the tflite model interpreter for inference on all .jpg files inside the directory `PATH_TO_TEST_IMAGES_DIR`.

Similary the output .jpg files are storesd in `PATH_TO_OUTPUT_DIR`.

We can also specify the minimum confidence (score) for a given detection box to be displayed with `MINIMUM_CONFIDENCE`.

Finally:
```
python python-tflite.py \
--MODEL_DIR /path/to/<model_name>/model_android \
--PATH_TO_TEST_IMAGES_DIR /path/to/test/images \
--PATH_TO_OUTPUT_DIR /path/to/output/images \
--MINIMUM_CONFIDENCE 0.01

```
'''
import glob
import os
import cv2 as cv
import argparse
import sys

# from examples.tflite_interpreter.basic.utils import visualization_utils as vis_util
# from examples.tflite_interpreter.basic.utils import cacli_models as models
from utils import visualization_utils as vis_util
from utils import cacli_models as models



MINIMUM_CONFIDENCE = 0.7
MODEL_DIR = "model"

MODEL_PATH = MODEL_DIR + "/model.tflite"
MODEL_ANCHOR_PATH = MODEL_DIR + "/anchors.json"
MODEL_LABEL_PATH = MODEL_DIR + "/labels.json"

if __name__ == "__main__":

    # Load model and allocate tensors
    model_interpreter = models.initiate_tflite_model(MODEL_PATH)
    # Load mobilenet-v1 anchor points
    anchor_points = models.json_to_numpy(MODEL_ANCHOR_PATH)

    # Load Category Index
    label_list = models.json_to_numpy(MODEL_LABEL_PATH)

    category_index = { i : {"name" : label_list[i]} for i in list(range(len(label_list))) }
    cam = cv.VideoCapture(1)
    while True:
        grabbed, frame = cam.read()
        cv.imwrite('gieri.png', frame)


        models.detect_objects(model_interpreter, "gieri.png", category_index, anchor_points, MINIMUM_CONFIDENCE, SAVE_DIR="out")

        image = cv.imread("out/gieri.png")
        image = image[0:558, 124:867] #crop per evitare bordi bianchi

        cv.imshow("image", image)
        if (cv.waitKey(1) == 27) | (cv.waitKey(1) & 0xFF == ord('q')):
          break
    cv.destroyAllWindows()
    sys.exit()
