# -*- coding: utf-8 -*-
"""
YOLOv3 detection module compatible with TF 2.x running in TF1 graph mode.
- Disables eager execution to keep K.placeholder & sess.run workflow.
- Works with legacy yolo_eval / detect_image code paths.
"""

import os
import colorsys
import numpy as np
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw

import tensorflow as tf

# IMPORTANT: run in TF1-style graph mode so placeholders work
tf.compat.v1.disable_eager_execution()

from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from Car_Detection_TF.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from Car_Detection_TF.yolo3.utils import letterbox_image


class YOLO(object):
    _defaults = {
        "model_path": "Car_Detection_TF/model_data/yolo.h5",
        "anchors_path": "Car_Detection_TF/model_data/yolo_anchors.txt",
        "classes_path": "Car_Detection_TF/model_data/coco_classes.txt",
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        return cls._defaults.get(n, f"Unrecognized attribute '{n}'")

    def __init__(self, **kwargs):
        # update defaults
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

        # TF1-style session (valid since we disabled eager)
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        with open(os.path.expanduser(self.classes_path)) as f:
            class_names = [c.strip() for c in f.readlines()]
        return class_names

    def _get_anchors(self):
        with open(os.path.expanduser(self.anchors_path)) as f:
            anchors = [float(x) for x in f.readline().split(",")]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(".h5"), "Keras model or weights must be a .h5 file."

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = (num_anchors == 6)  # 6 anchors => Tiny YOLO

        # Load model (or construct & load weights)
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except Exception:
            # Fallback: build architecture then load weights
            self.yolo_model = (
                tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes)
                if is_tiny_version
                else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            )
            self.yolo_model.load_weights(self.model_path)

        print(f"✅ YOLO model loaded from {model_path}")

        # Colors for classes
        hsv_tuples = [(x / len(self.class_names), 1.0, 1.0) for x in range(len(self.class_names))]
        self.colors = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*x)) for x in hsv_tuples]
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        # Placeholders/tensors for eval
        self.input_image_shape = K.placeholder(shape=(2,), dtype='float32')

        boxes, scores, classes = yolo_eval(
            self.yolo_model.output,
            self.anchors,
            len(self.class_names),
            self.input_image_shape,
            score_threshold=self.score,
            iou_threshold=self.iou,
        )
        return boxes, scores, classes

    # ---------------------------------------------------------
    # IOU helpers
    # ---------------------------------------------------------
    def intersection_over_union(self, boxA, boxB, threshold=0.5):
        xA = max(boxA[1], boxB[1])
        yA = max(boxA[0], boxB[0])
        xB = min(boxA[3], boxB[3])
        yB = min(boxA[2], boxB[2])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return False

        boxAArea = (boxA[3] - boxA[1]) * (boxA[2] - boxA[0])
        boxBArea = (boxB[3] - boxB[1]) * (boxB[2] - boxB[0])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou >= threshold

    def filterBoxes(self, t, c, out_boxes, out_classes, out_scores, same=False):
        index = []
        for i, Truck in enumerate(t):
            for j, Car in enumerate(c):
                if same and i == j:
                    continue
                if self.intersection_over_union(Truck[0], Car[0]):
                    if Truck[1] > Car[1]:
                        index.append(Car[2])
                    else:
                        index.append(Truck[2])
                    break
        out_classes = np.delete(out_classes, index, 0)
        out_boxes = np.delete(out_boxes, index, 0)
        out_scores = np.delete(out_scores, index, 0)
        return out_boxes, out_classes, out_scores

    # ---------------------------------------------------------
    # Detection on a single PIL image
    # ---------------------------------------------------------
    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0
            assert self.model_image_size[1] % 32 == 0
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_size = (
                image.width - (image.width % 32),
                image.height - (image.height % 32),
            )
            boxed_image = letterbox_image(image, new_size)

        image_data = np.array(boxed_image, dtype="float32") / 255.0
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0,
            },
        )

        # Draw results
        font = ImageFont.truetype(
            font="Car_Detection_TF/font/FiraMono-Medium.otf",
            size=max(10, int(3e-2 * image.size[1])),
        )
        thickness = (image.size[0] + image.size[1]) // 300
        ret = []
        t, c, b = [], [], []
        ind = []

        for i, cls in reversed(list(enumerate(out_classes))):
            if (out_boxes[i][2] - out_boxes[i][0]) * (out_boxes[i][3] - out_boxes[i][1]) > 0.75 * 480 * 360:
                ind.append(i)
            name = self.class_names[cls]
            if name == "car":
                c.append([out_boxes[i], out_scores[i], i])
            elif name == "truck":
                t.append([out_boxes[i], out_scores[i], i])
            elif name == "bus":
                b.append([out_boxes[i], out_scores[i], i])

        out_classes = np.delete(out_classes, ind, 0)
        out_boxes = np.delete(out_boxes, ind, 0)
        out_scores = np.delete(out_scores, ind, 0)

        out_boxes, out_classes, out_scores = self.filterBoxes(t, c, out_boxes, out_classes, out_scores)
        out_boxes, out_classes, out_scores = self.filterBoxes(t, b, out_boxes, out_classes, out_scores)
        out_boxes, out_classes, out_scores = self.filterBoxes(b, c, out_boxes, out_classes, out_scores)
        out_boxes, out_classes, out_scores = self.filterBoxes(c, c, out_boxes, out_classes, out_scores, same=True)

        draw = ImageDraw.Draw(image)
        for i, cls in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[cls]
            if predicted_class not in ["car", "truck", "bus"]:
                continue

            box = out_boxes[i]
            score = out_scores[i]
            label = f"{predicted_class} {score:.2f}"

            top, left, bottom, right = [int(v) for v in box]
            text_origin = np.array([left, max(top - 15, 0)])
            ret.append([predicted_class, left, right, top, bottom, score])

            for j in range(thickness):
                draw.rectangle(
                    [left + j, top + j, right - j, bottom - j],
                    outline=self.colors[cls],
                )
            draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)

        del draw
        print(f"Processed frame in {timer() - start:.2f}s with {len(ret)} detections.")
        return image, ret

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open video source")

    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, video_fps, video_size)

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        image = Image.fromarray(frame)
        image, _ = yolo.detect_image(image)
        frame = np.asarray(image)

        if out:
            out.write(frame)

        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    if out:
        out.release()
    yolo.close_session()
    cv2.destroyAllWindows()
