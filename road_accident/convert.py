# convert.py — FINAL WORKING VERSION for YOLOv3 to Keras .h5 (Argus)
import numpy as np
from pathlib import Path
from keras.layers import Input
from Car_Detection_TF.yolo3.model import yolo_body


def get_classes(classes_path):
    with open(classes_path) as f:
        return [c.strip() for c in f.readlines() if c.strip()]


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = [float(x) for x in f.readline().split(',')]
    return np.array(anchors).reshape(-1, 2)


MODEL_DATA = Path("Car_Detection_TF/model_data")
WEIGHTS_PATH = MODEL_DATA / "yolov3.weights"
OUTPUT_PATH = MODEL_DATA / "yolo.h5"
ANCHORS_PATH = MODEL_DATA / "yolo_anchors.txt"
CLASSES_PATH = MODEL_DATA / "coco_classes.txt"


def load_darknet_weights(model, weights_path):
    with open(weights_path, 'rb') as wf:
        _ = np.fromfile(wf, dtype=np.int32, count=5)  # header
        print("📦 Darknet header skipped.")

        for i, layer in enumerate(model.layers):
            if not layer.name.startswith('conv2d'):
                continue

            # Find if this conv layer has BN
            bn_layer = None
            for j in range(i + 1, len(model.layers)):
                if model.layers[j].name.startswith('batch_normalization'):
                    bn_layer = model.layers[j]
                    break
                if model.layers[j].name.startswith('conv2d'):
                    break

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]
            conv_shape = (filters, in_dim, size, size)

            # Load convolution weights
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            if conv_weights.size != np.product(conv_shape):
                print("⚠️ Reached unexpected end of weights file.")
                break
            conv_weights = conv_weights.reshape(conv_shape)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

            # Handle BN or bias
            if bn_layer is not None:
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                bn_weights = bn_weights.reshape((4, filters))
                bn_weights = bn_weights[[1, 0, 2, 3]]  # reorder to [gamma, beta, mean, var]
                layer.set_weights([conv_weights])
                bn_layer.set_weights(list(bn_weights))
            else:
                bias = np.fromfile(wf, dtype=np.float32, count=filters)
                layer.set_weights([conv_weights, bias])

        print("✅ All Darknet weights loaded successfully.")


def convert_yolo():
    anchors = get_anchors(str(ANCHORS_PATH))
    classes = get_classes(str(CLASSES_PATH))
    print(f"🔹 Anchors loaded: {len(anchors)} | Classes loaded: {len(classes)}")

    model = yolo_body(Input(shape=(None, None, 3)), len(anchors)//3, len(classes))
    print("🚀 Loading Darknet weights from:", WEIGHTS_PATH)
    load_darknet_weights(model, str(WEIGHTS_PATH))
    print("💾 Saving converted model to:", OUTPUT_PATH)
    model.save(str(OUTPUT_PATH))
    print("✅ Conversion complete! YOLOv3 -> yolo.h5")


if __name__ == "__main__":
    convert_yolo()