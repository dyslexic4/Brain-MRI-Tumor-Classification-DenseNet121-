# %% [BLOCK 2] PREDICT TUMOR TYPE FROM ONE IMAGE
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image

JOBLIB_PATH = "brain_mri_densenet.joblib"
IMAGE_PATH  = "/path/to/your_mri_image.jpg"     # <-- change this


def load_bundle(path=JOBLIB_PATH):
    b = joblib.load(path)
    model = tf.keras.models.model_from_json(b["architecture"])
    model.set_weights(b["weights"])
    return model, b["class_names"], tuple(b["img_size"])


def predict_image(image_path, model, class_names, img_size):
    img = Image.open(image_path).convert("RGB").resize(img_size)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)   # raw 0-255, model rescales
    proba = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(proba))

    print(f"\nImage      : {image_path}")
    print(f"Prediction : {class_names[idx].upper()}  ({proba[idx]*100:.1f}% confidence)\n")
    for name, p in zip(class_names, proba):
        print(f"  {name:>12}  {p*100:5.1f}%  {'█' * int(p * 30)}")
    if proba[idx] < 0.7:
        print("\nNote: confidence is low, treat this prediction with caution.")
    return class_names[idx], proba


if __name__ == "__main__":
    model, class_names, img_size = load_bundle()
    predict_image(IMAGE_PATH, model, class_names, img_size)