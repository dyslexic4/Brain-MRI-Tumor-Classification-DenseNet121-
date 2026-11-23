# Brain MRI Tumor Classification (DenseNet121)

Four-class brain MRI classifier built with transfer learning on DenseNet121, with a Streamlit web app for single-image prediction.

**Classes:** `notumor` · `glioma` · `meningioma` · `pituitary`

> ⚠️ **Disclaimer:** This is a research and educational project. It is **not** a medical device and must not be used for diagnosis or treatment decisions. Do not upload real patient scans to any hosted version of the app.

## How it works

- **Backbone:** DenseNet121 pretrained on ImageNet, 224×224 RGB input.
- **Head:** Global average pooling, then Dense(512) and Dense(256) layers with batch norm and dropout, then a 4-way softmax.
- **Training:** two phases.
  1. Train the head only, with the backbone frozen (LR 1e-3).
  2. Fine-tune the top 50 backbone layers at a low learning rate (LR 1e-5), keeping BatchNorm layers frozen.
- **Imbalance handling:** balanced class weights.
- **Augmentation:** horizontal flip, small rotation, zoom and contrast, applied only during training.
- **Saved model:** a single `.joblib` bundle (architecture, weights, class names, training history), plus a `.keras` backup.

## Project structure

```
├── block1_train.py      # train and save the model
├── block2_predict.py    # predict the tumor type for one image
├── block3_stats.py      # test-set metrics, confusion matrix, Excel export
├── app.py               # Streamlit frontend
├── requirements.txt
└── README.md
```
₹
## Dataset

The scripts expect this folder layout (not included in this repo):

```
training/
├── notumor/
├── glioma/
├── meningioma/
└── pituitary/
testing/
├── notumor/
├── glioma/
├── meningioma/
└── pituitary/
```

Download a brain tumor MRI dataset with these four classes and respect its license terms. Then set `TRAIN_DIR` (block 1) and `TEST_DIR` (block 3) to your local paths.

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Train**
```bash
python block1_train.py
```
Saves `brain_mri_densenet.joblib` (and `brain_mri_densenet.keras`). Set `MAX_PER_CLASS` in the config to limit images per class for quick experiments.

**2. Predict on one image**

Set `IMAGE_PATH` in `block2_predict.py`, then:
```bash
python block2_predict.py
```

**3. Evaluate on the test set**
```bash
python block3_stats.py
```
Prints accuracy, per-class precision, recall and F1, macro AUC, and tumor vs. no-tumor sensitivity and specificity. Saves `confusion_matrix.png` and `brain_mri_results.xlsx`.

**4. Run the web app**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. `brain_mri_densenet.joblib` must be in the same folder.

## Results
Evaluated on 1600 test images

=======================================================
MULTICLASS (tumor type)
=======================================================
Accuracy      : 88.25%
Macro AUC-ROC : 0.9789

              precision    recall  f1-score   support

     notumor     0.9192    0.9950    0.9556       400
      glioma     0.9228    0.7775    0.8440       400
  meningioma     0.8532    0.7700    0.8095       400
   pituitary     0.8422    0.9875    0.9091       400

    accuracy                         0.8825      1600
   macro avg     0.8844    0.8825    0.8795      1600
weighted avg     0.8844    0.8825    0.8795      1600

Confusion matrix:
 [[398   1   0   1]
 [ 27 311  48   14]
 [ 8  25  308   59]
 [ 0  0  5   395]]
Missed tumors (FN)  : 35
Confusion matrix:
 [[ 398    2]
 [  35 1165]]
## Notes and limitations

- The validation split is a random 20% of the training folder. MRI datasets often contain near-duplicate slices from the same patient, so validation scores can be optimistic. Use the held-out test set for real numbers.
- Only load `.joblib` model files you created yourself. They are pickles and can execute arbitrary code if tampered with.
- Results depend on the dataset and have not been validated clinically.
