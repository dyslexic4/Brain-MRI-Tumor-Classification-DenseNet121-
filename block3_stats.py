# %% [BLOCK 3] STATS ON THE TEST SET
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

JOBLIB_PATH = "brain_mri_densenet.joblib"
TEST_DIR    = "testing"
EXCEL_PATH  = "brain_mri_results.xlsx"
BATCH_SIZE  = 32


def run_stats():
    b = joblib.load(JOBLIB_PATH)
    model = tf.keras.models.model_from_json(b["architecture"])
    model.set_weights(b["weights"])
    class_names, img_size = b["class_names"], tuple(b["img_size"])
    labels = list(range(len(class_names)))

    # shuffle=False keeps predictions aligned with labels
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, labels="inferred", label_mode="int", class_names=class_names,
        image_size=img_size, batch_size=BATCH_SIZE, shuffle=False)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    proba = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(proba, axis=1)
    print(f"\nEvaluated on {len(y_true)} test images")

    # ---- 4-class ----
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_txt = classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True)
    auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)

    print("\n" + "=" * 55 + "\nMULTICLASS (tumor type)\n" + "=" * 55)
    print(f"Accuracy      : {acc*100:.2f}%")
    print(f"Macro AUC-ROC : {auc:.4f}\n")
    print(report_txt)
    print("Confusion matrix:\n", cm)

    # ---- Binary: tumor vs no tumor ----
    yt_b, yp_b = (y_true > 0).astype(int), (y_pred > 0).astype(int)
    cm_b = confusion_matrix(yt_b, yp_b, labels=[0, 1])
    tn, fp, fn, tp = cm_b.ravel()
    print("\n" + "=" * 55 + "\nBINARY (tumor detection)\n" + "=" * 55)
    print(f"Accuracy            : {accuracy_score(yt_b, yp_b)*100:.2f}%")
    print(f"Sensitivity (recall): {tp/(tp+fn)*100:.2f}%   <- tumors correctly flagged")
    print(f"Specificity         : {tn/(tn+fp)*100:.2f}%")
    print(f"Missed tumors (FN)  : {fn}")
    print("Confusion matrix:\n", cm_b)

    # ---- Confusion matrix plot ----
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("DenseNet121 Confusion Matrix")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.show()

    # ---- Excel export ----
    df = pd.DataFrame(report).T.loc[class_names + ["macro avg", "weighted avg"]].round(4)
    df["support"] = df["support"].astype(int)
    summary = pd.DataFrame({"Metric": ["Accuracy", "Macro AUC-ROC", "Binary sensitivity", "Binary specificity"],
                            "Value": [round(acc, 4), round(auc, 4), round(tp/(tp+fn), 4), round(tn/(tn+fp), 4)]})
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Per-class")
        summary.to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_excel(w, sheet_name="Confusion")
    print(f"\nSaved: confusion_matrix.png, {EXCEL_PATH}")


if __name__ == "__main__":
    run_stats()