# Plant Multi-Organ Classifier (TensorFlow)

This project trains an image classifier using TensorFlow/Keras and an EfficientNetB3 backbone in 3 stages:

1. Initial frozen-backbone training
2. Continue frozen training to 12 total epochs
3. Partial unfreezing fine-tuning

## Current Project Files

Tracked training scripts:
- `train_multiorgan.py`
- `train_continue_12epoches.py`
- `fine_tune_unfrozen.py`

Config/docs:
- `.gitignore`
- `requirements.txt`
- `README.md`

## What Is Ignored by Git

From `.gitignore`, local-only artifacts include:
- `venv/`
- `train_images/`
- `test_images/`
- `train_labels.csv`
- `test_labels.csv`
- model files like `*.keras`, `*.h5`

If these were tracked earlier, untrack once:

```powershell
git rm -r --cached train_images test_images venv
git rm --cached train_labels.csv test_labels.csv
git commit -m "Stop tracking local data and environment files"
```

## Python and Setup

Recommended Python: `3.10.x`

```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Expected Local Data Layout

```text
train_labels.csv
test_labels.csv
train_images/
  train_images/
    <image files referenced in train_labels.csv>
test_images/
  test_images/
    <image files referenced in test_labels.csv>
```

The CSV files are expected to have at least:
- `filename`
- `label`

## Training Pipeline

1) Initial training (frozen EfficientNetB3 backbone, 8 epochs):

```powershell
python train_multiorgan.py
```

Output:
- `multiorgan_model.keras`

2) Continue frozen training (adds 4 epochs, 12 total):

```powershell
python train_continue_12epoches.py
```

Output:
- `multiorgan_model_12epochs.keras`

3) Fine-tune (unfreeze top ~30% layers, 5 epochs, lower LR):

```powershell
python fine_tune_unfrozen.py
```

Output:
- `multiorgan_final_finetuned.keras`

## Notes

- Input image size is `300x300`.
- Preprocessing uses `tf.keras.applications.efficientnet.preprocess_input`.
- Current scripts use `test_labels.csv` + `test_images/` as validation data during training.
