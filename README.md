# Plant Leaf Species Classifier

This project trains and serves a leaf-species image classifier using TensorFlow and Streamlit.

## What Is In Git vs Not In Git

Tracked in git:
- Source code (`app.py`, `train.py`, `evaluate.py`, `split_dataset.py`)
- Project config files (`.gitignore`, `requirements.txt`, `README.md`)

Ignored by `.gitignore` (not pushed):
- `venv/`
- `dataset/`
- Model artifacts like `*.keras`, `*.h5`

Important: If `dataset/` or `venv/` were committed before `.gitignore`, remove them from tracking once:

```powershell
git rm -r --cached dataset venv
git commit -m "Stop tracking local dataset and virtual environment"
```

## Python Version

Recommended: Python `3.10.x` (this project was built on `3.10.11`).

## Setup (Reproducible Environment)

```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Expected Dataset Layout

`dataset/` should look like this:

```text
dataset/
  images/
    field/
      <species_name>/
    lab/
      <species_name>/
  train/
    <species_name>/
  test/
    <species_name>/
```

If you only have `dataset/images/field` and `dataset/images/lab`, generate train/test split:

```powershell
python split_dataset.py
```

## Train

```powershell
python train.py
```

This saves: `leaf_species_model_finetuned.keras`

## Evaluate

```powershell
python evaluate.py
```

## Run Streamlit App

`app.py` currently loads `leaf_species_model.keras`. Ensure that file exists, or update `app.py` to load your desired model file.

```powershell
streamlit run app.py
```

## Reproducing Your Version on Another Machine

1. Clone the repo.
2. Create and activate a new virtual environment.
3. Install `requirements.txt`.
4. Place dataset and model files in the expected local paths.
5. Run training/evaluation/app commands above.
