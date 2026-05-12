# Smoker Detection — Dual Approach

Two complementary smoker / not-smoker image classifiers + a single web app
that runs both side-by-side.

| Approach   | What it is                                            | Output      |
|------------|-------------------------------------------------------|-------------|
| Classical  | HOG + LBP + HSV + Face-ROI -> PCA -> RBF-SVM         | `models/classical.pkl` |
| Deep       | ResNet-50 pretrained on ImageNet, two-phase fine-tune | `models/resnet50.pth`  |

Both trained locally on the **augmented dataset** from your Google Drive
(the `smoker_project/augmented_dataset` folder referenced in the Colab notebook,
class folders `Smoking/` and `Not_Smoking/`).

---

## 1. Activate the conda env

```powershell
conda activate smoker_det
```

The env already has CUDA-enabled PyTorch + every classical dep installed.

---

## 2. Get the dataset locally

Place the dataset under `./dataset/` so the layout is:

```
dataset/
    Smoking/        (~1432 images)
    Not_Smoking/    (~1432 images)
```

Option A — share the Drive folder as "Anyone with the link" and use `gdown`:

```powershell
python download_dataset.py --folder_id <GOOGLE_DRIVE_FOLDER_ID>
```

Option B — zip `augmented_dataset` in Drive and use the file ID:

```powershell
python download_dataset.py --zip_id <GOOGLE_DRIVE_FILE_ID>
```

Option C — just download the folder from Drive in your browser and drop it
under `./dataset/` manually.

---

## 3. Train

```powershell
# Deep approach — uses your RTX 3060
python train_resnet.py --data_dir ./dataset --model_out ./models/resnet50.pth

# Classical approach (slower with GridSearch; add --fast to skip)
python train_classical.py --data_dir ./dataset --model_out ./models/classical.pkl
```

Each script writes a `.txt` evaluation report next to its model and (for the
ResNet) plots to `./eval_plots/resnet_*.png`.

---

## 4. Predict from the CLI

```powershell
python predict.py --model resnet    --input some_image.jpg
python predict.py --model classical --input ./test_images/
```

---

## 5. Run the web app (both models live)

```powershell
python app.py
```

Then open http://localhost:5000. The UI lets you switch between
**BOTH**, **RESNET-50**, and **CLASSICAL** — "both" shows agreement /
disagreement between the two models.

---

## Files

```
final_proj_digital_image/
    download_dataset.py    -- gdown helper
    features.py            -- HOG / LBP / HSV / Face-ROI extractor (classical)
    train_classical.py     -- HOG+LBP+SVM training + report
    train_resnet.py        -- ResNet-50 two-phase fine-tune + plots
    predict.py             -- CLI inference (either model)
    app.py                 -- Flask server serving both
    static/index.html      -- single-page UI with model toggle
    models/                -- output: classical.pkl, resnet50.pth
    eval_plots/            -- output: resnet_training.png, resnet_confusion.png, ...
```
# Smoker_detection
