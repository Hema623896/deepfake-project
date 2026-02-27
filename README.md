# Deepfake Detection Web App (ViT)

Simple Flask app using a pretrained Vision Transformer (ViT) to classify Real vs Fake images/videos and show a saliency heatmap overlay.

Quick start

1. Create a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install PyTorch first using the official wheels for your Python and CUDA version.

- CPU-only example (Windows):

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

- CUDA example (adjust CUDA version to your GPU; this is an example for CUDA 11.8):

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
```

3. Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python app.py
```

5. Open http://127.0.0.1:5000/

Notes
- The classifier head is reinitialized to 2 classes on top of a pretrained ViT feature extractor. For production accuracy you'll want to fine-tune on a labeled deepfake dataset.
- Explainability is implemented via input-gradient saliency maps and overlaid as a heatmap.

Fine-tuning

- Prepare data directory with the following structure:

```
data/
	train/
		real/    # place real human face images here
		fake/    # place fake/deepfake face images here
	val/
		real/    # validation split
		fake/
```

- You can quickly generate placeholder face images by running the helper script:

```bash
python create_sample_data.py --out data --n 16
```

  The script now pulls *real-looking* faces for the `real` class (via loremflickr) and AI‑generated faces for the
  `fake` class (via thispersondoesnotexist.com). If the network request fails it will fall back to coloured blocks.

- Run the training script (example):

```bash
python train.py --data data --epochs 5 --batch 16 --out model_finetuned.pth
```

- After training, `app.py` will automatically load `model_finetuned.pth` if it exists; no further modifications are required.  
  (The model loader already handles both formats stored by `train.py`.)

