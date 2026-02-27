from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse
import io
import urllib.request


def download_face_image(path, cls, seed=None):
    # retrieve a face image; for 'real' use a placeholder service that returns photos of people,
    # for 'fake' use an AI-generated face generator.
    if cls == 'real':
        url = f"https://loremflickr.com/224/224/face?random={seed or ''}"
    else:
        # thispersondoesnotexist returns a new AI‑generated face each request
        url = "https://thispersondoesnotexist.com/image"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data))
        img.convert('RGB').save(path)
        return True
    except Exception:
        return False


def make_color_image(path, color, text):
    # fallback if download fails
    img = Image.new('RGB', (224, 224), color=color)
    draw = ImageDraw.Draw(img)
    try:
        f = ImageFont.load_default()
    except Exception:
        f = None
    draw.text((10, 100), text, fill=(255,255,255), font=f)
    img.save(path)


def create_sample(out_dir, n=8):
    out = Path(out_dir)
    for split in ['train', 'val']:
        for cls in ['real', 'fake']:
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                path = d / f'{cls}_{i+1}.jpg'
                # try downloading an appropriate face image
                if not download_face_image(path, cls, seed=f"{cls}-{split}-{i}"):
                    # fallback to colored placeholder if network fails
                    if cls == 'real':
                        color = (40, 120, 200)
                        text = f'Real {i+1}'
                    else:
                        color = (200, 60, 90)
                        text = f'Fake {i+1}'
                    make_color_image(path, color, text)
    print('Sample data created at', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data', help='output data directory')
    parser.add_argument('--n', type=int, default=8, help='images per class per split')
    args = parser.parse_args()
    create_sample(args.out, n=args.n)
