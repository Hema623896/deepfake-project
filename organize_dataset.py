import os
import shutil
from pathlib import Path
import argparse


def organize_dataset(source_dir, output_dir='data', real_ratio=0.8):
    """
    Organize images from source folder into real/fake training and validation sets.
    
    Expected source structure:
        source_dir/
            real/        (contains real human face images)
            fake/        (contains fake/AI-generated human face images)
    
    Output structure created:
        data/
            train/
                real/    (80% of real images)
                fake/    (80% of fake images)
            val/
                real/    (20% of real images)
                fake/    (20% of fake images)
    """
    source = Path(source_dir)
    output = Path(output_dir)
    
    if not source.exists():
        print(f"Error: Source directory '{source_dir}' does not exist")
        return
    
    real_src = source / 'real'
    fake_src = source / 'fake'
    
    if not real_src.exists() or not fake_src.exists():
        print(f"Error: Expected subdirectories 'real' and 'fake' in '{source_dir}'")
        print(f"  Found 'real': {real_src.exists()}")
        print(f"  Found 'fake': {fake_src.exists()}")
        return
    
    # create output directories
    for split in ['train', 'val']:
        for cls in ['real', 'fake']:
            (output / split / cls).mkdir(parents=True, exist_ok=True)
    
    # process real images
    real_images = [f for f in real_src.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    split_idx = int(len(real_images) * real_ratio)
    
    for i, img_path in enumerate(real_images):
        split = 'train' if i < split_idx else 'val'
        dest = output / split / 'real' / img_path.name
        shutil.copy2(img_path, dest)
        print(f"Copied {img_path.name} -> data/{split}/real/")
    
    # process fake images
    fake_images = [f for f in fake_src.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    split_idx = int(len(fake_images) * real_ratio)
    
    for i, img_path in enumerate(fake_images):
        split = 'train' if i < split_idx else 'val'
        dest = output / split / 'fake' / img_path.name
        shutil.copy2(img_path, dest)
        print(f"Copied {img_path.name} -> data/{split}/fake/")
    
    print(f"\n✓ Dataset organized into {output}")
    print(f"  Real images: {len(real_images)} total ({split_idx} train, {len(real_images)-split_idx} val)")
    print(f"  Fake images: {len(fake_images)} total ({int(len(fake_images)*real_ratio)} train, {len(fake_images)-int(len(fake_images)*real_ratio)} val)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize real and fake images into train/val directories')
    parser.add_argument('source', help='Source directory with real/ and fake/ subdirectories')
    parser.add_argument('--out', default='data', help='Output data directory (default: data)')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of images for training (default: 0.8)')
    args = parser.parse_args()
    
    organize_dataset(args.source, output_dir=args.out, real_ratio=args.train_ratio)
