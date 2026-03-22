import os
from pathlib import Path
from PIL import Image

# --- 設定 ---
input_base = Path(r"D:\tmp\work\road_classification\dataset")    # 整理済みのフォルダ
output_base = Path(r"D:\tmp\work\road_classification\dataset_small") # 軽量版の保存先
target_size = (300, 300) # 学習に十分なサイズ

# --- 処理 ---
for split in ['train', 'val', 'test']:
    for cls in ['snow', 'rain', 'night', 'fog']:
        src_dir = input_base / split / cls
        dst_dir = output_base / split / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        if not src_dir.exists():
            continue
            
        print(f"Resizing: {split}/{cls}...")
        for img_path in src_dir.glob('*'):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                try:
                    with Image.open(img_path) as img:
                        # アスペクト比を維持しつつリサイズ
                        img.thumbnail(target_size)
                        # JPG形式で保存（さらに軽量化）
                        save_path = dst_dir / (img_path.stem + ".jpg")
                        img.convert('RGB').save(save_path, "JPEG", quality=85)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

print(f"完了！ {output_base} をZIPにしてアップロードしてください。")