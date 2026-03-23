import onnxruntime as ort
import numpy as np
from PIL import Image

# 1. ONNXモデルの読み込み
session = ort.InferenceSession("app/models/road_model.onnx")

# 2. main.pyと完全に一致させた前処理
def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    
    # --- main.pyのResize(256) + CenterCrop(224) を再現 ---
    # まず短辺を256に合わせてリサイズ
    w, h = img.size
    if w < h:
        new_w, new_h = 256, int(h * 256 / w)
    else:
        new_w, new_h = int(w * 256 / h), 256
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    # 中央を224x224で切り抜く
    left = (new_w - 224) / 2
    top = (new_h - 224) / 2
    img = img.crop((left, top, left + 224, top + 224))
    # --------------------------------------------------

    img_data = np.array(img).astype('float32')
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_data = (img_data / 255.0 - mean) / std
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    return img_data

# 3. 推論の実行
image_path = "test_images/test_photo_01.jpg" 
input_data = preprocess(image_path)
raw_result = session.run(None, {"input": input_data})

# 4. ラベル定義（main.pyと一致）
classes = ['Fog', 'Night', 'Rain', 'Snow']
prediction = classes[np.argmax(raw_result[0])]

print(f"--- ONNX Inference Result (Strict Mode) ---")
print(f"Prediction: {prediction}")