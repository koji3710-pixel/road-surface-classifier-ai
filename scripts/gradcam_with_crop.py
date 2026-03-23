import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# --- 設定 ---
device = torch.device("cpu")
model_path = "app/models/road_model_augmented.pth"
image_path = "test_images/test_photo_01.jpg" # 存在する画像パスに変更してください
output_path = "results/gradcam_with_crop.png"
# main.pyと同じ順番のリストにする
CLASS_NAMES = ['Fog', 'Night', 'Rain', 'Snow'] 

# --- 1. モデルの準備 (前と同じ) ---
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- 2. Grad-CAM用のフック (前と同じ) ---
# 最後の畳み込み層を指定 (ResNet18の場合は layer4)
target_layer = model.layer4

activations = []
gradients = []

def activation_hook(module, input, output):
    activations.append(output)

def gradient_hook(module, input_grad, output_grad):
    gradients.append(output_grad[0])

target_layer.register_forward_hook(activation_hook)
target_layer.register_full_backward_hook(gradient_hook)

# --- 3. [重要] ONNX版と完全に一致させた前処理 (手動実装) ---
img_pil = Image.open(image_path).convert('RGB')

# main.pyのResize(256) + CenterCrop(224) を手動再現
w, h = img_pil.size
if w < h:
    new_w, new_h = 256, int(h * 256 / w)
else:
    new_w, new_h = int(w * 256 / h), 256
img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)

left = (new_w - 224) / 2
top = (new_h - 224) / 2
img_cropped = img_resized.crop((left, top, left + 224, top + 224))

# どのような画像がAIに入力されたか、確認用に保存
img_cropped.save("results/gradcam_cropped_input.jpg")

# 残りのToTensorとNormalize (NumPyベースからPyTorchベースに戻す)
final_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = final_preprocess(img_cropped).unsqueeze(0).to(device)

# --- 4. 推論とGrad-CAMの計算 ---
# 推論結果 (Rain = 2番目のクラス の勾配を求める)
output = model(input_tensor)
target_class_index = CLASS_NAMES.index('Rain') # 今回は Rain ('2') を指定
print(f"Pred Scores: {output[0].detach().numpy()}")
print(f"Targeting class: {CLASS_NAMES[target_class_index]} (Index: {target_class_index})")

# 勾配をクリアして後ろ向き計算
model.zero_grad()
# 判定が外れたクラス(Rain)のスコアに対してbackwardを行う
output[0, target_class_index].backward()

# --- 5. ヒートマップの生成 (前と同じ) ---
# 勾配のグローバル平均プーリング
pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

# 特徴マップに勾配を掛けて、ReLUをかける
act = activations[0][0]
for i in range(act.shape[0]):
    act[i, :, :] *= pooled_gradients[i]

heatmap = torch.mean(act, dim=0).detach().numpy()
heatmap = np.maximum(heatmap, 0) # ReLU
heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1 # 正規化

# --- 6. 元画像（Crop後の画像）に重ね書き ---
# 保存したCrop画像を読み込む
img_cv = cv2.imread("results/gradcam_cropped_input.jpg")
heatmap_cv = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
heatmap_cv = np.uint8(255 * heatmap_cv)
heatmap_cv = cv2.applyColorMap(heatmap_cv, cv2.COLORMAP_JET)

# 重ね書き (0.4がヒートマップの透明度)
superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_cv, 0.4, 0)

# 結果の保存
os.makedirs("results", exist_ok=True)
cv2.imwrite(output_path, superimposed_img)
print(f"✅ Grad-CAM result saved to {output_path}")
print(f"Please check results/gradcam_cropped_input.jpg (What AI actually saw)")