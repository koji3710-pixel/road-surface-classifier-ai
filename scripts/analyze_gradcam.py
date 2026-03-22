import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import sys
import os

# --- 設定 ---
CLASS_NAMES = ['Fog', 'Night', 'Rain', 'Snow']

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0][class_idx]
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.nn.functional.relu(cam).detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        # 0-1に正規化
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def main(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. モデルの準備
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    # 2. 画像の読み込みと前処理
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # 3. Grad-CAMの実行
    grad_cam = GradCAM(model, model.layer4[-1])
    output = model(input_tensor)
    _, preds = torch.max(output, 1)
    class_idx = preds[0].item()

    heatmap = grad_cam.generate_heatmap(input_tensor, class_idx)

    # 4. 可視化結果の合成
    img_display = np.array(img.resize((224, 224)))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (heatmap_colored * 0.4 + img_display * 0.6).astype(np.uint8)

    # 5. 結果の保存
    output_name = f"gradcam_result_{CLASS_NAMES[class_idx]}.jpg"
    cv2.imwrite(output_name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print(f"Analysis complete.")
    print(f"Predicted: {CLASS_NAMES[class_idx]}")
    print(f"Result saved as: {output_name}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_gradcam.py <image_path> <model_path>")
    else:
        main(sys.argv[1], sys.argv[2])