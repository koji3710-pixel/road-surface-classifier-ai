import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os
import json

# クラス定義
CLASS_NAMES = ['Fog', 'Night', 'Rain', 'Snow']

def load_model(model_path, device):
    """学習済み重みをロードしたResNet18モデルを返す"""
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(image_path, model, device):
    """1枚の画像に対して推論を行い、結果を返す"""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        conf, pred = torch.max(probs, 0)
    
    return {
        "prediction": CLASS_NAMES[pred.item()],
        "confidence": float(conf.item()),
        "all_probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    model_file = "models/road_model_augmented.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = load_model(model_file, device)
        result = predict(img_path, model, device)
        # 他のシステムと連携しやすくするためJSON形式で標準出力
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)