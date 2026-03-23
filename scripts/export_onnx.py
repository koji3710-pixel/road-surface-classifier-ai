import torch
import torchvision.models as models
import torch.nn as nn
import onnx

# 1. モデルの構造を定義 (ResNet18)
device = torch.device("cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4クラス分類

# 2. 学習済み重みのロード
model_path = "app/models/road_model_augmented.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 3. ダミーデータの作成
# ONNXは「どんな形のデータが入ってくるか」を教えるために、一度テストデータを流す必要があります。
# (バッチサイズ1, 3チャンネル, 224x224ピクセル)
dummy_input = torch.randn(1, 3, 224, 224)

# 4. ONNXとして書き出し
onnx_path = "app/models/road_model.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True, 
    opset_version=11,      # 標準的なバージョン
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output']
)

print(f"✅ ONNX model has been exported to {onnx_path}")