# Road Surface Condition Classifier (ResNet18)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ed.svg)](https://www.docker.com/)

車載カメラ画像から路面状況（積雪・雨・霧・夜）を判定するAIプロトタイプです。
単なる精度の追求にとどまらず、**「説明可能AI（XAI）」を用いたモデルの妥当性検証**と、**「実運用を見据えた環境最適化」**に注力しました。

---

## 🌟 プロジェクトのハイライト

本プロジェクトでは、AI開発における「ブラックボックス化」と「リソース肥大化」という実務上の課題に対し、以下のエンジニアリングアプローチで解決を図りました。

1.  **リソースの最適化**:
    - 元データ約7GBを、精度を維持したまま画像リサイズと圧縮により**40MB（約1/175）まで軽量化**。Google Driveやクラウド環境のストレージ圧迫を回避し、学習コストを削減。
2.  **汎化性能の向上とロバスト性確保**:
    - `GaussianBlur`や`ColorJitter`等の高度なデータ拡張を採用。フロントガラスの汚れや地吹雪といった実環境特有のノイズに対する耐性を強化。
3.  **XAI（Grad-CAM）による推論根拠の可視化**:
    - AIが「どこを見て」判定したかをヒートマップで可視化。誤判定の原因を論理的に特定し、改善のサイクルを回すプロセスを構築。
4.  **コンテナ化による環境再現性**:
    - Dockerを採用。OSやライブラリの依存関係を排除し、あらゆる環境で即座に推論を実行可能。

---

## 🔍 Grad-CAMによる失敗分析と改善

「フロントガラスに雪が張り付いた画像」を用いた、改善前後の比較分析結果です。

| 項目 | 初期モデル (Baseline) | 改善モデル (Augmented) |
| :--- | :--- | :--- |
| **判定結果** | **Fog (確信度 86.1%)** | **Snow (確信度 98.1%)** |
| **注視点 (赤い領域)** | **画像中央の白い発光部分** | **路面上のタイヤ痕（直線的特徴）** |
| **技術的考察** | 画面全体の「白さ」を霧と誤認する**ショートカット学習**が発生。路面を無視していた。 | ノイズを無視し、路面の本質的なテクスチャを捉えるように**汎化性能が向上**。 |

![Grad-CAM Result](results/GradCAM_result.png)

---

## 📂 ディレクトリ構成

```text
.
├── app/
│   ├── main.py           # 推論実行メインスクリプト
│   └── models/
│       └── road_model_augmented.pth # 学習済み重み
├── scripts/
│   ├── resize_images.py   # データ前処理・軽量化スクリプト
│   └── analyze_gradcam.py # XAI分析用スクリプト
├── Dockerfile             # 実行環境定義
├── requirements.txt       # 依存ライブラリ
└── README.md
```

## 🚀 実行方法 (Docker)

Docker環境があれば、以下のコマンドで即座に推論テストが可能です。

```bash
# イメージのビルド
docker build -t road-ai .

# 推論の実行（画像のパスを指定）
docker run --rm -v $(pwd)/test_images:/data road-ai /data/sample_image.jpg
```

---

## 🛠 技術スタック

- **Framework**: PyTorch / Torchvision (ResNet18)
- **Infrastructure**: Docker
- **Library**: OpenCV (Interpretation), Pillow (Preprocessing), Matplotlib
- **Environment**: Google Colab (Training) / Local Windows (Development)

---

## 📈 今後の展望 (Next Steps)
- **エッジデバイスへの最適化**: ONNXへの変換による推論速度の向上。
- **時系列情報の活用**: 動画ストリーム（LSTM等）を用いた、時間軸での判定安定化ロジックの実装。
- **フェイルセーフ設計**: 確信度が閾値以下の画像に対し「判定不能」を返し、安全な運行管理をサポートする仕組みの導入。
