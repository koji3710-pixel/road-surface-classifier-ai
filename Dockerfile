# 1. 軽量なPython 3.12をベースに使用
FROM python:3.12-slim

# 2. 作業ディレクトリの作成
WORKDIR /app

# 3. 必要なライブラリをインストール
# --no-cache-dir でイメージサイズを削減
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションコードと学習済みモデルをコピー
COPY main.py .
COPY models/ ./models/

# 5. 画像を保持するためのディレクトリ（実行時にマウントすることを想定）
RUN mkdir /data

# 6. コンテナ起動時に実行するコマンド
# デフォルトで main.py を実行するように設定
ENTRYPOINT ["python", "main.py"]