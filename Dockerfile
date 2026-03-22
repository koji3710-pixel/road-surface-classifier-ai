# 1. 軽量なPython 3.12をベースに使用
FROM python:3.12-slim

# 2. 作業ディレクトリを /app に設定
WORKDIR /app

# 3. 依存関係ファイルをコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. app/ フォルダ内のソースコードとモデルを、コンテナの /app/ 直下にコピー
COPY app/main.py .
COPY app/models/ ./models/

# 5. データマウント用のディレクトリ作成
RUN mkdir /data

# 6. 実行コマンド
ENTRYPOINT ["python", "main.py"]