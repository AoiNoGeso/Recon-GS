# Recon-GS

単一の動画から 3D Gaussian Splatting (.ply) を生成するパイプライン。

## パイプライン

```
動画 → フレーム抽出 (15fps) → 動的物体マスキング (Grounded-SAM2)
     → カメラ姿勢推定 (hloc + COLMAP) → 3DGS 学習 (gsplat) → gaussian.ply
```

## セットアップ

```bash
# uv のインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# Step 1: torch を先にインストール（GroundingDINO のビルドに必要）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 2: 残りの依存関係をインストール
uv sync --python 3.12
```

> **Note**: GroundingDINO の `setup.py` はビルド時に torch を参照します。
> `uv sync` より前に torch を入れておく必要があります。

## 使い方

```bash
# フルパイプライン実行
uv run recon-gs pipeline --video input.mp4 --output output/

# 途中から再開（完了済みステップをスキップ）
uv run recon-gs pipeline --video input.mp4 --output output/ --resume

# 特定ステップから再実行
uv run recon-gs pipeline --video input.mp4 --output output/ --from-step mask
# steps: extract | mask | sfm | train
```

## 出力

```
output/
├── frames/         # 抽出フレーム (15fps PNG)
├── masks/          # 動的物体マスク
├── colmap/         # SfM 中間ファイル
├── pipeline.json   # ステップ完了フラグ（再開用）
└── gaussian.ply    # 最終出力
```

最終成果物は `output/gaussian.ply` のみ。中間ファイルは再開のために保持されます。

## 設定

[`recon_gs/config.py`](recon_gs/config.py) で以下を変更できます：

| 定数 | デフォルト | 説明 |
|------|-----------|------|
| `MASK_PROMPTS` | person, pedestrian, ... | マスク対象の動的物体カテゴリ |
| `EXTRACT_FPS` | 15 | フレーム抽出レート |
| `TRAIN_ITERATIONS` | 30,000 | 3DGS 学習イテレーション数 |
