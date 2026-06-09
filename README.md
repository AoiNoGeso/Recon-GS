# Recon-GS

スマートフォンで撮影した動画から 3D Gaussian Splatting (.ply) とメッシュ (.ply) を生成するパイプライン。

## パイプライン

```
動画
 │
 ├─ Step 1: フレーム抽出 (ffmpeg, 15fps)
 ├─ Step 2: 動的物体マスキング (GroundingDINO + SAM2)
 ├─ Step 3: カメラ姿勢推定 (hloc + COLMAP)
 ├─ Step 4: 3DGS 学習 (gsplat) + 重力アライメント
 └─ Step 5: メッシュ生成 (TSDF Fusion, Open3D)
```

## セットアップ

```bash
# 1. uv のインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. COLMAP のインストール
sudo apt install colmap   # Ubuntu

# 3. torch を先にインストール（SAM2 のビルドに必要、CUDA 12.1 向け）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. 残りの依存関係をインストール
uv sync --python 3.12
```

> **Note**: SAM2 の `setup.py` はビルド時に torch を参照するため、
> `uv sync` より前に `uv pip install torch` が必要です。

## 使い方

```bash
# フルパイプライン実行
uv run recon-gs pipeline --video input.mp4 --output output/

# 途中から再開（完了済みステップをスキップ）
uv run recon-gs pipeline --video input.mp4 --output output/ --resume

# 特定ステップから再実行
uv run recon-gs pipeline --video input.mp4 --output output/ --from-step mesh
# steps: extract | mask | sfm | train | mesh
```

## 出力

```
output/
├── frames/              # 抽出フレーム (15fps PNG)
├── masks/               # 動的物体マスク
├── colmap/              # SfM 中間ファイル
│   └── sparse/0/
│       └── gravity_rotation.json  # 重力アライメント回転行列（キャッシュ）
├── mesh/
│   ├── tsdf_fusion.ply       # 生メッシュ
│   └── tsdf_fusion_post.ply  # クリーニング済みメッシュ
├── pipeline.json        # ステップ完了フラグ（再開用）
└── gaussian.ply         # 3DGS 出力
```

## 設定

[`recon_gs/config.py`](recon_gs/config.py) で変更できます。

### 動的物体マスク

| 定数 | デフォルト | 説明 |
|------|-----------|------|
| `MASK_PROMPTS` | person, pedestrian, ... | マスク対象の動的物体カテゴリ |
| `GROUNDING_DINO_BOX_THRESHOLD` | 0.3 | 検出ボックス信頼度閾値 |
| `GROUNDING_DINO_TEXT_THRESHOLD` | 0.25 | テキスト一致信頼度閾値 |

### フレーム抽出

| 定数 | デフォルト | 説明 |
|------|-----------|------|
| `EXTRACT_FPS` | 15 | フレーム抽出レート |

### 3DGS 学習

| 定数 | デフォルト | 説明 |
|------|-----------|------|
| `TRAIN_ITERATIONS` | 30,000 | 学習イテレーション数 |

### メッシュ生成 (TSDF Fusion)

| 定数 | デフォルト | 説明 |
|------|-----------|------|
| `MESH_VOXEL_SIZE` | 0.05 | ボクセルサイズ [m]（小さくするほど精細） |
| `MESH_MAX_DEPTH` | 8.0 | 深度カットオフ [m]（シーン対角線に合わせて調整） |
| `MESH_MIN_CLUSTER_TRIANGLES` | 500 | 孤立クラスタ除去の最小三角形数 |

### サーフェスマスク（Grounded-SAM2 プロンプトベース）

Grounded-SAM2 (GroundingDINO + SAM2) でサーフェス領域を検出し、TSDF 統合から除外します。  
動的物体マスキング (Step 2) と同じモデルを使用します。

| 定数 | デフォルト | 説明 |
|------|-----------|------|
| `MESH_MASK_FLOOR` | True | 床をマスク（屋内） |
| `MESH_MASK_CEILING` | True | 天井をマスク（屋内） |
| `MESH_MASK_GROUND` | False | 地面・道路をマスク（屋外） |
| `MESH_MASK_SKY` | False | 空をマスク（屋外） |
| `MESH_FLOOR_PROMPTS` | floor, carpet, ... | 床検出プロンプト（カスタマイズ可） |
| `MESH_CEILING_PROMPTS` | ceiling | 天井検出プロンプト |
| `MESH_GROUND_PROMPTS` | ground, road, ... | 地面検出プロンプト |
| `MESH_SKY_PROMPTS` | sky | 空検出プロンプト |
| `MESH_GDINO_BOX_THRESHOLD` | 0.25 | GroundingDINO ボックス信頼度閾値 |
| `MESH_GDINO_TEXT_THRESHOLD` | 0.20 | GroundingDINO テキスト一致閾値 |

> **Note**: 別のシーンで再実行する場合は `colmap/sparse/0/gravity_rotation.json` を削除してください（重力アライメントキャッシュのリセット）。

## 技術的な詳細

### 重力アライメント (`recon_gs/align.py`)

COLMAP の再構成は重力方向が保証されないため、全カメラの up ベクトル平均から重力軸を推定し、
ロール方向の傾きを自動補正します。学習・メッシュ生成の両方に同じ回転行列を適用します。

### メッシュ生成 (`recon_gs/export_mesh.py`)

1. 学習済みガウシアンから各訓練カメラの RGB・深度をレンダリング
2. Grounded-SAM2 (GroundingDINO + SAM2) でサーフェス領域（床・天井など）をプロンプトベースで検出しマスク
3. Open3D の ScalableTSDFVolume で深度フレームを統合
4. 孤立クラスタ・縮退三角形を除去して出力
