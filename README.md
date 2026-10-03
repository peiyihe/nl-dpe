# NL-DPE SST-2, Noise-Aware Training, and SuperT Replay

[中文版](README_CN.md)

This repository is based on [Hewlett Packard Labs NL-DPE](https://github.com/HewlettPackard/nl-dpe). This fork adds an end-to-end SST-2 workflow for exporting one TinyBERT attention head, mapping its weights to differential conductances, performing noise-aware fine-tuning, evaluating weight-noise sensitivity, and replaying matrix-multiplication results returned by Super-T inside full-model inference.

This README describes only the additions made in this fork. GLUE datasets are not included in the repository.

## Experiment configuration

| Item | Configuration |
|---|---|
| Model | `models/wise-tern-584` |
| Dataset | SST-2 dev, 872 examples |
| Attention architecture | 4 layers, 12 heads, hidden size 312 |
| Selected block | Layer 1, Head 12; both indices are 1-based |
| Head dimension | 26 |
| Sequence length | 64 |
| Q/K/V weights | Three distinct `26 x 26` matrices |
| Q/K/V input | The same input matrix `X` |
| Conductance range | 0–150 μS |
| Differential-pair layout | Left column: `Gpositive`; right column: `Gnegative` |

## Additions

### 1. WQ, WK, and WV export

[`nl_dpe/export_attention_head.py`](nl_dpe/export_attention_head.py) exports the three `26 x 26` matrices for one layer and attention head.

The clean weights used for the hardware experiment are stored in:

`artifacts/sst2-clean-layer1-head12-weights-x-matmul/`

These matrices have been transposed from the native PyTorch `nn.Linear` `[out, in]` layout so that they can be used directly as:

```text
Q = X @ WQ
K = X @ WK
V = X @ WV
```

Do not transpose these exported clean matrices again. Bias values are not included.

```bash
python -m nl_dpe.export_attention_head \
  --model-path models/wise-tern-584 \
  --output-dir artifacts/reproduced-sst2-clean-weights \
  --layer 1 \
  --head 12 \
  --orientation x-matmul
```

### 2. Differential-conductance mapping

[`nl_dpe/map_differential_conductance.py`](nl_dpe/map_differential_conductance.py) maps each weight matrix independently to the 0–150 μS conductance range:

```text
scale = 150 / max(abs(W))
Gpositive = max(W, 0) * scale
Gnegative = max(-W, 0) * scale
W = (Gpositive - Gnegative) / scale
```

Each scaling factor is computed over one complete `26 x 26` matrix. It is not computed per column.

| Matrix | Scale (μS / weight) |
|---|---:|
| WQ | 298.722296239 |
| WK | 302.115447180 |
| WV | 247.284348627 |

Each logical output column is stored as two adjacent physical columns:

```text
physical[:, 2*j]     = Gpositive[:, j]   # left
physical[:, 2*j + 1] = Gnegative[:, j]   # right
```

The interleaved files named `W*_differential_interleaved.npy` therefore have shape `26 x 52`.

```bash
python -m nl_dpe.map_differential_conductance \
  --weights-dir artifacts/sst2-clean-layer1-head12-weights-x-matmul \
  --output-dir artifacts/reproduced-sst2-conductance \
  --g-min 0 \
  --g-max 150
```

The existing mapped arrays and their exact scaling factors are stored in:

`artifacts/sst2-clean-layer1-head12-conductance-differential/`

### 3. Q/K/V input export

[`nl_dpe/export_attention_inputs.py`](nl_dpe/export_attention_inputs.py) captures the common 26-dimensional input to the Layer 1 Q, K, and V projections.

The average-hidden-state attention architecture first reshapes each 312-dimensional hidden state and then averages across the 12 heads:

```text
hidden [batch, sequence, 312]
  -> [batch, sequence, 12, 26]
  -> mean over the head dimension
X [batch, sequence, 26]
```

The generated files are:

| File | Shape | Contents |
|---|---:|---|
| `X_all_positions.npy` | `55808 x 26` | 872 examples × 64 positions, including padding |


Full-model replay must use `X_all_positions.npy`. Its row order is:

```text
row = example_index * 64 + token_position
```

```bash
python -m nl_dpe.export_attention_inputs \
  --model-path models/wise-tern-584 \
  --data-dir datasets/glue/SST-2 \
  --output-dir artifacts/reproduced-sst2-input-x \
  --task-name sst-2 \
  --layer 1
```

### 4. SuperT vector matrix multiplication experiments for full SST-2 inference

Replay programs were added:

| Program | External projections |
|---|---|
| [`evaluate_external_qkv.py`](nl_dpe/evaluate_external_qkv.py) | Q, K, and V |

The `FromSuperT/out_mult_*_linear_corr.npy` files contain linear-corrected differential-pair vector matrix multiplication outputs returned by SuperT. Each has shape `55808 x 52`. The current files already include restoration of the global input-X scale required by Super-T.

The replay programs convert adjacent positive and negative columns back to the weight domain:

```text
Y = (paired[:, 0::2] - paired[:, 1::2]) / (scale_uS_per_weight * 1e-6)
```

They then add the original clean bias and replace only the selected outputs from Layer 1 Head 12. All other attention heads, layers, and the classifier remain software computations.

Example for full Q/K/V replay:

```bash
python -m nl_dpe.evaluate_external_qkv \
  --model-path models/wise-tern-584 \
  --data-dir datasets/glue/SST-2 \
  --x-file artifacts/sst2-clean-dev-layer1-qkv-input-x/X_all_positions.npy \
  --weights-dir artifacts/sst2-clean-layer1-head12-weights-x-matmul \
  --external-q FromSuperT/out_mult_wq_linear_corr.npy \
  --external-k FromSuperT/out_mult_wk_linear_corr.npy \
  --external-v FromSuperT/out_mult_wv_linear_corr.npy \
  --conductance-metadata artifacts/sst2-clean-layer1-head12-conductance-differential/metadata.json \
  --output-dir artifacts/reproduced-sst2-supert-qkv
```

Saved replay results:

| Inference path | Accuracy | Evaluation loss | Predictions changed from baseline |
|---|---:|---:|---:|
| Clean software baseline | 91.7431% | 0.237344 | 0 |
| SuperT Q + K + V | 91.6284% | 0.236777 | 3 |

Pearson correlation between the SuperT outputs and software matrix-multiplication references:

| Projection | Correlation |
|---|---:|
| Q | 0.96637 |
| K | 0.96269 |
| V | 0.90837 |

### 5. Noise-aware full-model fine-tuning (in-building, previous experiments use clean weight)

[`nl_dpe/train_noise_aware.py`](nl_dpe/train_noise_aware.py) uses the conductance-dependent write and read noise from `TCAD.models.utils.noise.DPENoise`.

The saved experiment uses the following configuration:

- Noise is injected into the Layer 1 Head 12 WQ, WK, and WV `26 x 26` slices during forward passes.
- The optimizer updates the entire TinyBERT checkpoint, which is why the result is named **NAF-Full**.
- `--no-analog-slicing` disables the residual correction cell. Each simulated weight uses one continuous-conductance cell.
- `std_scale=1.0` is a multiplier applied to the TCAD noise standard deviation. It does not mean 1% Gaussian noise.
- The conductance range is 0–150 μS.
- Training uses one epoch, batch size 128, and learning rate `5e-5`.
- The loss includes the regularizer `0.01 * max(abs(selected Q/K/V weights))`.

```bash
python -m nl_dpe.train_noise_aware \
  --task-name sst-2 \
  --model-path models/wise-tern-584 \
  --data-dir datasets/glue/SST-2 \
  --output-dir artifacts/reproduced-sst2-naf-full \
  --layer 1 \
  --head 12 \
  --epochs 1 \
  --batch-size 128 \
  --learning-rate 5e-5 \
  --weight-norm 0.01 \
  --g-min 0 \
  --g-max 150 \
  --std-scale 1 \
  --no-analog-slicing
```

The saved full checkpoint is located at:

`artifacts/sst2-naf-full-no-analog-slicing-layer1-head12/model/`

The training script also exports `weights/WQ.npy`, `WK.npy`, and `WV.npy`. These three NAF files retain the native PyTorch `F.linear` orientation, so their equation is `X @ W.T`. Their orientation differs from the clean `x-matmul` hardware files described above.

Results from the saved training run:

| Checkpoint | Clean accuracy | TCAD noisy accuracy |
|---|---:|---:|
| Clean checkpoint before training | 91.7431% | 91.7431% |
| NAF-Full | 90.2523% | 90.2523% |

This one-epoch NAF-Full run did not outperform the original clean checkpoint.

## Added files and artifacts

```text
FromSuperT/
  20261003_wq_ssm2.ipynb
  20261003_wk_ssm2.ipynb
  out_mult_wq_*.npy
  out_mult_wk_*.npy
  out_mult_wv_*.npy
  g_after_program_*.npy
  g_after_mult_*.npy

artifacts/
  sst2-clean-dev-layer1-qkv-input-x/
  sst2-clean-layer1-head12-weights-x-matmul/
  sst2-clean-layer1-head12-conductance-differential/
  sst2-naf-full-no-analog-slicing-layer1-head12/
  sst2-gaussian-noise-sweep-no-analog-slicing-layer1-head12/
  sst2-supert-wq-linear-corr-inference/
  sst2-supert-qk-linear-corr-inference/
  sst2-supert-qkv-linear-corr-inference/

nl_dpe/
  export_attention_head.py
  export_attention_inputs.py
  map_differential_conductance.py
  train_noise_aware.py
  evaluate_weight_noise_sweep.py
  plot_noise_sweep.py
  evaluate_external_wq.py
  evaluate_external_qk.py
  evaluate_external_qkv.py
```

Each artifact directory contains a `metadata.json` or `results.json` file that records source files, shapes, scaling factors, experiment parameters, and evaluation results.

## Environment and dataset

Create a separate Conda environment:

```bash
conda create -n nldpe-sst2 python=3.10 -y
conda activate nldpe-sst2
python -m pip install -e .
```

SST-2 is not included in this repository. Before running export, training, or inference, place the GLUE SST-2 files under:

```text
datasets/glue/SST-2/
```

Downloaded datasets under `datasets/` are excluded from Git.
