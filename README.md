# Non-Linear Dot Product Engine (NL-DPE)

This project is based on this [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT) repository.
Some of the changes we introduced:

- New model architectures based on original TinyBERT implementation.
- Support for newer Python versions.
- Code formatting.

The original top-level workflow provides inference rather than general model training. This checkout also includes a
dedicated noise-aware fine-tuning entry point,
[`nl_dpe/train_noise_aware.py`](./nl_dpe/train_noise_aware.py). It reuses the DPE write/read noise model from
`TCAD`, injects noise into one 26-dimensional attention head, and fine-tunes the full reduced TinyBERT model. For
example, to train with noise on layer 1, head 12 of the CoLA model and export its three 26x26 matrices:

```shell
python -m nl_dpe.train_noise_aware \
  --task-name cola \
  --model-path models/blushing-dove-984 \
  --data-dir datasets/glue/CoLA \
  --layer 1 \
  --head 12 \
  --epochs 1 \
  --output-dir artifacts/cola-noise-aware-layer1-head12
```

The output `weights` directory contains `WQ.npy`, `WK.npy`, and `WV.npy`, along with CSV copies and an
`attention_matrices.pkl` file compatible with the model update command described below.

## Pre-requisites

This project uses [uv](https://docs.astral.sh/uv/). Install it (Linux or MacOS):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# installing to $HOME/.local/bin
# To add $HOME/.local/bin to your PATH, either restart your shell or run `source $HOME/.local/bin/env`

uv --version
```
In addition, some of the commands described here (like downloading datasets) are implemented as taskipy tasks. To run
them, install taskipy as a standalone tool:
```bash
uv tool install taskipy   # Install standalone taskipy.
task --list               # List available tasks (they are described in this document).
```

## Install

```shell
# Clone the NL-DPE repository.
git clone https://github.com/HewlettPackard/nl-dpe
cd ./nl-dpe

# Create Python virtual environment.
uv venv --python 3.12     # This creates virtual environment in `./.venv`.
uv sync --no-group dev    # Install project dependencies (without development dependencies).

# Make sure project modules are importable.
export PYTHONPATH=$(pwd)
```

## Environment variables

`MLFLOW_TRACKING_URI` \[optional\]: MLflow tracking server URI to load models from (when model URI is `mlflow:{MLFLOW_RUN_ID}`).
This only works now when running evaluation on a system where the models under test were trained.

## Datasets

Evaluation script uses the GLUE dataset. See the README file from the TinyBERT repository ([Data Augmentation section](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT#data-augmentation)) for more details. From that README file:

> Before running data augmentation of GLUE tasks you should download the [GLUE](https://gluebenchmark.com/tasks) data by running this [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory.

We provide the [download_glue_data.py](./nl_dpe/download_glue_data.py) script which is an updated version of the
GitHub gist mentioned above.

- Download MRPC dataset by running `task download-mrpc` or running the following commands:

  ```shell
  mkdir -p ./datasets/mrpc && cd ./datasets/mrpc
  wget https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt
  wget https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt
  ls -lh
  sha1sum ./msr_paraphrase_test.txt ./msr_paraphrase_train.txt
  # size    sha1                                       file
  #  431K   4265196c15cf75620b0b592b8b921f543bda7e6c   msr_paraphrase_test.txt
  # 1023K   716e0f67af962f08220b7e97d229b293077ef41f   msr_paraphrase_train.txt
  ```

- Download GLUE dataset (after downloading the MRPC data) by running `task download-glue` or running the following commands:

  ```shell
  uv run --no-sync python ./nl_dpe/download_glue_data.py --data_dir=./datasets/glue --tasks=all --path_to_mrpc ./datasets/mrpc
  ```

## Models

Three models are located in the [models](./models/) directory.

- [wise-tern-584](./models/wise-tern-584/): **SST-2**.
  - From paper
    - BERT_base_teacher = 93.4, TinyBERT_4 = 92.6, TinyBERT_6 = 93.1
  - Our experiments
    - TinyBERT = 92.77
    - TinyBERT with hidden state averaging (run_name = wise-tern-584, run_id = 70457740292c4ff58fa8cb7ca009f154) = 91.74
- [blushing-dove-984](./models/blushing-dove-984/): **CoLA**
  - From paper
    - BERT_base_teacher = 52.8, TinyBERT_4 = 44.1, TinyBERT_6 = 51.1
  - Our experiments
    - TinyBERT = 40.92
    - TinyBERT with hidden state averaging (run_name = blushing-dove-984, run_id = 37059330c45b4a259ad5fadd812beaa5) = 31.18
- [spiffy-snake-501](./models/spiffy-snake-501/): **MRPC**
  - From paper
    - BERT_base_teacher = 87.5, TinyBERT_4 = 86.4, TinyBERT_6 = 87.3
  - Our experiments
    - TinyBERT = 87.36
    - TinyBERT with hidden state averaging (run_name = spiffy-snake-501, run_id = b3c699617932420da22862c37698724a) = 78.72

## Model evaluation

Look at [nl_dpe/evaluate.py](./nl_dpe/evaluate.py) file:

```shell
uv run python ./nl_dpe/evaluate.py <task_name> <model_uri> <data_dir>
#      <task_name>: One of 'cola', 'sst-2', 'mrpc', 'sts-b', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte', 'wnli'
#                   Task name must be consistent with the model being evaluated.
#      <model_uri>: Path to a model directory
#      <data_dir>:  Path to the data directory (e.g., ${GLUE_DIR}\\SST-2).
```

For these evaluation runs we also provide taskipy tasks (`eval-sst2`, `eval-cola` and `eval-mrpc`):
```shell
uv run python ./nl_dpe/evaluate.py sst-2 ./models/wise-tern-584/ ./datasets/glue/SST-2
# acc = 0.9174311926605505, eval_loss = 0.2411253090415682

uv run python ./nl_dpe/evaluate.py cola ./models/blushing-dove-984/ ./datasets/glue/CoLA
# mcc = 0.3118300074953714, eval_loss = 0.5854980647563934

uv run python ./nl_dpe/evaluate.py mrpc ./models/spiffy-snake-501/ ./datasets/glue/MRPC
# acc_and_f1 = 0.787219887955182, eval_loss = 0.583455924804394
```

## Extracting and updating self-attention weights

The [model.py](./nl_dpe/model.py) provides a number of CLI commands to extract self-attention weights, and then update
the original model with new weights.

Currently, all models in this repository have the same architecture - 4 encoder layers, 12 attention heads, hidden size
is 312, attention dimension 26 (`26 * 12 = 312`). One self-attention head then can be indexed with a tuple
`[layer, head]` where layer is a 1-based layer index (1, ..., 4 inclusive) and head is the 1-based index of the 
attention head (1, ..., 12 inclusive).

```shell
# Evaluate this model to get the expected performance.
uv run python ./nl_dpe/evaluate.py cola ./models/blushing-dove-984/ ./datasets/glue/CoLA
# eval_loss = 0.5854980764967023 mcc = 0.3118300074953714

# Save attention weights for 1st layer and last attention head
uv run python ./nl_dpe/model.py extract-attention-weights --model-path=./models/blushing-dove-984/ --output-file=att-matrices.pkl --layer=1 --head=12

# Update model with the unmodified weights. The model will be saved in a different directory.
uv run python ./nl_dpe/model.py update-attention-weights --model-path=./models/blushing-dove-984/ --weights-file=att-matrices.pkl --head=12 --output-dir=./models/blushing-dove-984-updated

# Evaluate this model to confirm the original performance
uv run python ./nl_dpe/evaluate.py cola ./models/blushing-dove-984-updated ./datasets/glue/CoLA
#   eval_loss = 0.5854980764967023, mcc = 0.3118300074953714


# Now, remove the saved model and add Gaussian noise to the serialized attention weights.
rm -r ./models/blushing-dove-984-updated
uv run python ./nl_dpe/model.py add-noise-to-attention-weights --weights-file=att-matrices.pkl --mean=0.0 --std=0.1

# Update the model again with slightly different attention weights
uv run python ./nl_dpe/model.py update-attention-weights --model-path=./models/blushing-dove-984/ --weights-file=att-matrices.pkl --head=12 --output-dir=./models/blushing-dove-984-updated

# Confirm that evaluation metics are different
uv run python ./nl_dpe/evaluate.py cola ./models/blushing-dove-984-updated ./datasets/glue/CoLA
#   eval_loss = 0.59219029545784, mcc = 0.29623405262647357
```

The `add-noise-to-attention-weights` command above is used only for testing purposes. The file with weights must always
have the following properties:

- It must be a pickle file containing Python dictionary.
- Keys in that dictionary must be valid tensor names in the original model. Users, when modifying saved weights file
  must keep these names as is.
- There must be 6 entries in this dictionary - weights and biases for K, Q and V matrices. Biases are 1D numpy arrays,
  while wights are 2D numpy arrays.
- Given models this repository provides, the weights arrays have `(26, 26)` shape, while biases arrays have `(26,)`
  shape.

## Hybrid inference: weights of self-attention head

The inference code in this project supports hybrid inference when different layers of the model run on different compute devices.
Concretely, the inference code supports computing `K`, `Q` and `V` weights of one self-attention head on an external device, while
majority of the model runs on a CPU in a host system. It can be achieved following these steps:

- Extract self-attention weights (`W_k`, `W_q` and `W_v`) using procedures outlined above.
- Program or transfer these self-attention weights to an external compute devices.
- Change the [dpe.py](./nl_dpe/dpe.py) implementation. Concretely:
  - Update `_BERT_LAYER_INDEX` and `_ATTENTION_HEAD_INDEX` variables to match the self-attention weights that have been exported. These variables provide brief doc strings.
  - Implement the `call` method in that file. We provide a placeholder implementation that demonstrates one possible implementation.
- Run model inference using CLI described above. The inference code will detect if this functionality is enabled and for what self-attention head. The inference code then will call the `dpe.py::call` function and will use its returned three matrices instead of those computed by the model itself. For implementation details on how the `call` method is called during the inference, see the `BertSelfAttention::forward` method in the  [modelling.py](./transformer/modeling.py) file. The relevant code segment starts with `if dpe.enabled(self.bert_layer_index)` line.
