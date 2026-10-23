# Training

## Training Config

All training is controlled by a JSON training config file and a JSON data config file. 

See [examples/config/](../examples/config/) for ready-to-use configs.

Training config file on Emilia is: [examples/config/train_config_emilia.json](../examples/config/train_config_emilia.json)

Data config file for Emilia is: [examples/config/data_config_emilia.json](../examples/config/data_config_emilia.json)


Key fields in training config file:

| Field | Description | Default |
|---|---|---|
| `llm_name_or_path` | local LLM path or huggingface id | Qwen/Qwen3-0.6B |
| `steps` | Total training steps | 300,000 |
| `learning_rate` | Peak learning rate | 1e-4 |
| `batch_tokens` | Tokens per batch on each GPU | 8192 |

`output_dir` and `data_config` are passed via command line (see below).

## Launching Training

```bash
accelerate launch \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --num_processes 8 \
    -m omnivoice.cli.train \
    --train_config config/train_config_emilia.json \
    --data_config config/data_config_emilia.json \
    --output_dir exp/omnivoice_emilia
```

## Resuming Training

Set `resume_from_checkpoint` in your training config to resume from an existing checkpoint:

```json
{
    "resume_from_checkpoint": "exp/omnivoice/checkpoint-100000"
}
```

## Initializing from a Pretrained Model

To start training from a pretrained VoiceStudio checkpoint (for fine-tuning):

```json
{
    "init_from_checkpoint": "exp/omnivoice/checkpoint-100000"
}
```

## LoRA adapters (fine-tuning on one voice)

Full fine-tuning needs roughly 10 GB of GPU memory before activations. A LoRA adapter trains about
10 million parameters instead and writes a small adapter next to the unchanged base model. Switch it on
in the training config; everything else (data, launch command) stays the same:

| Field | Description | Default |
|---|---|---|
| `lora_enabled` | Wrap the backbone's attention and MLP layers with LoRA and freeze the rest | `false` |
| `lora_r` / `lora_alpha` / `lora_dropout` | Rank, scale and dropout of the update | 16 / 32 / 0.05 |
| `lora_target_modules` | Layer names to wrap | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| `lora_train_modules` | Extra top-level modules trained in full next to the adapter, e.g. `["audio_heads"]` | `[]` |
| `eval_at_start` | Run the held-out evaluation once before training, so the base model's loss is on record | `false` |

The adapter code is in [`omnivoice/training/lora.py`](../omnivoice/training/lora.py) (plain PyTorch, no `peft`).
Each checkpoint writes `checkpoint-N/adapter/` (`adapter_model.safetensors` + `adapter_config.json`) instead of a
full model, plus `eval_history.jsonl` in the output folder with one line per evaluation. Resuming works as before
(`resume_from_checkpoint`). `lora.merge_adapter(model)` folds an adapter into the base weights.

VoiceStudio's Fish **Train** dialog builds these configs, tokenises the audio and runs the trainer for you; see
[engines/fish-audio.md](engines/fish-audio.md).

## Monitoring

Training logs to TensorBoard:
```bash
tensorboard --logdir exp/omnivoice_emilia/tensorboard
```
