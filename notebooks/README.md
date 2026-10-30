# Notebooks

- **`Qwen3_TTS_Colab_Training.ipynb`**: the Qwen3-TTS fine-tuning notebook (1.7B Base is the focus). Use this one; do not keep dated copies.
- `OmniVoice_Studio_Colab.ipynb`: OmniVoice Studio on Colab.

The Qwen notebook is **generated** by `render_notebook(...)` in `backend/engines/fish_api/qwen3_colab.py` (the training export writes a fresh copy; `write_default_notebook` refreshes this one, and a test keeps it equal to the generator). Do not hand-edit it: change the generator and regenerate.

Settings (learning rate, epochs, batch size, `GRAD_ACCUM`, `FLASH_WHEEL`, Drive folder, `SMOKE_RUN`) live in the first code cell, the parameters cell. `TRAIN_MODE = 'lora'` (with `LORA_LR`, `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_SCALE`) trains adapters instead of the whole model; it is unverified on a GPU; defaults are lr 2e-6, rank 16, alpha 32, scale 0.3. `SPEAKERS` trains several speakers (at most 4) in one model.

Colab: upload the notebook and the dataset zip (to Drive `Qwen3TTS/datasets/`), pick an A100 runtime, set the parameters cell, then Run all. Details: `docs/engines/qwen3-tts-colab-runtime.md`.

Every export also contains `train_local.sh` / `train_local.ps1` (the same runner on your own GPU, not Colab) and a bundle id shown in the notebook, the scripts, README.txt and manifest.json; if they differ, use the newest export.
