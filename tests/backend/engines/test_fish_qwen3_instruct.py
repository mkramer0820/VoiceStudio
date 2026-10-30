"""--instruct conditioning of the Qwen3-TTS fine-tune: tag stripping, dropout, and the dynamic prefix/speaker slot of the collator, run through
the REAL pinned dataset.py (qwen_tts/librosa stubbed) and the patched sft script text."""
from __future__ import annotations

import ast
import os
import random
import re
import sys
import types

import numpy as np
import pytest
import torch

from engines.fish_api import qwen3_colab_runner as vs

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
HEADER, USER, END, NL, ASSIST = 151644, 872, 151645, 198, 77091


class FakeProcessor:
    """Tokenises like the Qwen chat template does for these strings: specials are single ids, every other word is one id."""
    SPECIAL = {"<|im_start|>": HEADER, "<|im_end|>": END, "\n": NL, "assistant": ASSIST, "user": USER}

    def __call__(self, text, return_tensors="pt", padding=True):
        ids = []
        for tok in re.findall(r"<\|im_start\|>|<\|im_end\|>|\n|assistant|user|[^\s<]+", text):
            ids.append(self.SPECIAL.get(tok, 1000 + sum(map(ord, tok)) % 5000))
        return {"input_ids": torch.tensor([ids])}


def _config():
    tc = types.SimpleNamespace(codec_nothink_id=2155, codec_think_bos_id=2156, codec_think_eos_id=2157, codec_pad_id=2148, codec_bos_id=2149,
                               codec_eos_token_id=2150)
    return types.SimpleNamespace(tts_pad_token_id=151671, tts_bos_token_id=151672, tts_eos_token_id=151673, talker_config=tc)


@pytest.fixture
def dataset_cls(monkeypatch):
    stubs = {"librosa": types.SimpleNamespace(load=lambda *_a, **_k: (np.zeros(24000, dtype=np.float32), 24000)),
             "qwen_tts": types.ModuleType("qwen_tts"), "qwen_tts.core": types.ModuleType("qwen_tts.core"),
             "qwen_tts.core.models": types.ModuleType("qwen_tts.core.models"),
             "qwen_tts.core.models.configuration_qwen3_tts": types.SimpleNamespace(Qwen3TTSConfig=object),
             "qwen_tts.core.models.modeling_qwen3_tts": types.SimpleNamespace(mel_spectrogram=lambda *a, **k: torch.zeros(1, 128, 40))}
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    ns: dict = {"__name__": "dataset_fixture"}
    exec(compile(open(os.path.join(FIXTURES, "dataset_022e286.py"), encoding="utf-8").read(), "dataset_022e286.py", "exec"), ns)
    base = ns["TTSDataset"]
    ns2 = {"TTSDataset": base, "torch": torch}
    helper = vs.INSTRUCT_PAIRS[0][1].replace("\ntarget_speaker_embedding = None", "")
    exec(helper, ns2)
    return base, ns2["TTSDataset"], ns2


ROWS = [
    {"audio": "a.wav", "text": "[whispering, tense] The shadows are moving... [sigh] stay back.", "audio_codes": [[i] * 16 for i in range(5)], "ref_audio": "r.wav"},
    {"audio": "b.wav", "text": "Nothing moves here at all.", "audio_codes": [[i + 3] * 16 for i in range(4)], "ref_audio": "r.wav"},
]


def test_tags_become_one_flat_deduplicated_csv_and_the_text_is_cleaned():
    assert vs.split_tags("[whispering, tense] Don't move... [sigh] stay right there.") == ("Don't move... stay right there.", "whispering, tense, sigh")
    assert vs.split_tags("stay [sigh]. ok") == ("stay. ok", "sigh")
    assert vs.split_tags("plain words") == ("plain words", "")
    assert vs.flatten_tags("[Whispering]", "whispering, calm", ["a", "[b, c]"]) == "Whispering, calm, a, b, c"
    assert vs.flatten_tags(None, "", "  ") == ""


def test_instruct_dropout_keeps_the_timbre_lock_ratios():
    rnd = random.Random(7)
    tagged = [vs.choose_instruct("whispering, tense", rnd, 0.30) for _ in range(4000)]
    assert 0.26 < tagged.count(None) / len(tagged) < 0.34 and set(tagged) <= {None, "whispering, tense"}
    plain = [vs.choose_instruct("", rnd, 0.30) for _ in range(4000)]
    assert 0.46 < plain.count(None) / len(plain) < 0.54 and set(plain) - {None} == set(vs.INSTRUCT_BASELINES)


def _dump(batch, i, upto):
    ids, lab = batch["input_ids"][i], batch["codec_0_labels"][i]
    rows = []
    for pos in range(upto):
        rows.append((pos, int(ids[pos, 0]), int(ids[pos, 1]), pos == int(batch["speaker_pos"][i]), int(lab[pos]),
                     bool(batch["text_embedding_mask"][i, pos, 0]), bool(batch["codec_embedding_mask"][i, pos, 0])))
    return rows


def test_collator_places_instruction_header_codec_prefix_and_speaker_slot_like_generation(dataset_cls, capsys):
    _, cls, _ = dataset_cls
    ds = cls(ROWS, FakeProcessor(), _config())
    assert ds.data_list[0]["text"] == "The shadows are moving... stay back." and ds.data_list[0]["_vs_instruct"] == "whispering, tense, sigh"
    items = [ds[0], ds[1]]
    items[0]["instruct_ids"] = ds._tokenize_texts("<|im_start|>user\nwhispering, tense, sigh<|im_end|>\n")  # force the conditioned path
    items[1]["instruct_ids"] = None
    L = items[0]["instruct_ids"].shape[1]
    assert L == 3 + 3 + 2  # <|im_start|> user \n, three tags, <|im_end|> \n
    batch = ds.collate_fn(items)
    cfg = _config()
    ids = batch["input_ids"][0]
    assert ids[:L, 0].tolist() == items[0]["instruct_ids"][0].tolist() and not ids[:L, 1].any()
    assert ids[L:L + 3, 0].tolist() == [HEADER, ASSIST, NL]
    assert ids[L + 3:L + 7, 0].tolist() == [cfg.tts_pad_token_id] * 4 and ids[L + 7, 0] == cfg.tts_bos_token_id
    assert ids[L + 3:L + 8, 1].tolist() == [2155, 2156, 2157, 0, 2148]
    assert int(batch["speaker_pos"][0]) == L + 6 and int(batch["speaker_pos"][1]) == 6
    assert not batch["codec_embedding_mask"][0, :L].any() and not batch["codec_embedding_mask"][0, L + 6, 0] and batch["codec_embedding_mask"][0, L + 3, 0]
    assert batch["text_embedding_mask"][0, :L].all()
    assert (batch["codec_0_labels"][0, :L + 8] == -100).all() and not batch["codec_mask"][0, :L + 8].any()
    tl = items[0]["text_ids"].shape[1]
    first = L + 8 + tl - 1
    assert batch["codec_0_labels"][0, first:first + 5].tolist() == [0, 1, 2, 3, 4] and batch["codec_0_labels"][0, first + 5] == cfg.talker_config.codec_eos_token_id
    assert batch["codec_mask"][0, first:first + 5].all() and batch["codec_ids"][0, first + 2].tolist() == [2] * 16
    assert ids.shape == batch["input_ids"][1].shape
    assert batch["input_ids"][1, :3, 0].tolist() == [HEADER, ASSIST, NL] and batch["input_ids"][1, 3:8, 1].tolist() == [2155, 2156, 2157, 0, 2148]
    with capsys.disabled():
        print("\n  sample 0 (instruct: %s)  pos text codec speaker_slot label text_mask codec_mask" % ds.data_list[0]["_vs_instruct"])
        for row in _dump(batch, 0, first + 3):
            print("  %3d %7d %5d %-5s %6d %-5s %s" % row)
        print("  sample 1 (no instruct)")
        for row in _dump(batch, 1, 10):
            print("  %3d %7d %5d %-5s %6d %-5s %s" % row)


def test_a_batch_without_any_instruction_is_identical_to_the_original_collator(dataset_cls):
    base, cls, _ = dataset_cls
    plain = [{**r, "text": "Nothing moves here at all."} for r in ROWS]
    b0 = base(plain, FakeProcessor(), _config())
    b1 = cls(plain, FakeProcessor(), _config())
    items0 = [b0[0], b0[1]]
    items1 = [b1[0], b1[1]]
    for it in items1:
        it["instruct_ids"] = None
    old, new = b0.collate_fn(items0), b1.collate_fn(items1)
    for key in old:
        assert torch.equal(old[key], new[key]), key
    assert new["speaker_pos"].tolist() == [6, 6]


def test_speaker_vector_goes_to_each_samples_own_slot(dataset_cls):
    _, _, ns = dataset_cls
    emb = torch.zeros(2, 12, 4)
    ns["_vs_put_speaker"](emb, {"speaker_pos": torch.tensor([9, 6])}, torch.tensor([[1.0] * 4, [2.0] * 4]))
    assert emb[0, 9].tolist() == [1.0] * 4 and emb[1, 6].tolist() == [2.0] * 4 and emb.sum() == 12.0
    emb = torch.zeros(1, 12, 4)
    ns["_vs_put_speaker"](emb, {}, torch.ones(1, 4))  # a batch from the original collator keeps slot 6
    assert emb[0, 6].tolist() == [1.0] * 4


def test_the_patched_script_applies_exactly_once_compiles_and_the_plain_one_is_unchanged():
    src = open(os.path.join(FIXTURES, "sft_12hz_022e286.py"), encoding="utf-8").read()
    text = src
    for old, new in vs.ALIGN_PAIRS:
        assert text.count(old) == 1
        text = text.replace(old, new)
    plain = text
    for old, new in vs.INSTRUCT_PAIRS + vs.LORA_PAIRS:
        assert text.count(old) == 1, old[:60]
        text = text.replace(old, new)
    ast.parse(text)
    assert text.count("_vs_put_speaker(input_codec_embedding") == 2
    assert "class TTSDataset(_VsBaseDataset)" in text and "_BaseTTSDataset = TTSDataset" in text  # the speaker subclass wraps the instruct one
    ast.parse(plain)
    assert "_vs_put_speaker" not in plain and "_VsBaseDataset" not in plain  # without the flag the script has no trace of it


def test_params_default_off_and_validate_the_dropout():
    p = vs.params_from({}, env={})
    assert p.instruct_conditioning is False and p.instruct_dropout == 0.30
    assert vs.params_from({"instruct_conditioning": "true", "instruct_dropout": 0.2}, env={}).instruct_conditioning is True
    with pytest.raises(vs.StepFailed, match="INSTRUCT_DROPOUT"):
        vs.params_from({"instruct_dropout": 1.5}, env={})
