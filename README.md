---
library_name: transformers
license: cc-by-nc-4.0
---


This is a port of the multilingual SONAR text encoder and decoder (https://huggingface.co/facebook/SONAR) to the `transformers` format from `fairseq2`.

**Developed by:** @shakibyzn and @yhamidullah

**License:** cc-by-nc-4.0

Thanks to @cointegrated for providing the weights for the SONAR encoder part.

## Cite as
```Markdown
@InProceedings{hamidullah-EtAl:2025:WMT,
  author    = {HAMIDULLAH, Yasser  and  Yazdani, Shakib  and  Oguz, Cennet  and  van Genabith, Josef  and  EspaÃ±a-Bonet, Cristina},
  title     = {SONAR-SLT: Multilingual Sign Language Translation via Language-Agnostic Sentence Embedding Supervision},
  booktitle      = {Proceedings of the Tenth Conference on Machine Translation (WMT 2025)},
  month          = {November},
  year           = {2025},
  address        = {Suzhou, China},
  publisher      = {Association for Computational Linguistics},
  pages     = {301--313},
  abstract  = {},
  url       = {https://aclanthology.org/2025.wmt-1.18}
}
```

## Multimodal adapter dependencies

The lightweight `FusionAdapter` (keypoints only) ships with the default project
dependencies (PyTorch, Transformers, NumPy) and is now automatically selected
whenever `tools_finetune_sonar_slt.py` runs without `--video-dir`. This means
you can fine-tune and run inference on keypoints alone without installing
vision backbones such as ViT/VideoMAE.

Supplying paired RGB clips via `--video-dir` switches the trainer/inference to
the multimodal `VisualFusionAdapter`, which requires the additional packages:

```bash
pip install timm transformers[video] opencv-python
```

## Inference defaults for `tools_infer_sonar_slt.py`

The inference helper reads the saved training configuration from adapter checkpoints.
When `--sonar-model-dir` is omitted the script reuses the `model_name` that was used
for training; specifying a different decoder explicitly will raise a warning so you
can double-check the mismatch. Likewise, when `--tgt-lang` is not provided the
checkpoint's `tgt_lang` value becomes the default so that the generated text matches
the fine-tuning target language. If the checkpoint lacks these fields the script
falls back to the previous defaults (`mtmlt/sonar-nllb-200-1.3B` and `spa_Latn`).

Runtime generation can now be tuned from the CLI as well: pass `--do-sample` to
switch to sampling and combine it with knobs such as `--temperature`, `--top-p`,
`--top-k`, `--repetition-penalty`, `--no-repeat-ngram-size`, and
`--length-penalty` to better match your quality/fluency trade-offs.

## Fine-tuning with InfoNCE alignment

The `tools_finetune_sonar_slt.py` helper exposes an optional InfoNCE term that
encourages better alignment between visual (`z`) and textual (`s`) embeddings:

```
L_joint = λ_sem L_sem + λ_ce L_ce + λ_ae L_ae + λ_nce L_nce.
```

Activate it by passing `--lam-nce` and tune the contrastive temperature with
`--nce-temperature`. The implementation maintains a queue of recent sentence
embeddings to provide additional negatives during training.

⚠️ **Memory note:** The fine-tuning helper now loads a second, fully frozen
SONAR decoder that serves as the teacher for `TextPooler`. This guarantees that
LoRA updates only touch the student decoder but requires keeping two copies of
the checkpoint in memory (roughly 2× the original SONAR footprint).

## Monitoring validation losses

`tools_finetune_sonar_slt.py` now accepts a held-out metadata file via
`--dev-csv` as well as an `--eval-every` interval. When both are set the trainer
builds a second `KPTextDataset` that mirrors the preprocessing (T sampling,
frame clipping, and optional video loading) used for the training split so the
semantic (`L_sem`), decoder CE (`L_ce`), auto-encoding (`L_ae`), and InfoNCE
(`L_nce`) losses are comparable. Aggregated dev metrics are appended to
`train_log.jsonl` with `"split": "dev"` and mirrored to `dev_log.jsonl`, letting
you monitor whether CE/AE improvements transfer to your validation clips without
pausing training.

## Uses

```Python
# !pip install transformers

import torch
from transformers import M2M100ForConditionalGeneration
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

model_name = "mtmlt/sonar-nllb-200-1.3B"
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode_mean_pool(texts, tokenizer, encoder, lang='eng_Latn', norm=False):
    tokenizer.src_lang = lang
    with torch.inference_mode():
        batch = tokenizer(texts, return_tensors='pt', padding=True)
        seq_embs = encoder(**batch).last_hidden_state
        mask = batch.attention_mask
        mean_emb = (seq_embs * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)
        if norm:
            mean_emb = torch.nn.functional.normalize(mean_emb)
    return mean_emb, seq_embs

sentences = ['My name is SONAR.', 'I can embed the sentences into vector space.']
embs, last_hidden_state = encode_mean_pool(sentences, tokenizer, model.model.encoder, lang="eng_Latn")
print(embs.shape)
# torch.Size([2, 1024])

target_lang = "eng_Latn"
encoder_outputs = BaseModelOutput(last_hidden_state=embs.unsqueeze(1))
# text reconstruction
generated_tokens = model.generate(
    encoder_outputs=encoder_outputs,  # pass the embedding vectors as seq_len=1
    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
    max_new_tokens=32,
)
generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(generated_texts)
# ['My name is SONAR.', 'I can embedd the sentences into vector space.']
```

For advanced examples of usage, please take a look at the readme in https://github.com/facebookresearch/SONAR.