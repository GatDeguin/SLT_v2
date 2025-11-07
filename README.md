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