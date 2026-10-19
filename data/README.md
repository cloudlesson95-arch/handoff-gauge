# Data

| File | Rows | What it is |
|---|---|---|
| `calibration.jsonl` | 120 | Banking77 messages, 12 per intent, from the training split. Used only to fit calibration maps and the threshold |
| `test.jsonl` | 60 | Banking77 messages, 6 per intent, from the test split. Used only for reporting |
| `adversarial.jsonl` | 8 | Synthetic messages written for this project: fake PII, prompt injection, off-topic, empty |
| `llm_cache.jsonl` | | Cached model responses (keyed by model, prompt, temperature and seed) so CI and the regression gate replay without an API key |
| `examples.json` | 6 | Precomputed decisions and spans for the demo's example buttons and its fallback |

`scripts/prepare_data.py` builds the first three files with seed 7. Texts are unchanged apart
from trimmed whitespace, and labels are lowercased.

## Attribution

Banking77 by PolyAI: Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson and
Ivan Vulić, "Efficient Intent Detection with Dual Sentence Encoders", Proceedings of the 2nd
Workshop on NLP for ConvAI, 2020. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source:
<https://github.com/PolyAI-LDN/task-specific-datasets>. This project uses a 180-message
subset of 10 of its 77 intents.

The adversarial messages are made up: the card number is a standard test number, the phone
number is in a range reserved for fiction, and the email uses `example.com`.
