# Migration from historical Base scripts

## Public entry points

The former trainer referenced missing `save_load.py` and `utils/log_helper.py`, passed unsupported model arguments, used hard-coded data paths, and accumulated an uninitialized loss. Its test mode required undistributed experiment inputs and fixed checkpoint names. The new `model_train.py` provides explicit tensor training and resume; `inference.py` provides tensor imputation.

The former multi-GPU, automatic GPU selection, CSV/VCF experiment-name, rotary, and checkpoint-test-loop flags are not aliases for the new CLI. Consult `--help`. The public recipe uses full precision and constant-rate Adam. It does not reproduce the old masking, scheduler, mixed-precision, or experiment-evaluation settings.

Experimental `utils/kgp_2.py` class names remain for source compatibility. These retain assumptions such as fixed masking policies and caller-supplied MAF fields. The public workflow uses validated tensor dictionaries instead.

## Padding

The fixed vocabulary places `[PAD]` at index 3 and REF at index 0. The historical `pad_token_id=0` marked REF embeddings as padding. New configurations require PAD=3, matching ignored MLM labels. Tensor names and shapes are unchanged. The change affects gradients in new training; loading existing embedding values does not change their numerical values.

## Attention correction

The old `BertAttention.forward` used `self_outputs[0]` after the attention return tuple had already been unpacked. This broadcast the first query's attended vector across all sites. New runs set `legacy_attention_broadcast=false` and use each query's own output.

This changes model computation. Strict weight loading can still succeed because parameter names and shapes are unchanged, but historical weights need independent evaluation under corrected attention. Load compatibility is not evidence of scientific reproduction.

Use `legacy_attention_broadcast=true` explicitly to retain the old numerical attention behavior. The lower-level model defaults to old behavior if the field is absent; the public configuration loader defaults to corrected behavior. This flag creates no trainable parameters.

## Historical checkpoint conversion

The public loader requires `config` and `model_state_dict`; Ref checkpoints are incompatible. For a trusted historical Base plain state dictionary or `state_dict` wrapper, match the architecture and wrap explicitly:

```python
from pathlib import Path
import json
import torch

config = json.loads(Path("bert_1KGPMLM/config.json").read_text())
# Match dimensions, layer count, attention heads, and weight sharing to the old run.
config["legacy_attention_broadcast"] = True
weights = torch.load("historical_base.pt", map_location="cpu", weights_only=True)
state = weights.get("state_dict", weights)
if state and all(key.startswith("module.") for key in state):
    state = {key.removeprefix("module."): value for key, value in state.items()}
torch.save({"config": config, "model_state_dict": state}, "base_for_inference.pt")
```

This conversion is for inference. Resume requires the optimizer, epoch, mask rate, and random state from the new trainer. Strict loading rejects mismatched keys and shapes. Unknown checkpoint formats require a conversion based on their actual schema.

## Source cleanup

All comments, docstrings, messages, and documentation are English. Python bytecode has been removed from the tracked tree and ignore rules exclude caches, environments, genomic data, checkpoints, and generated outputs. Original source APIs remain available without personal experiment paths or debugging transcripts.
