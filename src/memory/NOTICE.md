# Attribution

The ToMe implementation (`tome/`) and position embedding originate from
MemVLN/MemVLN commit 48eeaa9620b217f51559eeb94918a7c788de36f0.
The memory bank adapts its detached raw storage, FIFO/FRME consolidation,
pose/time grounding and latent retrieval. The ToMe component is subject to
CC BY-NC 4.0; retain the upstream MemVLN LICENSE and THIRD_PARTY_NOTICES.md
when redistributing this component. Qwen/PanoVLN keep their upstream terms.

Local numerical adaptation: production ToMe matching uses stable tie ordering
and deterministic FP32 matrix accumulation instead of CUDA BF16 atomic scatter
sums, with one final cast after the weighted mean. This preserves the matching
and weighted-mean semantics while making inference cache rebuilds reproducible.
