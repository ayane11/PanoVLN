"""Detached episode storage and trainable retrieval, following MemVLN.

Storage mutations are separate from checkpointed tensor computation. All time
indices are physical observation indices and all yaw values are degrees.
"""
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .position_embedding import PositionEmbedding
from .tome.tome import ToMeBlock
from .tome.merge import bipartite_soft_matching, merge_wavg


def integrate_actions(actions):
    x = y = yaw = 0.0
    for action in actions:
        if action in (1, 'forward'):
            x += .25 * math.sin(math.radians(yaw))
            y += .25 * math.cos(math.radians(yaw))
        elif action in (2, 'left'):
            yaw = (yaw - 15 + 180) % 360 - 180
        elif action in (3, 'right'):
            yaw = (yaw + 15 + 180) % 360 - 180
        else:
            raise ValueError(f'Invalid nonterminal executed action: {action!r}')
    return (x, y, yaw)


class RetrievalBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, memory):
        x = self.norm1(query + self.attn(query, memory, memory, need_weights=False)[0])
        return self.norm2(x + self.ffn(x))


class EpisodeMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.settings = dict(config)
        dim = config.get('feature_dim', 2560)
        heads = config.get('heads', 16)
        self.capacity = config.get('bank_frames', 100)
        self.frame_tokens = config.get('frame_tokens', 100)
        self.chunk_frames = config.get('chunk_frames', 4)
        self.mode = config.get('consolidation', 'frme')
        if self.mode not in ('fifo', 'frme') or min(self.capacity, self.frame_tokens, self.chunk_frames) < 1:
            raise ValueError('Invalid memory capacity/compression settings')
        self.use_pose = config.get('use_pose', True)
        self.use_time = config.get('use_time', True)
        self.checkpoint_blocks = config.get('checkpoint_blocks', True)
        self.latents = nn.Parameter(torch.randn(1, config.get('tokens', 300), dim) * .02)
        self.pos_embed = PositionEmbedding('fourier', dim, 32, True, config.get('pose_hidden', 512), 'rff')
        self.tome = nn.ModuleList([ToMeBlock(dim, heads) for _ in range(config.get('compression_layers', 2))])
        self.retrieval = nn.ModuleList([RetrievalBlock(dim, heads) for _ in range(config.get('retrieval_layers', 2))])
        self.last_written = {}
        self.banks = {}
        self.grounded = {}

    def reset(self, episode=None):
        if episode is None:
            self.last_written.clear()
            self.banks.clear()
            self.grounded.clear()
        else:
            self.last_written.pop(episode, None)
            self.banks.pop(episode, None)
            self.grounded.pop(episode, None)

    def train(self, mode=True):
        self.grounded.clear()
        return super().train(mode)

    def _apply(self, fn):
        self.grounded.clear()
        self.pos_embed._cached_param_dtype = None
        self.pos_embed._cached_param_device = None
        return super()._apply(fn)

    @torch.no_grad()
    def write(self, episode, step, odom, features):
        step = int(step)
        bank = self.banks.setdefault(episode, OrderedDict())
        previous_step = self.last_written.get(episode, -1)
        if step == previous_step:
            return
        if step < previous_step:
            raise ValueError(f'Out-of-order memory write in {episode}: {step}')
        if bank and features.shape != next(iter(bank.values()))[1].shape:
            raise ValueError('FRME requires a fixed current-observation token grid')
        bank[step] = (torch.as_tensor(odom).detach().clone(), features.detach().clone())
        self.last_written[episode] = step
        cache = self.grounded.setdefault(episode, {})
        while len(bank) > self.capacity:
            if self.mode == 'fifo':
                old, _ = bank.popitem(last=False)
                cache.pop(old, None)
            else:
                keys = list(bank)
                feats = torch.stack([v[1] for v in bank.values()])
                idx = int(F.cosine_similarity(feats[:-1], feats[1:], dim=-1).mean(-1).argmax())
                a, b = keys[idx:idx + 2]
                bank[a] = (bank[a][0], (bank[a][1] + bank[b][1]) * .5)
                del bank[b]
                cache.pop(a, None)
                cache.pop(b, None)

    def time_embedding(self, steps):
        dim = self.latents.shape[-1]
        freq = torch.exp(torch.arange(0, dim, 2, device=steps.device).float() * (-math.log(10000.) / dim))
        angles = steps.float().unsqueeze(-1) * freq
        out = torch.zeros((*steps.shape, dim), device=steps.device)
        out[..., 0::2] = angles.sin()
        out[..., 1::2] = angles[..., :dim // 2].cos()
        return out.to(self.latents.dtype)

    def _compress(self, x):
        for block in self.tome:
            def apply(value, block=block):
                norm = block.norm1(value)
                b, n, c = norm.shape
                h = block.attn.num_heads
                q, k, v = block.attn.qkv(norm).reshape(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4).unbind(0)
                out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
                value = value + block.attn.proj(out)
                remove = min(n // 2, max(0, n - self.frame_tokens))
                if remove:
                    merge, _ = bipartite_soft_matching(k.mean(1), remove, deterministic=True)
                    value, _ = merge_wavg(merge, value, None)
                return block.forward_mlp(value)
            x = checkpoint(apply, x, use_reentrant=False) if self.training and self.checkpoint_blocks else apply(x)
            if x.shape[1] <= self.frame_tokens:
                break
        if x.shape[1] > self.frame_tokens:
            raise ValueError('Insufficient ToMe layers for the current token grid; increase compression_layers explicitly')
        return x

    def read(self, episode, step, odom):
        device, dtype = self.latents.device, self.latents.dtype
        bank = self.banks.get(episode, {})
        if not bank:
            raise ValueError(f'Cannot read empty memory for {episode}; replay the episode first')
        if self.last_written[episode] > int(step):
            raise ValueError('Memory contains future observations')
        keys = list(bank)
        cache = self.grounded.setdefault(episode, {})
        grounded = []
        for start in range(0, len(keys), self.chunk_frames):
            chunk = keys[start:start + self.chunk_frames]
            missing = chunk if self.training else [k for k in chunk if k not in cache]
            fresh = {}
            if missing:
                raw = torch.stack([bank[k][1].to(device=device, dtype=dtype) for k in missing])
                values = self._compress(raw)
                if self.use_pose:
                    values = values + self.pos_embed(torch.stack([bank[k][0].to(device) for k in missing])).unsqueeze(1)
                if self.use_time:
                    values = values + self.time_embedding(torch.tensor(missing, device=device)).unsqueeze(1)
                fresh = dict(zip(missing, values.unbind(0)))
                if not self.training:
                    cache.update({k: v.detach() for k, v in fresh.items()})
            grounded.extend(fresh[k] if k in fresh else cache[k] for k in chunk)
        memory = torch.cat(grounded).unsqueeze(0)
        query = self.latents
        if self.use_pose:
            query = query + self.pos_embed(torch.as_tensor(odom, device=device).reshape(1, 3)).unsqueeze(1)
        if self.use_time:
            query = query + self.time_embedding(torch.tensor([step], device=device)).unsqueeze(1)
        for block in self.retrieval:
            query = checkpoint(block, query, memory, use_reentrant=False) if self.training and self.checkpoint_blocks else block(query, memory)
        return query.squeeze(0)

    def export_state(self):
        return dict(version=2, last_written=dict(self.last_written),
            banks={e: [(t, o.cpu(), f.cpu()) for t, (o, f) in b.items()] for e, b in self.banks.items()})

    def import_state(self, state):
        if state.get('version') != 2:
            raise ValueError('Unsupported memory state version')
        self.reset()
        for episode, entries in state['banks'].items():
            self.banks[episode] = OrderedDict((int(t),
                (o.to(device=self.latents.device, dtype=torch.float32),
                 f.to(device=self.latents.device, dtype=self.latents.dtype))) for t, o, f in entries)
        self.last_written.update(state['last_written'])
