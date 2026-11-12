"""Offset-only episode index and rank-local, consumption-aware schedules."""
import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path

from .bank import integrate_actions


def build_episode_index(path, summaries=True, sub_summaries_path=None):
    episodes = OrderedDict()
    previous = {}
    with open(path, 'rb') as handle, open(path, 'rb') as lookup:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            uid = f"{row['dataset']}::{row['episode_id']}"
            step = row['step_index']
            actions = row['history_actions']
            images = row['images']
            if len(actions) != step or len(images) != step + 1:
                raise ValueError(f'{uid}:{step}: action/image time mismatch')
            integrate_actions(actions)
            old_offset = previous.get(uid)
            old = None
            if old_offset is not None:
                lookup.seek(old_offset)
                old = json.loads(lookup.readline())
            if old is not None:
                if step < old['step_index'] or row['instruction'] != old['instruction']:
                    raise ValueError(f'{uid}: unordered steps or inconsistent instructions')
                if actions[:old['step_index']] != old['history_actions'] or images[:len(old['images'])] != old['images']:
                    raise ValueError(f'{uid}: inconsistent trajectory prefixes')
            previous[uid] = offset
            episodes.setdefault(uid, []).append(dict(offset=offset, step=step, task='navigation'))
    for uid, entries in episodes.items():
        with open(path, 'rb') as lookup:
            lookup.seek(previous[uid])
            row = json.loads(lookup.readline())
        if entries[0]['step'] != 0:
            raise ValueError(f'{uid}: missing episode start')
        if summaries and row['dataset'] in ('r2r', 'rxr'):
            if not row['action_sequence'] or row['action_sequence'][0] != 'stop':
                raise ValueError(f'{uid}: full summary requires a terminal observation')
            entries.append(dict(offset=entries[-1]['offset'], step=entries[-1]['step'], task='summary'))
    if sub_summaries_path:
        with open(sub_summaries_path) as labels:
            for line in labels:
                annotation = json.loads(line)
                uid = f"{annotation['dataset']}::{annotation['episode_id']}"
                if annotation['dataset'] not in ('r2r', 'rxr') or uid not in episodes:
                    raise ValueError('Sub-summary must reference an indexed expert episode')
                text = annotation['text'].strip()
                if not text or not annotation.get('alignment_verified', False):
                    raise ValueError('Sub-summary requires nonempty text and verified frame alignment')
                candidates = [e for e in episodes[uid] if e['task'] == 'navigation' and e['step'] == annotation['step_index']]
                if len(candidates) != 1:
                    raise ValueError('Sub-summary must reference one exact supervised observation')
                episodes[uid].append(dict(candidates[0], task='summary', summary_text=text))
        for entries in episodes.values():
            entries.sort(key=lambda e: (e['step'], e['task'] != 'navigation'))
    stat = Path(path).stat()
    return dict(version=1, source=str(Path(path).resolve()), size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, summaries=summaries,
                sub_summaries_source=str(Path(sub_summaries_path).resolve()) if sub_summaries_path else None,
                sub_summaries_mtime_ns=Path(sub_summaries_path).stat().st_mtime_ns if sub_summaries_path else None, episodes=episodes)


class EpisodeSchedule:
    """Does not advance progress on prefetch. Call consume after backward."""
    def __init__(self, index, rank=0, world_size=1, seed=42, epoch=0):
        if not 0 <= rank < world_size:
            raise ValueError('Invalid rank/world size')
        self.index, self.rank, self.world_size = index, rank, world_size
        self.seed, self.epoch, self.consumed = seed, epoch, 0
        keys = list(index['episodes'])
        random.Random(seed + epoch).shuffle(keys)
        shards = [keys[r::world_size] for r in range(world_size)]
        lengths = [sum(len(index['episodes'][k]) for k in shard) for shard in shards]
        limit = min(lengths)
        if limit == 0:
            raise ValueError('Not enough complete episode streams for this world size')
        self.truncated_counts = [n - limit for n in lengths]
        self.items = []
        for key in shards[rank]:
            rows = index['episodes'][key]
            for i, entry in enumerate(rows):
                self.items.append(dict(entry, episode_uid=f'{key}::epoch{epoch}',
                    episode_start=i == 0, episode_end=i == len(rows) - 1))
        self.items = self.items[:limit]
        self.fingerprint = hashlib.sha256(json.dumps(self.items, sort_keys=True).encode()).hexdigest()

    def __iter__(self):
        yield from self.items[self.consumed:]

    def __len__(self):
        return len(self.items) - self.consumed

    def consume(self, item):
        if self.consumed >= len(self.items) or item != self.items[self.consumed]:
            raise ValueError('Consumption does not match the next scheduled sample')
        self.consumed += 1

    def state_dict(self):
        return dict(fingerprint=self.fingerprint, rank=self.rank, world_size=self.world_size, seed=self.seed,
                    epoch=self.epoch, consumed=self.consumed,
                    source=self.index['source'], size=self.index['size'], mtime_ns=self.index['mtime_ns'])

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in expected:
            if key != 'consumed' and state[key] != expected[key]:
                raise ValueError(f'Stream resume mismatch: {key}; start a new phase instead')
        if not 0 <= state['consumed'] <= len(self.items):
            raise ValueError('Invalid stream cursor')
        self.consumed = state['consumed']


def select_episode_subset(index, count, seed):
    if count is None:
        return index
    if count < 1:
        raise ValueError('episode_subset must be positive')
    keys = list(index['episodes'])
    random.Random(seed).shuffle(keys)
    return dict(index, episodes={key: index['episodes'][key] for key in keys[:count]})


def training_budget(index, world_size, seed, epochs, accumulation, max_updates=None):
    if epochs < 1 or accumulation < 1:
        raise ValueError('epochs and accumulation must be positive')
    lengths = [len(EpisodeSchedule(index, 0, world_size, seed, epoch)) for epoch in range(epochs)]
    available = sum(n // accumulation for n in lengths)
    if max_updates is not None and (max_updates < 1 or max_updates > available):
        raise ValueError(f'Update budget {max_updates} exceeds available {available}; increase the episode subset or epochs')
    if available < 1:
        raise ValueError('Insufficient samples for one global batch')
    return max_updates if max_updates is not None else available
