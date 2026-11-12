"""Cross-stage invariants for checkpoint -> collection -> actor continuation."""


def validate_continuation(saved, payload, world_size, *, allow_world_size_change=False):
    if saved['policy_version'] != payload['policy_version']:
        raise ValueError('Resume policy version differs from collected trajectories')
    if 'cursor_before' not in payload:
        raise ValueError('Resumed batch lacks its collection start cursor')
    if saved['cursor'] != payload['cursor_before']:
        raise ValueError('Resumed collection did not begin at the saved cursor')
    for key in ('fingerprint', 'seed', 'epochs', 'shuffle'):
        if saved['cursor'][key] != payload['cursor'][key]:
            raise ValueError(f'Data configuration changed: {key}')
    before, after = saved['cursor'], payload['cursor']
    if (after['epoch'], after['offset']) <= (before['epoch'], before['offset']):
        raise ValueError('Collected batch did not advance the data cursor')
    if len(saved['rng_by_rank']) != world_size and not allow_world_size_change:

        raise ValueError('Exact RNG resume requires the same world size')


def restore_rng(saved, world_size, rank, *, allow_world_size_change=False, seed=42):
    """Exact RNG restore for same world; explicit deterministic reseed on reshard."""
    import hashlib
    import random
    import torch
    old_world=len(saved['rng_by_rank'])
    if not 0<=rank<world_size:raise ValueError('Invalid rank')
    if old_world==world_size:
        rng=saved['rng_by_rank'][rank]
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python'])
        return dict(mode='exact',old_world_size=old_world,world_size=world_size,rank=rank)
    if not allow_world_size_change:raise ValueError('World size changed without explicit reshard permission')
    material=f'{seed}:{saved["policy_version"]}:{old_world}:{world_size}:{rank}'.encode()
    rank_seed=int.from_bytes(hashlib.sha256(material).digest()[:8],'big')%(2**63-1)
    torch.manual_seed(rank_seed);torch.cuda.manual_seed(rank_seed);random.seed(rank_seed)
    return dict(mode='deterministic_reseed_after_reshard',base_seed=seed,rank_seed=rank_seed,
                old_world_size=old_world,world_size=world_size,rank=rank)
