"""Checkpoint cadence for a live actor and collector; no model reload between updates.

Only export() may advance durable state. A process failure therefore replays at
most save_freq-1 updates from the last full checkpoint. GPU/Ray concerns belong
to the adapter, which must keep the same actor and cursor in the callbacks.
"""
from dataclasses import dataclass


@dataclass
class ResidentProgress:
    policy_version: int
    durable_version: int
    optimizer_steps: int = 0
    durable_optimizer_steps: int = 0


class ResidentLoop:
    def __init__(self, *, policy_version, optimizer_steps=0, save_freq=5):
        if save_freq<1 or policy_version<0 or optimizer_steps<0:
            raise ValueError('Invalid resident checkpoint cadence')
        self.progress=ResidentProgress(policy_version,policy_version,optimizer_steps,optimizer_steps)
        self.save_freq=save_freq
        self.pending=None
        self.failed=False

    def run(self, collect, update, export, *, max_updates=0):
        """Callbacks run synchronously on every rank (adapter owns communication).

        collect(version) returns a validated payload or None at exhaustion.
        update(payload) returns metrics after updating the live actor.
        export(payload, metrics) must atomically export and commit model, Adam,
        RNG and cursor; returning means durable publication has succeeded.
        """
        if self.failed:raise RuntimeError('Failed resident loop cannot be reused; restore durable checkpoint')
        if max_updates<0:raise ValueError('Invalid update limit')
        try:
            while not max_updates or self.progress.policy_version<max_updates:
                payload=collect(self.progress.policy_version)
                if payload is None:break
                if payload['policy_version']!=self.progress.policy_version:
                    raise ValueError('Collection policy version differs from live actor')
                metrics=update(payload)
                if metrics['policy_version']!=self.progress.policy_version+1:
                    raise ValueError('Actor must advance exactly one policy version')
                steps=metrics['optimizer_steps']
                if not isinstance(steps,int) or steps<1:raise ValueError('Update must contain optimizer steps')
                self.progress.policy_version=metrics['policy_version']
                self.progress.optimizer_steps+=steps
                self.pending=(payload,dict(metrics))
                if self.progress.policy_version-self.progress.durable_version>=self.save_freq:
                    self.flush(export)
            self.flush(export)
            return self.progress
        except BaseException:
            # The live actor may have changed even if a callback failed before
            # returning. Never retry from these in-memory objects after failure.
            self.failed=True
            raise

    def flush(self,export):
        if self.pending is None:return
        payload,metrics=self.pending
        metrics.update(optimizer_steps=self.progress.optimizer_steps-self.progress.durable_optimizer_steps,
                       cumulative_optimizer_steps=self.progress.optimizer_steps,
                       previous_durable_policy_version=self.progress.durable_version)
        export(payload,metrics)
        self.progress.durable_version=self.progress.policy_version
        self.progress.durable_optimizer_steps=self.progress.optimizer_steps
        self.pending=None
