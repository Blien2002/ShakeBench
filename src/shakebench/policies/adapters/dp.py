"""Diffusion Policy adapter extension point."""


class DiffusionPolicyAdapter:
    requires_privileged: tuple[str, ...] = ()

    def __init__(self, policy) -> None:
        self.policy = policy

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def act(self, observation):
        return self.policy(observation)
