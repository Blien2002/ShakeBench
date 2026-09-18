"""Opt-in continuous chunk inference; existing synchronous policies are unchanged.

A spawned process predicts from the latest observation while the simulator steps.
Chunks carry their observation index: expired actions are discarded, and a newer
chunk replaces overlapping future actions. History is sampled at simulation rate.
"""

import multiprocessing as mp
import time
from collections import deque
from queue import Empty, Full

import numpy as np

from robosuite.utils.shakebench_pretrained_policy import PretrainedPolicy, policy_class, validate_offsets
from robosuite.utils.shakebench_rollout import CAMERAS, PolicyOutputError, PolicyTimeoutError, validated_actions


class ChunkPolicy(PretrainedPolicy):
    """Use the external model's full chunk API, bypassing its open-loop action queue."""

    def predict(self, observation):
        import torch

        batch = {}
        frames = observation["observation_history"]
        for key in self.model.config.input_features:
            values = np.stack([frame[key] for frame in frames])
            if not np.isfinite(values).all():
                raise PolicyOutputError(f"non-finite input: {key}")
            if key in CAMERAS:
                if values.dtype != np.uint8 or values.ndim != 4 or values.shape[-1] != 3:
                    raise ValueError(f"expected uint8 RGB image history: {key}")
                tensor = torch.from_numpy(values).permute(0, 3, 1, 2).float() / 255
            else:
                if values.shape != (len(self.offsets), 8):
                    raise ValueError("expected native 8D state history")
                tensor = torch.as_tensor(values, dtype=torch.float32)
            batch[key] = tensor[None].to(self.device)
        batch["task"] = [observation["task"]]
        with torch.inference_mode():
            actions = self.model.predict_action_chunk(batch).detach().cpu().numpy()
        if actions.shape != (1, self.model.config.chunk_size, 7) or not np.isfinite(actions).all():
            raise PolicyOutputError("model must return a finite [1, chunk_size, 7] chunk")
        return validated_actions(np.clip(actions[0], -1, 1))


def make_chunk_policy(*, checkpoint, device="cuda"):
    return ChunkPolicy(checkpoint=checkpoint, device=device)


def _offer(queue, value):
    """Bound the mailbox; never wait for an inference or a slow consumer."""
    try:
        queue.put_nowait(value)
        return
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            # multiprocessing's feeder may not have published the queued item yet.
            return
    try:
        queue.put_nowait(value)
    except Full:
        pass


def _worker(factory, arguments, incoming, outgoing, time_scale):
    try:
        policy = policy_class(factory)(**arguments)
        outgoing.put({"kind": "ready", "offsets": getattr(policy, "offsets", [0])})
        while True:
            request = incoming.get()
            # The latest camera frame wins. Reset/warmup are sent only while idle.
            while True:
                try:
                    request = incoming.get_nowait()
                except Empty:
                    break
            generation, index, observation = request
            if observation is None:
                reset = getattr(policy, "reset", None)
                if callable(reset):
                    reset()
                outgoing.put({"kind": "reset", "generation": generation})
                continue
            started = time.perf_counter()
            actions = validated_actions(policy.predict(observation))
            inference_s = time.perf_counter() - started
            # Scale measured inference latency with the declared simulation wall-time scale.
            time.sleep(inference_s * (time_scale - 1))
            _offer(outgoing, {
                "kind": "chunk", "generation": generation, "index": index,
                "actions": actions, "inference_s": inference_s,
            })
    except BaseException as exc:
        outgoing.put({
            "kind": "error", "error": f"{type(exc).__name__}: {exc}",
            "output_violation": isinstance(exc, PolicyOutputError),
        })


class AsyncPolicy:
    """Nonblocking control-rate facade over a continuous inference worker.

    Zero OSC delta holds the arm and gripper target when no current action exists;
    repeating the previous delta would keep moving the robot without a prediction.
    """

    chunk_size = 1

    def __init__(self, factory, arguments, *, time_scale=1.0, timeout_s=10.0, startup_timeout_s=300.0):
        for name, value in (("time_scale", time_scale), ("timeout_s", timeout_s),
                            ("startup_timeout_s", startup_timeout_s)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if time_scale < 1:
            raise ValueError("time_scale must be >= 1 (wall seconds per simulation second)")
        self.time_scale, self.timeout_s = time_scale, timeout_s
        self.generation = 0
        context = mp.get_context("spawn")
        self.incoming, self.outgoing = context.Queue(2), context.Queue(2)
        self.process = context.Process(
            target=_worker, args=(factory, arguments, self.incoming, self.outgoing, time_scale), daemon=True,
        )
        self.process.start()
        try:
            ready = self._wait("ready", startup_timeout_s)
            self.offsets = validate_offsets(ready["offsets"])
        except BaseException:
            self.close()
            raise
        self.identity = {"adapter": "async_chunks", "factory": factory, "arguments": arguments,
                         "observation_offsets": self.offsets}

    def _check(self, message):
        if message["kind"] == "error":
            error = PolicyOutputError if message.get("output_violation") else RuntimeError
            raise error(message["error"])
        return message

    def _wait(self, kind, timeout):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                message = self._check(self.outgoing.get(timeout=min(0.1, max(0.001, deadline - time.perf_counter()))))
            except Empty:
                if not self.process.is_alive():
                    raise RuntimeError("inference worker exited")
                continue
            if message["kind"] == kind and message.get("generation", self.generation) == self.generation:
                return message
        raise PolicyTimeoutError(f"inference worker did not return {kind} within {timeout}s")

    def reset(self):
        self.generation += 1
        self.history = deque(maxlen=1 - min(self.offsets))
        self.pending = {}
        self.index = 0
        self.stats = {"completed_chunks": 0, "expired_actions": 0, "fallback_steps": 0,
                      "inference_wall_time_s": 0.0, "max_observation_age_steps": 0}
        # Retire in-flight work before allowing a new episode's observations.
        while True:
            try:
                self.incoming.get_nowait()
            except Empty:
                break
        self.incoming.put((self.generation, 0, None), timeout=self.timeout_s * self.time_scale)
        self._wait("reset", self.timeout_s * self.time_scale)

    def _observation(self, observation):
        # Copy before handing data to the multiprocessing feeder.
        frame = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in observation.items()}
        self.history.append(frame)
        selected = [self.history[max(0, len(self.history) - 1 + offset)] for offset in self.offsets]
        return {**frame, "index": self.index, "observation_history": selected}

    def prime(self, observation):
        """Warm up on the reset observation before starting the episode clock."""
        request = self._observation(observation)
        self.incoming.put((self.generation, 0, request), timeout=self.timeout_s * self.time_scale)
        message = self._wait("chunk", self.timeout_s * self.time_scale)
        self._merge(message)
        self.last_response = time.perf_counter()

    def _merge(self, message):
        if message["generation"] != self.generation:
            return
        actions = validated_actions(message["actions"])
        start = message["index"]
        if start > self.index:
            raise PolicyOutputError("chunk is anchored in the future")
        age = self.index - start
        self.stats["expired_actions"] += min(age, len(actions))
        self.stats["max_observation_age_steps"] = max(self.stats["max_observation_age_steps"], age)
        self.stats["completed_chunks"] += 1
        self.stats["inference_wall_time_s"] += message["inference_s"]
        # Latest chunk replaces only its covered future indices; retain any older tail.
        self.pending.update({start + i: action for i, action in enumerate(actions) if start + i >= self.index})
        self.last_response = time.perf_counter()

    def predict(self, observation):
        if self.index:
            request = self._observation(observation)
            _offer(self.incoming, (self.generation, self.index, request))
        while True:
            try:
                message = self._check(self.outgoing.get_nowait())
            except Empty:
                break
            if message["kind"] == "chunk":
                self._merge(message)
        if not self.process.is_alive():
            raise RuntimeError("inference worker exited")
        if time.perf_counter() - self.last_response > self.timeout_s * self.time_scale:
            raise PolicyTimeoutError("continuous inference stopped producing chunks")
        action = self.pending.pop(self.index, None)
        if action is None:
            self.stats["fallback_steps"] += 1
            action = np.zeros(7, dtype=np.float32)
        self.index += 1
        return validated_actions(action)

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)
        for queue in (self.incoming, self.outgoing):
            queue.cancel_join_thread()
            queue.close()


class PacedTask:
    """Keep the existing rollout/error contract, adding an explicit simulation clock."""

    def __init__(self, task, policy, *, time_scale=1.0):
        self.task, self.policy = task, policy
        self.period = time_scale / 20
        self.overruns = 0
        self.max_overrun_s = 0.0
        self.wall_time_s = 0.0

    def __getattr__(self, name):
        return getattr(self.task, name)

    def reset(self):
        observation, info = self.task.reset()
        self.policy.prime(observation)
        self.started = self.tick = time.perf_counter()
        return observation, info

    def step(self, action):
        result = self.task.step(action)
        remaining = self.tick + self.period - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            self.overruns += 1
            self.max_overrun_s = max(self.max_overrun_s, -remaining)
        # No catch-up bursts after a slow render/physics step.
        self.tick = time.perf_counter()
        self.wall_time_s = self.tick - self.started
        return result
