"""Task extension seam for the shared Panda CPU rollout.

Import an extension module to register its TaskDefinition. State files never
import code. Each task owns state validation, environment arguments, language
and execution identity; physics, sensors and scheduling remain environment-owned.
The existing tasks.TaskSpec and make_task_env remain pick-place conveniences.
"""

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskDefinition:
    """Adapter supplied by a task module.

    normalize_state validates and returns a state mapping without changing task
    identity. env_kwargs validates task-specific poses/specs and returns keyword
    arguments for env_factory. describe returns task_id and instruction strings.
    fingerprint returns a JSON-serializable execution identity, excluding IDs
    and split labels but including all poses, goals and random seeds.

    env_factory accepts the shared Panda runtime options. Its environment uses
    robosuite's four-value step, exposes _get_observations(), control_freq,
    action_dim, close(), get_policy_task_context(), policy_observation_keys,
    observation_contract() and get_metrics(). Metrics must contain
    success.passed (a latched bool); task_rule_violation is an optional bool.
    Camera mode additionally requires the existing Panda camera/IMU interface.
    """

    env_factory: Callable[..., Any]
    normalize_state: Callable[[Mapping], Mapping]
    env_kwargs: Callable[[Mapping], dict]
    describe: Callable[[Mapping], dict]
    fingerprint: Callable[[Mapping], Any]


_TASKS: dict[str, TaskDefinition] = {}
_STATE_LOADERS: dict[str, Callable[[Mapping], dict]] = {}


def register_task(task_type: str, definition: TaskDefinition) -> None:
    """Register once; never silently replace an existing task or pick_place."""
    if not isinstance(task_type, str) or not task_type or task_type == "pick_place":
        raise ValueError("extension task_type must be nonempty and not pick_place")
    if task_type in _TASKS:
        raise ValueError(f"task already registered: {task_type}")
    if not isinstance(definition, TaskDefinition) or not all(
        callable(getattr(definition, field)) for field in TaskDefinition.__dataclass_fields__
    ):
        raise TypeError("TaskDefinition requires callable adapters")
    _TASKS[task_type] = definition


def task_type(state: Mapping) -> str:
    """Resolve explicit task identity; only absent task means legacy pick-place."""
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    if "task" not in state:
        return "pick_place"
    spec = state["task"]
    if not isinstance(spec, Mapping) or not isinstance(spec.get("task_type"), str) or not spec["task_type"]:
        raise ValueError("task must declare a nonempty task_type")
    return spec["task_type"]


def get_task_definition(state: Mapping) -> TaskDefinition:
    name = task_type(state)
    if name == "pick_place":
        return _PICK_PLACE
    try:
        return _TASKS[name]
    except KeyError:
        raise ValueError(f"unregistered task {name!r}; import its registration module first") from None


def prepare_task_state(state: Mapping) -> dict:
    """Validate a copy through its task, preserving the caller's source record."""
    name = task_type(state)
    result = dict(get_task_definition(state).normalize_state(deepcopy(state)))
    if task_type(result) != name:
        raise ValueError("state normalization must preserve task_type")
    return result


def describe_task(state: Mapping) -> dict[str, str]:
    result = get_task_definition(state).describe(deepcopy(state))
    if not isinstance(result, Mapping) or set(result) != {"task_id", "instruction"}:
        raise ValueError("task description must contain task_id and instruction")
    if not all(isinstance(value, str) and value.strip() for value in result.values()):
        raise ValueError("task description values must be nonempty strings")
    return dict(result)


def register_state_loader(schema_id: str, loader: Callable[[Mapping], dict]) -> None:
    """Register a verifier/loader returning states and authority, or raising.

    The loader must authenticate its own schema/version and deterministic state
    construction before returning. Built-in schemas are reserved. Extension
    artifacts remain non-scoreable regardless of the loader's authority.
    """
    if (
        not isinstance(schema_id, str)
        or not schema_id
        or schema_id
        in {
            "shakebench.phase07.dev_states",
            "shakebench.phase08.committed_states",
            "shakebench.phase09.task_states",
            "shakebench.train_states",
        }
    ):
        raise ValueError("a new, nonempty state schema_id is required")
    if schema_id in _STATE_LOADERS or not callable(loader):
        raise ValueError("state loader must be callable and schema_id unregistered")
    _STATE_LOADERS[schema_id] = loader


def load_extension_states(payload: Mapping) -> dict | None:
    loader = _STATE_LOADERS.get(payload.get("schema_id"))
    if loader is None:
        return None
    result = loader(deepcopy(payload))
    states = result["states"]
    if not isinstance(states, list) or not states:
        raise ValueError("state loader must return a nonempty states list")
    identifiers = []
    for state in states:
        prepared = prepare_task_state(state)
        identifier = prepared.get("state_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("state must have a nonempty state_id")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate state_id in task asset")
    return {"states": deepcopy(states), "authority": {**result["authority"], "scoreable": False}}


def require_pick_place(state: Mapping, *, consumer: str) -> None:
    if task_type(state) != "pick_place":
        raise ValueError(f"{consumer} supports only pick_place; a task-specific adapter is required")


def split_overlap(train_states, eval_states) -> list[str]:
    """Compare task-owned execution identities, including task type."""

    def identity(state):
        if not isinstance(state, Mapping):
            return str(state)
        prepared = prepare_task_state(state)
        return json.dumps(
            [task_type(prepared), get_task_definition(prepared).fingerprint(prepared)],
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )

    train = {
        identity(state): str(state.get("state_id", "unnamed-state")) if isinstance(state, Mapping) else str(state)
        for state in train_states
    }
    evaluation = {identity(state) for state in eval_states}
    return sorted(train[key] for key in train.keys() & evaluation)


def assert_split_disjoint(train_states, eval_states) -> None:
    from shakebench.utils.train_states import TrainStateError

    overlap = split_overlap(train_states, eval_states)
    if overlap:
        raise TrainStateError(f"train/evaluation state overlap: {overlap[:5]}")


def _pick_place_environment(**kwargs):
    from shakebench.environments.vibration_pick_place import VibrationPickPlace

    return VibrationPickPlace(**kwargs)


def _pick_place_state(state):
    from shakebench.utils.state_schema import normalize_state

    return normalize_state(state)


def _pick_place_kwargs(state):
    from shakebench.utils.tasks import task_env_kwargs

    return task_env_kwargs(state) if "task" in state else {"object_start_xy": tuple(state["object_xy_m"])}


def _pick_place_description(state):
    from shakebench.utils.tasks import OBJECTS, TaskSpec

    if "task" not in state:
        return {
            "task_id": "pick_place.mug",
            "instruction": "Pick up the object from the phenolic table and place it in the target crate.",
        }
    spec = TaskSpec.from_mapping(state["task"])
    region = state.get("grasp_region", "body")
    object_name = spec.grasp_plan(region).get("instruction", OBJECTS[spec.object_id]["instruction"])
    return {
        "task_id": spec.variant_id,
        "instruction": f"Pick up the {object_name} from the phenolic table "
        "and place it in the target crate.",
    }


def _pick_place_fingerprint(state):
    from shakebench.utils.train_states import execution_state_fingerprint

    return execution_state_fingerprint(state)


_PICK_PLACE = TaskDefinition(
    _pick_place_environment, _pick_place_state, _pick_place_kwargs, _pick_place_description, _pick_place_fingerprint
)
