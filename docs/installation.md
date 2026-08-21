# Installation

ShakeBench requires Python 3.11+, Isaac Lab 3.0, the Isaac Newton integration, and a CUDA-capable setup supported by Isaac Lab. It is not designed to run its physical scene under system Python.

```bash
git clone <repository-url> ShakeBench
cd ShakeBench
export ISAACLAB_ROOT=/path/to/IsaacLab-3.0
"$ISAACLAB_ROOT/.venv/bin/python" -m pip install -e .
./run_tests.sh
```

`run.sh`, `run_python.sh`, and `run_tests.sh` source `scripts/isaac_env.sh`; the last command also disables pytest plug-in autoload because unrelated host plug-ins can break the Isaac virtual environment.

## Backends

The scoreable `official` profile is Newton/MJWarp at 1000 Hz with four C2_CLITE solver substeps and 50 iterations. The 240 Hz `training` profile trades contact fidelity for throughput and is not scoreable. NewtonGL supports albedo, roughness, and metallic material inputs. Normal maps, opacity, and emissive textures do not reach the shape shader; directional shadow mapping is deliberately disabled because its close-up moving-camera output is unstable.

## Common failures

- `ModuleNotFoundError: isaaclab`: set `ISAACLAB_ROOT` and use the repository runners.
- CUDA or USD initialization failures: verify the Isaac Lab installation with its own examples first.
- `SIGSEGV`, double-free, or allocator abort during Newton model construction: this is an observed upstream instability; retry and do not treat a missing JSON/MP4 as a valid run.
- An official run returning exit code 2 can be a valid failed episode (`success=false`), not a launcher failure.
