# Pod Setup and AIMET Development

## TL;DR

```bash
# 1. Launch a pod (syncs your repo to /scratch/aimet and opens a shell)
bash scripts/kube/dev.sh

# 2. On the pod: set up environment (installs Python, CUDA, dependencies)
#    Use --skip-aimet when you plan to build from source in step 3; omit it to install pre-built AIMET from pip.
bash scripts/environment/setup_genai.sh --skip-aimet

# 3. Build AIMET from source (will prompt for torch/onnx/both)
bash scripts/environment/build_aimet.sh --cuda-arch 80 --clean

# 4. Run a quick smoke test
pytest TrainingExtensions/torch/test/python/v2/quantsim/test_quantsim.py::TestQuantsim::test_invalid_bw_instantiation -v  # torch
pytest TrainingExtensions/onnx/test/python/test_quantsim.py::TestQuantSim::test_insert_quantize_op_nodes -v               # onnx
```

---

## Quick Start

### 1. Launch a pod

```bash
bash scripts/kube/dev.sh
```

This submits an Argo workflow, waits for the pod, syncs your local repo to `/scratch/aimet` on the pod, and opens a shell.

Options:
- `-c 8 -g 1 -m 32Gi` — request specific CPU/GPU/memory
- `-p <pod-name>` — reconnect to an existing pod
- `-e <command>` — run a command instead of opening a shell

### 2. Set up the environment

Once on the pod, run:

```bash
bash scripts/environment/setup_genai.sh
```

This installs system packages, creates a Python venv, installs PyTorch with CUDA, and sets up all dependencies.

### 3. Install AIMET

Choose one of the two approaches below.

---

## Option A: Install AIMET from pip (pre-built)

Use this when you don't need to modify AIMET source code.

```bash
bash scripts/environment/setup_genai.sh
```

This installs AIMET from PyPI (or pre-built wheels if available). No additional steps needed.

Run tests:
```bash
pytest TrainingExtensions/onnx/test/python/test_quantsim.py -v
```

---

## Option B: Build AIMET from source

Use this when you are developing or testing local AIMET changes.

```bash
# Step 1: Set up environment without installing AIMET
bash scripts/environment/setup_genai.sh --skip-aimet

# Step 2: Build from source (interactive — prompts for variant selection)
bash scripts/environment/build_aimet.sh --cuda-arch 80 --clean
```

When run without a variant flag, the script will prompt you to select a build variant:
```
Select build variant:
  1) torch only
  2) onnx only
  3) doc only (builds both torch + onnx; not recommended for development)
Enter choice [1/2/3]:
```

You can also specify the variant directly via flags:
```bash
bash scripts/environment/build_aimet.sh --torch-only --cuda-arch 80
bash scripts/environment/build_aimet.sh --onnx-only --cuda-arch 80
```

Common `--cuda-arch` values:
| GPU   | Arch |
|-------|------|
| V100  | 70   |
| T4    | 75   |
| A100  | 80   |
| A10G  | 86   |
| H100  | 90   |

Run tests:
```bash
pytest TrainingExtensions/onnx/test/python/test_quantsim.py -v
```

For full build options:
```bash
bash scripts/environment/build_aimet.sh --help
```

---

## Stopping a pod

```bash
# List your running pods
bash scripts/kube/stop_pod.sh -l

# Stop all your pods
bash scripts/kube/stop_pod.sh -a

# Stop a specific workflow
bash scripts/kube/stop_pod.sh <workflow-name>
```

---

## Troubleshooting

### Sync stopped working
If local file edits aren't appearing on the pod, the sync process may have died. Reconnect to the same pod to restart it:
```bash
bash scripts/kube/dev.sh -p <pod-name>
```

### Missing tools on local machine
`dev.sh` requires: `kubectl`, `argo`, `rsync`, `jq`, `inotifywait`. If any are missing, the script will auto-install them via `scripts/kube/install_deps.sh`.
