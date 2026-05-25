---
name: cann-bench-operator-generation
description: Use when generating or debugging Ascend CANN-Bench source-dir operator projects
---

# CANN-Bench Operator Generation

## Overview

Generate Ascend CANN-Bench operator projects as benchmark artifacts, not as
hand-maintained application code. The task files define the operator contract;
the generated source-dir project is only valid if it builds, installs, imports,
runs all CANN-Bench cases, passes accuracy, and scores without post-generation
source edits.

Use this skill for any operator request under `cann-bench/tasks/` that should
produce a CANN-Bench `--source-dir` project under `generated/cannbench/<op>/`.

## Source Of Truth

Read the matching task directory before writing code:

- `proto.yaml` for schema, attrs, dtypes, shape support, and defaults
- `cases.yaml` / `cases.csv` for required shapes, dtypes, ranges, thresholds,
  and performance baselines
- `desc.md` for semantic constraints
- `golden.py` for numerical behavior, including compute precision and casts

Do not encode operator facts that are not present in the request or task files.

## Required Layout

CANN-Bench source-dir evaluation expects a package named `cann_bench`:

```text
generated/cannbench/<op>/
├── build.sh
├── CMakeLists.txt
├── setup.py
├── cann_bench/
│   └── __init__.py
├── cmake/
├── csrc/
│   ├── extension.cpp
│   └── ops/<op>/
│       ├── CMakeLists.txt
│       ├── op_kernel/
│       └── op_plugin/
└── scripts/
    └── build_wheel.sh
```

Use `cann-bench/examples/direct_launch_example/` as the structural reference for
direct-launch CMake, package, and torch registration behavior.

The generated project must expose either `cann_bench.<op_snake>(...)` or
`torch.ops.cann_bench.<op_snake>(...)`, preferably both through a Python wrapper
that delegates to the torch op.

## Artifact Boundary

Treat generated source as immutable benchmark evidence.

1. Remove stale `generated/cannbench/<op>/` before a clean generation.
2. Generate the complete source-dir project.
3. Immediately create a source manifest before build or evaluation.
4. Run build, install, import, and CANN-Bench evaluation against that artifact.
5. If build, install, import, runtime, accuracy, or scoring fails, record the
   failure and stop. Do not fix files inside `generated/cannbench/<op>/`.

Allowed generated-tree changes after the manifest are build/runtime artifacts
only: `build/`, `dist/`, `*.egg-info`, `__pycache__/`, `.pytest_cache/`, reports,
and profiler output. Source edits after the manifest invalidate the run.

## Torch Registration Rules

Register custom ops in the `cann_bench` namespace:

```cpp
TORCH_LIBRARY_FRAGMENT(cann_bench, m) {
    m.def("op_name(Tensor x, float attr) -> Tensor");
}
```

For NPU implementations, register at least `Meta` and `PrivateUse1`; also
register `AutogradPrivateUse1` when autograd dispatch would otherwise bypass the
NPU implementation:

```cpp
TORCH_LIBRARY_IMPL(cann_bench, Meta, m) {
    m.impl("op_name", op_meta);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m) {
    m.impl("op_name", op_npu);
}

TORCH_LIBRARY_IMPL(cann_bench, AutogradPrivateUse1, m) {
    m.impl("op_name", op_npu);
}
```

PyTorch schema `float` attrs are Python floating-point values. In C++ dispatcher
function signatures used with `TORCH_LIBRARY_IMPL`, use `double` parameters for
those attrs, then cast deliberately to kernel-side `float` only after all host
precomputation is complete.

Good:

```cpp
torch::Tensor op_npu(const torch::Tensor& x, double scale, double shift) {
    float scale_f = static_cast<float>(scale);
    float shift_f = static_cast<float>(shift);
    // launch kernel with scale_f and shift_f
}
```

Bad:

```cpp
torch::Tensor op_npu(const torch::Tensor& x, float scale, float shift);
```

Evidence: Exp round 3 exposed this as a reusable build rule. The schema string
used `float`, but the C++ dispatcher functions needed `double`; see
`LOGS/ascend-superpowers-exp/round-3/round.md` and
`generated/cannbench/exp/csrc/ops/exp/op_plugin/exp_plugin.cpp`.

## AscendC API Policy

Do not guess AscendC API signatures, headers, namespaces, type support, tiling
contracts, or launch constraints. Search `asc-devkit/` first and record the
evidence path in the run log.

Examples of acceptable evidence:

- `asc-devkit/include/...` header defining the API
- `asc-devkit/impl/...` implementation or checker showing type constraints
- `asc-devkit/examples/...` example demonstrating correct usage
- `asc-devkit/tests/...` test showing supported types or expected failures

If no evidence exists, treat the API assumption as unverified and avoid baking
it into the generated project or this skill.

## Dtype And Precision

Follow `golden.py`, not intuition.

- If FP16/BF16 golden behavior computes in FP32 and casts back, implement that
  behavior explicitly.
- If an AscendC kernel only supports a subset of required dtypes, handle missing
  dtypes through a general, documented conversion path only when it matches the
  golden semantics.
- Preserve output shape and dtype exactly unless the task says otherwise.

## Verification Ladder

Run from `cann-bench/` with the CANN environment loaded:

```bash
PYTHONPATH=src python -m kernel_eval.cli list
PYTHONPATH=src python -m kernel_eval.cli info --operator <OpName>
PYTHONPATH=src python -m kernel_eval.cli eval --source-dir ../generated/cannbench/<op> --operator <OpName> --device-id 0
```

Record exact commands, exit codes, logs, report paths, and the source manifest
comparison. A run is successful only when compile, install, import, all cases,
all accuracy checks, and the required score are proven by current evidence.
