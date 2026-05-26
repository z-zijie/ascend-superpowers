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

## Packaging Template

Treat direct-launch packaging files as infrastructure, not operator logic. Do
not rewrite them from memory.

Copy `cann-bench/examples/direct_launch_example/setup.py` verbatim for generated
source-dir projects. It is intentionally operator-independent:

- package name is always `cann_bench`
- version is always `1.0.0`
- it builds the CMake extension through a custom `cmake_build` command
- it emits the required `cp38-abi3-linux_aarch64` wheel tag
- it uses setuptools APIs compatible with the server environment

Do not import `setuptools.command.clean.clean`, do not subclass
`setuptools.command.build_ext.build_ext` for the CMake command, and do not invent
an `options={'bdist_wheel': ...}` replacement for the `ABI3Wheel(bdist_wheel)`
subclass. Those rewrites are packaging bugs, not simplifications.

Evidence: Exp round 4 failed before CANN-Bench because generated `setup.py`
rewrote the reference and imported removed setuptools API
`setuptools.command.clean`; see
`LOGS/ascend-superpowers-exp/round-4/build.log` and
`LOGS/ascend-superpowers-exp/round-4/round.md`.

## Artifact Boundary

Treat generated source as immutable benchmark evidence.

1. Remove stale `generated/cannbench/<op>/` before a clean generation.
2. Generate the complete source-dir project.
3. Immediately create a source manifest before build or evaluation.
4. Run build, install, import, and CANN-Bench evaluation against that artifact.
5. If build, install, import, runtime, accuracy, or scoring fails, record the
   failure and stop. Do not fix files inside `generated/cannbench/<op>/`.

Allowed generated-tree changes after the manifest are build/runtime artifacts
only: `build/`, `dist/`, `*.egg-info`, `__pycache__/`, `.pytest_cache/`,
compiled shared objects such as `*.so` copied into the Python package, reports,
and profiler output. Source edits after the manifest invalidate the run. When
comparing manifests, normalize paths relative to the generated project root and
exclude those build/runtime artifact patterns in both snapshots.

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

PyTorch schema default literals follow Python-style parsing, not C++ literal
spelling. In `TORCH_LIBRARY_FRAGMENT` schema strings, write boolean defaults as
`False` or `True`, never `false` or `true`.

Good:

```cpp
m.def("op_name(Tensor x, bool maximize=False) -> Tensor");
```

Bad:

```cpp
m.def("op_name(Tensor x, bool maximize=false) -> Tensor");
```

Integer defaults such as `int step=1` and mixed defaults such as
`float eps=1e-8, int step=1, bool flag=False` are accepted by the PyTorch schema
parser on the CANN-Bench server. If schema default support is uncertain for a
future attr type, omit the default from the C++ schema and provide the default in
the Python wrapper that CANN-Bench calls.

Evidence: ApplyAdamW round 1 built and installed, then failed at import because
the generated schema used `bool maximize=false`; see
`LOGS/ascend-superpowers-apply_adam_w/round-1/round.md` and
`LOGS/ascend-superpowers-apply_adam_w/round-1/schema-default-probe.log`.

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

For formulas with float scalar attrs, compute derived scalar constants in the
host/plugin C++ wrapper from the original dispatcher parameters, then cast each
final derived constant to the kernel-side type before launch. Do not first cast
a schema `float` attr to `float` and then derive numerically sensitive constants
such as complements, powers, reciprocal denominators, or combined coefficients.
AscendC/NPU Scalar computation does not support `double`; never use `double`
inside `__global__`, `__aicore__`, or other NPU-side kernel code. The kernel
should receive already-derived `float` scalars.

Good:

```cpp
double inv_bias2 = 1.0 / (1.0 - std::pow(beta2, static_cast<double>(step)));
float one_minus_beta2_f = static_cast<float>(1.0 - beta2);
float inv_bias2_f = static_cast<float>(inv_bias2);
```

Better for optimizer/bias-correction formulas, precompute combined coefficients
in the host wrapper and pass the final `float` coefficients to the kernel:

```cpp
double denom2 = 1.0 - std::pow(beta2, static_cast<double>(step)); // host C++ only
float v_coeff = static_cast<float>(beta2 / denom2);
float grad2_coeff = static_cast<float>((1.0 - beta2) / denom2);
```

Bad:

```cpp
float beta2_f = static_cast<float>(beta2);
float one_minus_beta2 = 1.0f - beta2_f;
float inv_bias2 = 1.0f / (1.0f - std::pow(beta2_f, static_cast<float>(step)));
```

This matters for attrs near 1.0: `beta2=0.999` rounds to
`0.9990000128746033f`, so `1.0f - beta2_f` becomes
`0.0009999871` instead of the task/golden value rounded from `1.0 - 0.999`.
That small scalar error can create large MARE when outputs are near zero.

Evidence: ApplyAdamW round 2 passed build/import but failed case 4 accuracy
with score 69.47 because generated code derived `1 - beta2` and bias correction
after casting `beta2` to `float`. The diagnostic shows generated-style constants
produce MARE about `0.598`, while double-derived complements/bias constants
reduce MARE below the `0.05` float32 MARE threshold; see
`LOGS/ascend-superpowers-apply_adam_w/round-2/round.md` and
`LOGS/ascend-superpowers-apply_adam_w/round-2/scalar-derived-constants-diagnostic.log`.

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
