---
name: cann-bench-operator-generation
description: Use when generating or debugging Ascend CANN-Bench source-dir operator projects
---

# CANN-Bench Operator Generation

## Overview

Generate Ascend CANN-Bench operator projects as benchmark artifacts, not as
hand-maintained application code. The task files define the operator contract;
the delivered source-dir project is only valid if the final candidate builds,
installs, imports, runs all CANN-Bench cases, passes accuracy, and scores with
source-manifest evidence for that candidate.

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

## Role Boundary And Candidate Freezing

Codex orchestrates the loop and owns Harness changes. Do not hand-edit generated
operator source from Codex. The server-side generator/debugger may iterate inside
`generated/cannbench/<op>/` before delivery to fix build, runtime, accuracy, or
score failures.

Use source manifests as candidate evidence, not as a ban on generator-side
debugging:

1. Remove stale `generated/cannbench/<op>/` before a clean generation.
2. Generate the complete source-dir project.
3. Build, install, import, and evaluate. If the generator fixes generated source,
   treat that as a new candidate and record the fix in the round log.
4. Before reporting a candidate as final or comparing a score, create a source
   manifest for the exact source tree being evaluated.
5. Do not edit source while reusing that candidate's build/eval result as
   evidence. If source changes, create a new manifest and rerun the relevant
   checks.

Allowed generated-tree changes that do not require a new source candidate are
build/runtime artifacts only: `build/`, `dist/`, `*.egg-info`, `__pycache__/`,
`.pytest_cache/`, compiled shared objects such as `*.so` copied into the Python
package, reports, and profiler output. When comparing manifests, normalize paths
relative to the generated project root and exclude those build/runtime artifact
patterns in both snapshots.

If a failure reveals a reusable Harness gap, log the exact diagnostics, the
generator-side workaround if one was attempted, evidence paths, and the proposed
general rule. Mac Codex then updates `ascend-superpowers/` and starts a fresh
or continued generation round with the improved Harness.

## Pre-Delivery Gate

Never deliver a generated operator as `complete` from a failed CANN-Bench run.
A failed run may be useful debugging evidence, but it is not a deliverable.

Before writing a `complete` conclusion, run the report gate against the latest
CANN-Bench JSON report for the exact source candidate being delivered:

```bash
python ascend-superpowers/skills/cann-bench-operator-generation/scripts/cannbench_report_gate.py \
  cann-bench/reports/<latest_eval>.json \
  --min-score <required_score>
```

The gate must pass before delivery. A non-zero exit means at least one required
condition is false: not all cases ran, not all cases succeeded, not all accuracy
checks passed, or the score is below the requested target. In that state:

- do not write `complete`
- do not describe the operator as delivered or done
- continue debugging the generated project when reasonable
- otherwise write `continue` or `blocked` with failed case ids, error messages,
  score, report path, and the next reusable Harness rule needed

If the generated source changes after a report is produced, the report is stale:
create a new source manifest, rebuild or reinstall as needed, rerun CANN-Bench,
and rerun the report gate. The final `round.md` must include the gate command,
exit code, JSON report path, all failed cases if any, and the conclusion.

Evidence: Softmax round 3 produced a CANN-Bench report with 18/20 cases passing
and score 57.09. That report was useful for debugging, but it failed the
delivery gate because accuracy did not pass for all cases and the score was
below 75.

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

When a torch plugin passes host-computed launch parameters into an
`OpCommand::RunOpApi` or other local lambda, do not capture C++ structured
binding variables from the enclosing function. Some server toolchains reject
references to structured-binding names from lambdas. Unpack tuples into
ordinary named variables before defining the lambda, or copy the values into
explicit `const` launch variables.

Good:

```cpp
auto prepared = prepare_2d(x, dim);
torch::Tensor x_2d = std::get<0>(prepared);
const int64_t outer_for_launch = std::get<1>(prepared);
const int64_t reduce_for_launch = std::get<2>(prepared);

auto acl_call = [&]() -> int {
    launch_kernel(ptr, out, outer_for_launch, reduce_for_launch, stream);
    return 0;
};
```

Bad:

```cpp
auto [x_2d, outer_size, reduce_size] = prepare_2d(x, dim);
auto acl_call = [&]() -> int {
    launch_kernel(ptr, out, outer_size, reduce_size, stream);
    return 0;
};
```

Evidence: Softmax round 1 build failed because the generated plugin referenced
structured-binding names `outer_size` and `reduce_size` inside the launch
lambda; see `LOGS/ascend-superpowers-softmax/round-1/claude-run.filtered.log`.

## AscendC API Policy

Do not guess AscendC API signatures, headers, namespaces, type support, tiling
contracts, preprocessor availability, or launch constraints. Search both
`asc-devkit/` and the installed CANN headers, then record the evidence paths in
the run log. A declaration that exists behind an incompatible `__NPU_ARCH__`
guard is not available for the target build.

Examples of acceptable evidence:

- `asc-devkit/include/...` header defining the API
- `asc-devkit/impl/...` implementation or checker showing type constraints
- `asc-devkit/examples/...` example demonstrating correct usage
- `asc-devkit/tests/...` test showing supported types or expected failures
- `/home/developer/Ascend/cann-9.0.0/compiler/ascendc/include/...` installed
  header defining the API and its `#if (__NPU_ARCH__ == ...)` guards

If no evidence exists, treat the API assumption as unverified and avoid baking
it into the generated project or this skill.

For the CANN-Bench server target `NPU_ARCH=ascend910b` / `--npu-arch=dav-2201`,
check scalar binary vector ops especially carefully. In installed CANN 9.0.0,
`AscendC::Subs` Level 2 is guarded to NPU_ARCH 3510, 5102, 3003, and 3113, so it
is unavailable on dav-2201. Use `AscendC::Adds(dst, src, -scalar, count)` for
scalar subtraction on dav-2201. `AscendC::Adds` and `AscendC::Muls` Level 2 have
been verified as available for dav-2201; do not assume `Subs` or `Divs` are
available without checking the installed guard.

Evidence: Softmax round 2 build failed because the generated kernel used
`AscendC::Subs` for scalar max subtraction. The installed header
`/home/developer/Ascend/cann-9.0.0/compiler/ascendc/include/basic_api/interface/kernel_operator_vec_binary_scalar_intf.h`
contains the `Subs` guard excluding dav-2201; see
`LOGS/ascend-superpowers-softmax/round-2/round.md`.

Inside AscendC `__global__` / `__aicore__` kernels, do not use C++ lambdas to
wrap tile processing, `AllocTensor`, `DataCopyPad`, vector math, or queue
operations. Bisheng can classify the lambda call path as host-side and emit
errors such as "call to [aicore] function from [host] function." Use explicit
inline code, class member methods marked `__aicore__ inline`, or ordinary
duplicated main-loop/tail blocks instead.

Evidence: ApplyAdamW round 2 initially generated a lambda inside the AscendC
kernel and build failed with bisheng host/aicore call errors before the source
was frozen; see `LOGS/ascend-superpowers-apply_adam_w/round-2/round.md`.

For GM-to-UB and UB-to-GM transfers in direct-launch AscendC kernels, use
`AscendC::GlobalTensor<T>` wrappers or an already established local project
pattern. Do not pass raw `(__gm__ T*)` pointers directly to `DataCopy` or
`DataCopyPad` overloads from generated kernels unless the exact overload has
been verified in `asc-devkit/`.

Preferred pattern:

```cpp
AscendC::GlobalTensor<T> xGm, yGm;
xGm.SetGlobalBuffer((__gm__ T*)x + blockOffset);
yGm.SetGlobalBuffer((__gm__ T*)y + blockOffset);

AscendC::DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
```

Avoid:

```cpp
AscendC::DataCopy(xLocal, (__gm__ T*)x + offset, count);
AscendC::DataCopyPad(xLocal, (__gm__ T*)x + offset, copyParams, padParams);
```

Evidence: Softmax round 1 failed during kernel build because generated code
called `DataCopy` with raw GM pointers and only three arguments; CANN headers
expose `DataCopy` overloads for `LocalTensor` plus `GlobalTensor` or additional
parameters. See
`LOGS/ascend-superpowers-softmax/round-1/claude-run.filtered.log`,
`asc-devkit/include/basic_api/kernel_operator_data_copy_intf.h`, and the
working generated examples under
`generated/cannbench/foreach_addcdiv_scalar/csrc/ops/foreach_addcdiv_scalar/op_kernel/`.

## Dtype And Precision

Follow `golden.py`, not intuition.

- If FP16/BF16 golden behavior computes in FP32 and casts back, implement that
  behavior explicitly.
- If an AscendC kernel only supports a subset of required dtypes, handle missing
  dtypes through a general, documented conversion path only when it matches the
  golden semantics.
- Preserve output shape and dtype exactly unless the task says otherwise.

For performance-critical direct-launch operators, do not satisfy FP16/BF16
FP32-compute semantics by inserting full-tensor conversions in the torch plugin,
such as `x.to(torch::kFloat32)` for every input and `y.to(input_dtype)` after
the kernel. That turns a fused operator into several separate CANN operators and
can pass accuracy while missing the CANN-Bench score target.

Prefer dtype-specific AscendC kernels or template launches for `float`, `half`,
and `bfloat16_t`. Load tensors in their original dtype, cast each tile-local
input to `float` buffers inside the kernel, compute in FP32, then cast the
tile-local output back to the original dtype before copying to GM. For BF16
outputs, use the rounding mode required by evidence and golden behavior; prior
CANN-Bench evidence used `AscendC::RoundMode::CAST_RINT` for BF16 cast-back and
`CAST_NONE` for FP16.

Keep host-side scalar precomputation in the plugin when it needs `double`, but
pass only final `float` constants to kernels. Do not move `double` scalar math
into AscendC kernel code: NPU Scalar computation does not support `double`.

Evidence: ApplyAdamW round 3 passed all 20 accuracy cases but scored only 73.72
because FP16/BF16 paths performed host/plugin full-tensor casts before and after
the fused kernel; see
`LOGS/ascend-superpowers-apply_adam_w/round-3/round.md` and
`generated/cannbench/apply_adam_w/csrc/ops/apply_adam_w/op_plugin/apply_adam_w_plugin.cpp`.
ForeachAddcdivScalar round 1 reached the score target using dtype-specific
launches and tile-local FP32 casts inside the AscendC kernel; see
`LOGS/ascend-superpowers-foreach_addcdiv_scalar/round-1/round.md` and
`generated/cannbench/foreach_addcdiv_scalar/csrc/ops/foreach_addcdiv_scalar/op_kernel/foreach_addcdiv_scalar_kernel.cpp`.

For multi-input elementwise fusion kernels, treat tile size and buffering as
part of correctness for benchmark viability, not only as an optimization detail.
Use double-buffered queues (`PIPELINE_DEPTH = 2`) when UB capacity permits, and
avoid inflating queue counts or scratch queues so much that each tile becomes
too small for memory throughput. A fixed tile size around a few thousand
elements is a reasonable starting point for vector elementwise kernels; adjust
downward only when the actual number of input, output, and FP32 temporary buffers
does not fit in UB. If a generated kernel needs many FP32 temporaries, prefer
reusing `TBuf` calculation buffers in a clear sequence over adding enough queue
buffers to severely reduce tile length.

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

After eval, identify the latest JSON report and run the pre-delivery gate. For
CANN-Bench goals that specify score >= 75:

```bash
python ../ascend-superpowers/skills/cann-bench-operator-generation/scripts/cannbench_report_gate.py \
  reports/<latest_eval>.json \
  --min-score 75
```

Record exact commands, exit codes, logs, report paths, gate output, and the
source manifest comparison. A run is successful only when compile, install,
import, all cases, all accuracy checks, and the required score are proven by
current evidence and the report gate exits 0.
