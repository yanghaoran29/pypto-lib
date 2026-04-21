# Dependencies

PyPTO-Lib is a **primitive tensor function library** built on top of the pypto
programming framework. It does not ship its own compiler, assembler, or
runtime — those come from three external repositories that must be installed
and importable before any example in this repository can build or run.

## External repositories

| Name        | Repository                          | Role                                                        |
|-------------|-------------------------------------|-------------------------------------------------------------|
| **pypto**   | `hw-native-sys/pypto`               | Compiler framework: front-end IR, passes, codegen           |
| **ptoas**   | `hw-native-sys/PTOAS`               | PTO assembler & optimizer toolchain (lowers IR to PTO-ISA)  |
| **simpler** | `hw-native-sys/simpler`             | PTO2 runtime (`pto-rt2`) that schedules and executes kernels |

All three live under the internal `hw-native-sys/` GitHub organization and
are required in combination — there is no one-package shortcut.

## What each dependency provides

### pypto (compiler framework)
- The `pypto.language` Python module (`import pypto.language as pl`)
  used by every example to declare `@pl.program` classes, `@pl.function`
  kernels, and tensor/tile operations (`pl.load`, `pl.matmul`, `pl.parallel`,
  `pl.at`, `pl.auto_incore`, etc.).
- IR representation, front-end passes, and the lowering pipeline that
  emits AIC/AIV kernel source files and an orchestration C++ entry point
  into `build_output/`.

### ptoas (assembler / optimizer)
- Consumes the kernel C++ files produced by pypto and assembles them
  into PTO-ISA object files (`.o`) for AIC and AIV cores.
- Provides instruction scheduling / register allocation and the
  cross-core fusion passes that produce `MixedKernels` groups
  (`incore_N_aic.cpp` + `incore_N_aiv.cpp` pairs).

### simpler (runtime, aka pto-rt2)
- Implements the on-device scheduler/orchestrator that drives the
  generated `aicpu_orchestration_entry(...)` function at runtime.
- Owns the swimlane trace (`merged_swimlane_*.json`) emitted under
  `build_output/.../swimlane_data/` when `runtime_profiling=True`.
- Documented at a high level in `docs/pto2_rt.md`.

## Installation sketch

The typical layout is to clone all four repositories side-by-side and
install pypto/ptoas/simpler into the active Python environment before
running any pypto-lib example:

```bash
# one parent directory
mkdir -p ~/src && cd ~/src

git clone https://github.com/hw-native-sys/pypto.git
git clone https://github.com/hw-native-sys/PTOAS.git ptoas
git clone https://github.com/hw-native-sys/simpler.git
git clone https://github.com/yanghaoran29/pypto-lib.git

# install the three dependencies (each repo has its own build/install instructions)
cd pypto   && pip install -e . && cd ..
cd ptoas   && pip install -e . && cd ..
cd simpler && pip install -e . && cd ..

# now pypto-lib examples can compile and run
cd pypto-lib
python examples/beginner/hello_world.py
```

The `.claude/skills/setup_env/` directory in this repo contains a
project-specific helper that automates the same sequence once the three
upstream repositories are accessible.

## Version compatibility

pypto-lib tracks the `main` branch of all three dependencies. There is
currently no pinned version matrix; breaking IR changes in pypto, ISA
changes in ptoas, or runtime ABI changes in simpler may require
rebuilding everything together.

When in doubt: check out the latest commit of each dependency that
matches the commit date of the pypto-lib commit you are building.

## Runtime-only dependencies

- Python ≥ 3.10
- `torch` (used by the `golden/` reference implementations and
  `build_tensor_specs()` helpers in examples)
- A device that matches the `--platform` flag on each example
  (`a2a3`, `a2a3sim`, `a5`, `a5sim`)

These are *in addition* to pypto / ptoas / simpler, not a replacement
for them.
