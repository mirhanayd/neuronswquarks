# LO Inclusive DIS and Legacy Cornell Visualization

This repository contains tested electron-proton DIS kinematics, a leading-order
electromagnetic inclusive cross section using real LHAPDF grids, and the original
Rust Cornell-potential neural-network desktop demonstration. The DIS and legacy
visualization paths are deliberately separated.

> **Scientific scope:** the cross section is fixed-α, LO photon exchange—not an
> NLO/NNLO calculation or event generator. The Cornell application remains an
> educational visualization.

The active Rust crate is [`quark_sim`](quark_sim/). Run Cargo commands from that directory.
The kinematics specification and independent fixture are in
[`quark_sim/docs/dis_kinematics.md`](quark_sim/docs/dis_kinematics.md).
Native setup and LO formulas are documented in
[`quark_sim/docs/lhapdf_integration.md`](quark_sim/docs/lhapdf_integration.md) and
[`quark_sim/docs/lo_dis_cross_section.md`](quark_sim/docs/lo_dis_cross_section.md).

## Environment and build

The project is tested in WSL Ubuntu. From the repository root:

```bash
cd quark_sim
rustc --version
cargo --version
```

Install and activate the pinned user-local LHAPDF 6.5.6 plus CT18LO member 0:

```bash
bash scripts/setup_lhapdf_wsl.sh
source scripts/lhapdf_env.sh
```

CPU is the default and does not enable any Candle CUDA features:

```bash
cargo run --release
```

CUDA support is opt-in at compile time:

```bash
cargo run --release --features cuda
```

The CUDA build requires a working CUDA toolkit, driver, and supported GPU. When compiled with `cuda`, the program uses CUDA device 0 if Candle detects it and otherwise reports a fallback to CPU. A successful CPU build does not prove that CUDA works.

## CLI

```text
quark_sim
quark_sim --load <session.json>
quark_sim --load-model <model.safetensors>
quark_sim dis-kinematics [OPTIONS]
quark_sim dis-cross-section [OPTIONS]
quark_sim -h
quark_sim --help
```

- No arguments: train a new model, create outputs, run the path visualization, save a session, and open the GUI.
- `--load`: load saved arrays and paths for static viewing. It does not load a model or resume a simulation.
- `--load-model`: load weights and the required sibling `<model>_config.json`, generate fresh plots and paths, save a new session, and open the interactive GUI.
- `dis-kinematics`: calculate massive-beam `s`, `Q²`, `x`, `y`, and `W²`
  without initializing Candle or the GUI.
- `dis-cross-section`: query a selected installed PDF set/member and print
  `x f_i`, LO `F₂`, `y`, and `d²σ/(dx dQ²)` in GeV⁻⁴ and pb/GeV².
- `--help` / `-h`: print help without training or launching the GUI.
- Unknown arguments, extra arguments, and missing file operands are rejected instead of starting training.

Examples:

```bash
cargo run --release -- --help
cargo run --release -- dis-kinematics --help
cargo run --release -- dis-kinematics --electron-energy 27.5 --proton-energy 920.0 --scattered-electron-energy 15.0 --theta-deg 20.0
cargo run --release -- dis-cross-section --x 0.01 --q2 100.0 --electron-energy 27.5 --proton-energy 920.0 --pdf-set CT18LO --pdf-member 0
cargo run --release -- --load outputs/20251203_123953_GMT/session.json
cargo run --release -- --load-model outputs/20251203_123953_GMT/trained_model.safetensors
```

## Current defaults

These values come from the active source files, not from historical output folders:

| Setting | Current value |
| --- | ---: |
| Training samples | 15,000 |
| Training radius | 0.05 to 3.0 fm |
| Epochs | 5,000 (full-batch) |
| Optimizer | AdamW |
| Learning rate | 0.01 |
| Weight decay | 0.01 |
| Network | `3 -> 256 -> 128 -> 64 -> 1` |
| Activations | ReLU after the first three linear layers |
| Static electron count | 30 |
| Impact-parameter range | -2.5 to +2.5 |
| Initial velocity | 0.5 |
| Integration time step | 0.05 |
| Maximum steps | 400 |
| Force scale | 0.2 |

Training is random and currently has no configurable seed, so two runs are not exactly reproducible.

## Outputs

A normal run creates `outputs/YYYYMMDD_HHMMSS_GMT/` with:

```text
trained_model.safetensors
trained_model_config.json
training_loss.svg
cornell_potential.svg
scattering.svg
session.json
```

`trained_model_config.json` stores the target mean and standard deviation required to denormalize predictions. A legacy weight file without this companion file cannot be loaded through `--load-model`.

A model-loaded run creates an `outputs/YYYYMMDD_HHMMSS_LOADED/` directory containing:

```text
training_loss.svg       # labelled placeholder: no training occurred in this run
cornell_potential.svg
scattering.svg
session.json
```

Existing output directories are accepted and files may be replaced. Because directory names have one-second resolution, two runs started in the same second can target the same directory. Historical outputs are retained by the repository and are not rewritten during normal validation.

## GUI behavior

The GUI contains:

- a central proton/electron plot;
- a training-loss plot or a clear “no training data” message;
- theoretical and neural Cornell-potential curves with optional test points.

In live mode, clicking empty plot space creates an electron and clicking a target quark flips its displayed spin. In session-view mode, saved electron trajectories are drawn as static polylines.

## Session format and limitations

`session.json` saves:

- training-loss history;
- theoretical and neural potential points;
- test distances, Cornell values, and neural predictions;
- output-path strings for the generated SVG files;
- optional electron trajectories and impact parameters.

Loading restores those serialized arrays. Missing optional `scattering_file` or `electrons` fields default to `None`, which preserves compatibility with older sessions.

Session loading does **not**:

- animate or replay electron trajectories;
- restore the neural model, optimizer, or model normalization values;
- restore CPU/CUDA device state;
- restore RNG state or a seed;
- restore training/scattering parameters, target-spin edits, or electrons created in the GUI;
- resume training or simulation;
- read and display the saved SVG files (the GUI redraws from JSON arrays).

Sessions are written before the interactive GUI opens, so later GUI interactions are not persisted. The file format has no explicit schema version.

## Project structure

```text
quark_sim/
├── Cargo.toml
├── Cargo.lock
├── compare_runs.py
├── docs/
│   ├── dis_kinematics.md
│   ├── lhapdf_integration.md
│   └── lo_dis_cross_section.md
├── scripts/
│   ├── setup_lhapdf_wsl.sh  # pinned, user-local LHAPDF + CT18LO setup
│   └── lhapdf_env.sh        # WSL environment activation
├── tests/
│   ├── dis_cli.rs
│   ├── dis_kinematics.rs
│   └── lhapdf_integration.rs # ignored native-backend regressions
├── outputs/                 # historical and newly generated runs
└── src/
    ├── lib.rs               # reusable library surface
    ├── main.rs              # CLI and orchestration
    ├── model.rs             # 4-linear-layer Candle model
    ├── training.rs          # data, training, save/load, regression test
    ├── physics/
    │   ├── constants.rs
    │   ├── four_vector.rs
    │   ├── dis_kinematics.rs
    │   ├── pdf.rs
    │   ├── structure_functions.rs
    │   ├── cross_section.rs
    │   └── legacy_cornell.rs
    ├── scattering.rs        # educational path integration and SVG
    ├── plotting.rs          # loss and potential SVG generation
    └── gui.rs               # live/static desktop visualization and sessions
```

`src/main.rs.backup`, the `Yeni klasör` backup directory/archive, and historical
output files are not active Rust sources. `compare_runs.py` is a historical,
hard-coded comparison helper rather than a general CLI tool.

## Locked dependency baseline

The repaired manifest follows the existing lockfile baseline:

- Candle Core / Candle NN 0.8.4
- eframe / egui / egui_plot 0.26.2
- Plotters 0.3.7
- rand 0.8.x
- Serde 1.0.x
- managed-lhapdf 0.4.2, using its `cxx` bridge to native LHAPDF 6.5.6
- CT18LO data version 1, member 0, for the pinned external regression

LHAPDF is a native, unconditional dependency, so activate
`scripts/lhapdf_env.sh` before Cargo builds. Candle CUDA is forwarded only
through the crate's optional `cuda` feature.

## Scientific limitations

The analytic function used by the demo is:

```text
V(r) = -4 alpha_s / (3 r) + k r
alpha_s = 0.5
k = 0.9
```

Important limitations include:

- The network learns values generated by this analytic function; it is not trained on experimental or lattice-QCD data.
- The Cornell potential describes a static quark-antiquark model. Applying it to electron trajectories is not a physical DIS calculation.
- The separate DIS path computes relativistic kinematics, reads real LHAPDF
  grids, evaluates LO electromagnetic `F₂`, and reports an inclusive
  differential cross section. It is not connected to the Cornell trajectory
  visualization.
- The DIS result is fixed-α, leading-order single-photon exchange. It has no
  nonzero `F_L`, `xF₃`, weak exchange, higher-order coefficient functions, PDF
  uncertainty propagation, scale variation, or radiative corrections.
- There is no phase-space integration, hadronization, detector response, or
  generated inelastic final state.
- Target quarks are fixed 2D points. Their displayed spin force is ad hoc and is not derived from QCD or QED.
- The motion is a simple finite-difference, unitless-looking integration. It has no electron mass or charge model.
- `HBARC` is declared but not used in the current Cornell expression, so the source's `fm` and `GeV` labels should not be interpreted as a dimensionally complete calculation.
- Rotational symmetry is sampled/learned rather than enforced by the network architecture.
- Static SVG axes can clip trajectories that continue beyond the plotted range.

The appropriate next physics phase is an independently benchmarked higher-order
structure-function backend (for example APFEL++) with scale and PDF uncertainty
handling, still kept separate from the legacy Cornell model.

## Validation

```bash
source scripts/lhapdf_env.sh
cargo fmt --all -- --check
cargo check
cargo clippy --all-targets -- -D warnings
cargo test
cargo test --test lhapdf_integration -- --ignored
cargo run --release -- --help
cargo run --release -- dis-kinematics --help
cargo run --release -- dis-cross-section --help
```

The test suite includes a CPU save/load regression: it initializes and saves a model, loads the parameters into a separately instantiated architecture, and requires matching predictions within `1e-6`.
