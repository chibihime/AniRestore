# Environment Setup

## Why Kaggle Notebooks

Local development machine is an Apple M4 Mac (arm64, no NVIDIA GPU), so CUDA/Triton
kernel work is not possible locally. **Kaggle Notebooks** was chosen as the free GPU
environment because it offers the most free GPU-hours/week (30 hrs, NVIDIA T4 x2 or
P100), a stable/pinned CUDA + driver toolchain, and lets us work close to a normal
file-based repo workflow rather than being locked into ephemeral notebook cells.

## One-time Kaggle setup

1. Create a free account at kaggle.com (email verification + phone verification is
   required to enable GPU access on notebooks).
2. Create a new Notebook: **Create > New Notebook**.
3. In the notebook's **Settings** panel (right sidebar):
   - **Accelerator**: select `GPU T4 x2`. **Do not select `GPU P100`** — Kaggle's
     preinstalled PyTorch build (2.10.0+cu128 as of writing) only ships CUDA
     kernels for compute capability sm_70 and up. The P100 is Pascal (sm_60), so
     CUDA calls fail with `no kernel image is available for execution on the
     device` even though `torch.cuda.is_available()` returns `True` (that check
     only confirms driver visibility, not kernel compatibility). T4 is Turing
     (sm_75) and works fine with the stock install.
   - **Internet**: turn **ON** (required to `pip install`, clone the repo, and hit
     the MAL API).
   - **Persistence**: "Files only" is fine for iterative work; the notebook session
     itself still resets after ~12 hours or on idle timeout, so always re-clone /
     re-run setup cells at the start of a session rather than relying on state
     surviving between sessions.
4. In the first cell, clone the repo and install deps:

   Since the repo is **private**, add a GitHub Personal Access Token as a Kaggle
   **Secret** named `AniRestore_GH_PAT` (Add-ons > Secrets), then pull it in and
   clone with it:

   ```python
   from kaggle_secrets import UserSecretsClient
   user_secrets = UserSecretsClient()
   gh_pat = user_secrets.get_secret("AniRestore_GH_PAT")

   !git clone https://{gh_pat}@github.com/chibihime/AniRestore.git
   %cd AniRestore
   !pip install -r requirements.txt
   ```

   Note: do **not** `pip install torch` — Kaggle ships a working torch+CUDA build
   already matched to its driver; reinstalling can break that pairing (see the
   comment at the top of `requirements.txt`).

5. Verify the environment before doing anything else:

   ```bash
   !python src/check_env.py
   ```

   This must print `All checks passed` (GPU visible, CUDA available, a trivial
   Triton kernel compiles and runs correctly) before proceeding to model or kernel
   work.

## Weekly quota notes

- Kaggle GPU quota is ~30 hours/week, shared across P100/T4 sessions, and resets on
  a rolling weekly basis.
- Sessions auto-stop after ~20 minutes of idle time (with the browser tab closed)
  or ~12 hours total — save/download any results (benchmark numbers, output images)
  before ending a session, since ephemeral notebook disk is not guaranteed to
  persist across sessions the way "Kaggle Datasets" output does.
- For anything that needs to survive across sessions (checkpoints, dataset), push
  small artifacts to the GitHub repo and larger ones (raw images, model weights) to
  a Kaggle Dataset attached to the notebook, rather than relying on local notebook
  disk.
