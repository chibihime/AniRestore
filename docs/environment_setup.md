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

   ```python
   import os
   import shutil

   os.chdir("/kaggle/working")
   shutil.rmtree("/kaggle/working/AniRestore", ignore_errors=True)
   !git clone https://github.com/chibihime/AniRestore.git
   os.chdir("/kaggle/working/AniRestore")
   !pip install -r requirements.txt
   print(os.getcwd())
   !ls src
   ```

   Use `os.chdir` with an **absolute path**, never the `%cd AniRestore` magic with
   a relative path, and always `shutil.rmtree` + fresh-clone rather than
   `git pull` on a directory that might already exist. Two failure modes this
   avoids:
   - A relative `%cd` compounds on whatever the cwd already was if the cell
     reruns, silently cloning into a nested `AniRestore/AniRestore/`.
   - Reusing an existing checkout with `git pull` can report "Already up to
     date" without actually updating the working tree if that checkout was
     left in a stale/partial state by an earlier interrupted cell run in the
     same session — this repo is tiny, so there's no real cost to always
     wiping and re-cloning instead of trying to make `git pull` work reliably
     against directory state you can't fully see.

   Always confirm with `!ls src` before running anything else — it should list
   every file/folder actually in the repo (`benchmark.py`, `data_loader.py`,
   `mal/`, `model/`, `profile_baseline.py`, `run_baseline_inference.py`,
   `check_env.py`), not just a subset.

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

## Attaching the cover-art dataset

The cover art itself isn't redistributed with this repo (it's copyrighted), so
the `pruthaannadate/anirestore-raw-covers` Kaggle dataset is private. To
reproduce, build `data/raw_covers/` yourself with the scripts described in
`data/README.md`, then upload it as your own private Kaggle dataset and
substitute your username/slug in the paths below.

A Kaggle dataset mounts at a path with an extra
`datasets/<owner>/` segment on this account's Kaggle environment, **not** the
flat `/kaggle/input/<slug>/` path older Kaggle docs describe:

```
/kaggle/input/datasets/pruthaannadate/anirestore-raw-covers
```

Always confirm with `!ls /kaggle/input/` (and one level deeper) after attaching
a dataset rather than assuming the flat path — it silently doesn't exist there
on this environment, and scripts reading `data/raw_covers` will fail with
`FileNotFoundError` if the symlink points at the wrong place:

```bash
!rm -f data/raw_covers
!ln -s /kaggle/input/datasets/pruthaannadate/anirestore-raw-covers data/raw_covers
!ls data/raw_covers | wc -l   # should print 2000
```

Also note: uploading a dataset via `kaggle datasets create -p <dir>` uploads
every file in `<dir>` as a **separate** request — for 2000 small images this
was unreliably slow and eventually stalled outright. Zipping the folder into a
single archive first (with a `dataset-metadata.json` alongside it) and
uploading that instead is far more reliable — Kaggle auto-extracts the zip
into individual files server-side once the dataset finishes processing
(confirm with `kaggle datasets status <owner>/<slug>`, wait for `ready`).

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
