# autoresearch — 垃圾分类 (Garbage Classification)

This is an experiment to have an AI agent autonomously optimize a garbage classification model.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, image loading, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/data/train/` contains image subdirectories. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Task: Garbage Classification (垃圾分类)

The goal is to build a computer vision model that classifies garbage images into categories. The default dataset is organized into Chinese waste categories following the national 4-category standard:
- **可回收物 (Recyclable)**: paper, plastic, glass, metal, cardboard
- **有害垃圾 (Hazardous)**: batteries, bulbs, medicine, paint
- **厨余垃圾 (Kitchen waste)**: food scraps, fruit peels, bones
- **其他垃圾 (Other waste)**: tissues, ceramics, non-recyclable items

The baseline uses TrashNet data mapped to these categories. The framework auto-detects classes from subdirectory names, so it works with any dataset structure.

## Experimentation

Each experiment runs on a single GPU (CUDA or MPS). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture (backbone, head, attention mechanisms), optimizer, hyperparameters, training loop, batch size, data augmentation strategy, learning rate schedule, regularization, etc.
- Experiment with different backbones: MobileNetV2, ResNet, EfficientNet, Vision Transformer (ViT), ConvNeXt, etc.
- Try different training strategies: transfer learning, progressive unfreezing, mixup/cutmix, knowledge distillation, etc.
- Modify the classification head: add attention, use different pooling, multi-scale features, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, transforms, and training constants (time budget, image size, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_acc.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. The target platform has 24GB (Apple M4 Pro). Some increase is acceptable for meaningful val_acc gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_acc:          0.850000
val_f1:           0.820000
val_loss:         0.450000
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     2048.0
num_steps:        953
num_epochs:       15
num_params_M:     3.5
num_trainable_M:  1.2
num_classes:      4
batch_size:       32
```

You can extract the key metric from the log file:

```
grep "^val_acc:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_acc	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_acc achieved (e.g. 0.850000) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 2.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_acc	memory_gb	status	description
a1b2c3d	0.850000	2.0	keep	baseline MobileNetV2
b2c3d4e	0.875000	2.1	keep	increase head LR to 3e-3
c3d4e5f	0.840000	2.0	discard	switch to ResNet50 (worse)
d4e5f6g	0.000000	0.0	crash	EfficientNet-B7 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_acc:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_acc improved (higher), you "advance" the branch, keeping the git commit
9. If val_acc is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**Ideas to try** (non-exhaustive):
- Different backbones: ResNet18/34/50, EfficientNet-B0/B1, ConvNeXt-Tiny, ViT-Small
- Classifier head designs: add BatchNorm, change hidden dims, add SE attention
- Progressive unfreezing: unfreeze more backbone layers over time
- Data augmentation: CutMix, MixUp, RandAugment, AutoAugment
- Learning rate: different schedules, warm restarts, different head/backbone LR ratios
- Regularization: dropout rates, stochastic depth, weight decay tuning
- Label smoothing values
- Batch size tuning (trade off noise vs. updates per time budget)
- Test Time Augmentation (TTA) during evaluation
- Feature Pyramid or multi-scale feature fusion
- Attention mechanisms in the classifier head (CBAM, SE, etc.)

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. The loop runs until the human interrupts you, period.
