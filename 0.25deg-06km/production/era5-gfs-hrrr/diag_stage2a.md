# Stage2a diagnosis: why ERA5 pretraining doesn't beat training gfs-hrrr from scratch

Started 2026-10-06.

## Current status (updated 2026-10-09) — read this first

- **2026-10-09: LR sweep submitted** — t5 (3e-3) 59599636, t6 (7e-3) 59599637, t7 (1e-2) 59599638, all warmup 1000;
  t4 rerun 59599236 (first t4 job died of NODE_FAIL at step ≈ 780, epoch-0 val 0.0492). See §10.
  The scratch control is deliberately deferred: first map how sensitive stage2a is to LR.

- **Main finding so far:** with tp=8 and LR 1e-3, the ERA5-pretrained stage2a (t1) ties the gfs-hrrr
  scratch reference at the end (val 0.0204 vs 0.0203). It is ahead for the first ~40 epochs.
  So pretraining is **faster, not better** (§3).
- **LR stability edge is about 1e-2.** t3 (peak 5e-2, warmup 3000) diverged at step ≈ 761 (LR ≈ 1.27e-2)
  and was NaN from step ≈ 776. Before that, LRs of 5e-3 to 1.2e-2 trained faster than t1 did at 1e-3 (§8).
- **Running now:** t4 = peak LR 5e-3, warmup 1000, otherwise the same as t1. Job **59492299**, submitted
  2026-10-07 and PENDING at the time of writing (§9).
- **When resuming:** run `sacct -u bhuang95 -S 2026-10-07`, then check the t4 step loss after step 1000
  against the decision rules in §9.3. When the job hits walltime, make the 5 resume edits (§9.2).
- **Still open:** the matched gfs-hrrr scratch control at 5e-3; the frozen-backbone trial (§5.1);
  obs-metrics for t1 and t2.

## 1. Background

The transfer chain:

```
stage1a  (1-deg ERA5,     stage1a_bo.yaml)
  -> stage1b  (0.25-deg ERA5,  stage1b_bo.yaml)
    -> stage2a  (gfs-hrrr,       stage2a_bo.yaml)
```

The production stage2a did worse than training on gfs-hrrr from scratch
(`../gfs-hrrr/stage1a_bo.yaml`). We are testing ways to improve stage2a.

## 2. Experiments

All runs use 60000 steps, a cosine schedule with `warmup: 1000` (except t3: 3000) and `min: 3e-7`, AdamW
(betas 0.9/0.95, weight decay 0.1), 16 nodes × 4 GPUs and `num_gpus_per_model: 4`.

`training.lr.rate` in the yaml is multiplied by num_nodes × gpus_per_node / gpus_per_model
= 16 × 4 / 4 = **16**. So **effective max LR = 16 × rate**.

| label in figure | yaml | init | trainable_parameters (data/hidden/d2h/h2d) | `lr.rate` | effective max LR | mlflow run(s) |
|---|---|---|---|---|---|---|
| gfs-hrrr-s1a (reference, black) | `../gfs-hrrr/stage1a_bo.yaml` | scratch | 8/8/8/8 | 6.25e-5 | **1e-3** | 6289a755 + 0515305b |
| era5-gfs-hrrr-s2a (blue) | `stage2a_bo.yaml` | stage1b 61e40fef | 0/0/0/0 | 6.25e-6 | **1e-4** | 245d94e9 + 9925759a |
| era5-gfs-hrrr-s2a_t1 (green) | `stage2a_t1_bo.yaml` | stage1b 61e40fef | 8/8/8/8 | 6.25e-5 | **1e-3** | 94e381bc + 872d9907 |
| era5-gfs-hrrr-s2a_t2 (red) | `stage2a_t2_bo.yaml` | stage1b 61e40fef | 8/8/8/8 | 3.125e-5 | **5e-4** | 3bf770ae + 417fbbd1 |
| s2a_t3 (probe, warmup 3000) | `stage2a_t3_bo.yaml` | stage1b 61e40fef | 8/8/8/8 | 3.125e-3 | **5e-2** | 929f (v_num); job 59454952, **NaN at step ≈ 776** (§8) |
| s2a_t4 | `stage2a_t4_bo.yaml` | stage1b 61e40fef | 8/8/8/8 | 3.125e-4 | **5e-3** | 59492299 NODE_FAIL at step ≈ 780 (not divergence); rerun from step 0 as 59599236 (§9) |
| s2a_t5 | `stage2a_t5_bo.yaml` | stage1b 61e40fef | 8/8/8/8 | 1.875e-4 | **3e-3** | job 59599636 (§10) |
| s2a_t6 | `stage2a_t6_bo.yaml` | stage1b 61e40fef | 8/8/8/8 | 4.375e-4 | **7e-3** | job 59599637 (§10) |
| s2a_t7 | `stage2a_t7_bo.yaml` | stage1b 61e40fef | 8/8/8/8 | 6.25e-4 | **1e-2** | job 59599638, edge probe (§10) |

Figure: `stage2c/diagnostics/plots/figures-era-gfs-hrrr/loss_vs_epoch_stage2a_diag_logy.png`
(made by `stage2c/diagnostics/plots/plot_loss_curves_stage2a.py`).

## 3. Results (epoch-mean `multi_dataset_loss`, from mlflow)

| run | train ep1 | train ep5 | train ep20 | **train final** | val ep1 | val ep5 | val ep20 | **val final (= min)** |
|---|---|---|---|---|---|---|---|---|
| reference gfs-hrrr-s1a, tp8, 1e-3 | 0.1584 | 0.0320 | 0.0191 | **0.0128** | 0.0869 | 0.0377 | 0.0252 | **0.0203** |
| s2a, tp0, 1e-4 | 0.1554 | 0.0437 | 0.0271 | 0.0210 | 0.1074 | 0.0506 | 0.0336 | 0.0273 |
| s2a_t1, tp8, 1e-3 | **0.1056** | **0.0287** | **0.0184** | 0.0130 | **0.0658** | **0.0343** | **0.0245** | 0.0204 |
| s2a_t2, tp8, 5e-4 | 0.1197 | 0.0327 | 0.0206 | 0.0147 | 0.0781 | 0.0383 | 0.0269 | 0.0219 |

All runs completed the full 60000 steps, about 83 epochs.

### What the numbers say

1. **s2a_t1 has caught up to the reference.** Its final val loss is 0.0204 vs 0.0203,
   a tie within noise. The ERA5 init gives a clear head start: at epoch 1 the val loss is 0.066
   vs 0.087, and t1 stays ahead until about epoch 40. The advantage then washes out.
   So the remaining issue is that pretraining makes training **faster but not better**.
2. **The original stage2a gap was mostly about capacity, not LR.** The blue run has tp=0,
   which removes about 67M trainable node and edge embedding parameters on the
   cut084-13.nn0532 graph. That is more than the 57.5M backbone. Blue vs black changes
   two things at once, so it doesn't isolate the LR effect.
3. **t1 vs t2 is the only clean LR comparison.** It shows that with tp=8, 5e-4 is
   worse than 1e-3 in both train loss (0.0147 vs 0.0130) and val loss (0.0219 vs 0.0204).
   Train loss is also higher, so t2 is **under-optimized, not overfitting**. That supports
   "LR matters" for this 60k-step budget, but only over the 5e-4 → 1e-3 range.
4. The new trainable embeddings are **zero-initialized** (`TrainableTensor` → `nn.init.constant_(…, 0)`
   in anemoi-models `layers/graph.py`). Their parameter blocks therefore start from nothing
   and have to learn under the same LR as the pretrained backbone.

## 4. Question: raise max LR to 5e-2 and lengthen the warmup?

### Max LR = 5e-2: no

- 5e-2 is **50×** the reference's 1e-3 (`rate: 3.125e-3`). For a graph transformer with
  AdamW and 16-mixed precision, an LR that high will almost certainly diverge or produce
  loss spikes. `gradient_clip` (value 32) won't save it, because clipping limits gradient
  size while Adam's step size is set by the LR.
- The evidence doesn't extrapolate that far. We have one doubling (5e-4 → 1e-3) that helped,
  and at 1e-3 the pretrained run already matches the scratch run.
- If "5e-2" was a typo for **5e-3**, that is a plausible upper bracket. It is still
  aggressive and needs a longer warmup.

### Recommended LR test

- **Primary: 2e-3** (`lr.rate: 1.25e-4`). This is a 2× step, the same factor that helped between t2 and t1.
- **Optional upper bracket: 3e-3** (`lr.rate: 1.875e-4`). Only run this if 2e-3 is stable and better.
  Treat 5e-3 (`3.125e-4`) as the stability-limit probe.
- **Critical control:** run the gfs-hrrr-from-scratch reference at the same LR. If a higher
  LR helps, it will probably help the scratch run too. Without the matched reference we
  can't tell "a better LR" apart from "pretraining helps". The question we care about is
  pretrained vs scratch at equal settings.

### Warmup

- The current warmup is 1000 steps: about 1.7% of 60k, or about 1.4 epochs at ~730 steps/epoch.
  For a peak LR of 1e-3 that is reasonable.
- A longer warmup mainly buys **stability** at a higher peak LR. It has little effect on
  the final loss after a full cosine decay.
- In the transfer setting it does have one real benefit. The zero-initialized embeddings
  and any freshly initialized layers produce large, noisy gradients early on. A gentler ramp
  limits how much those gradients disturb the pretrained backbone before the new parameters settle.
- Recommendation: **warmup 2000-3000 steps** (about 3-5% of 60k) when going to 2e-3 to 3e-3. If you stay at 1e-3, keep 1000.
- Changing the warmup and the peak LR together confounds them again. Either change both
  in the pretrained run *and* the matched scratch run, or accept that the run tests "the
  recipe" rather than each knob on its own.

## 5. Levers that target the actual finding (faster, not better)

Since t1 already ties the reference, a different LR probably won't make pretraining
*beat* scratch. These options aim at the transfer problem more directly:

1. **Two-phase / frozen-backbone warmup.** Train with `submodules_to_freeze: [processor]`
   (plus the encoder and decoder if needed) for about 2-5k steps, so the zero-initialized
   embeddings adapt to the frozen pretrained backbone. Then unfreeze and run the normal
   schedule. This protects the ERA5 features from early destructive updates.
2. **Equal-compute comparison.** The head start is worth about 20-30 epochs. Running the
   pretrained model with `max_steps` of about 20-30k is a legitimate win if it reaches the
   same val loss in half the compute.
3. **Discriminative LR.** Use a higher LR for the new embeddings and a lower one for the
   pretrained backbone. This needs param groups in the optimizer setup, which means a code
   change in anemoi-training.
4. **Evaluate beyond `multi_dataset_loss`.** Pretraining may help per-variable skill or
   longer rollouts, such as obs-metrics on the global and HRRR domains, even when the
   aggregate one-step loss ties.

## 6. Next steps

- [x] Decide on the LR/warmup trial. Tried t3 at 5e-2 / warmup 3000, which diverged (§8).
      Then t4 at 5e-3 / warmup 1000, submitted 2026-10-07 (§9).
- [ ] Matched gfs-hrrr scratch run at the t4 settings (5e-3, warmup 1000).
- [ ] Consider the frozen-backbone two-phase trial (§5.1).
- [ ] Run inference / obs-metrics for s2a_t1 and s2a_t2 vs the reference (configs already exist
      under `stage2c/diagnostics/`).

## 7. Follow-up Q (2026-10-06): is max LR 5e-2 with a 3000-step warmup worth trying?

**Not as a full 60k-step run.** It is worth at most a short probe to find where training becomes unstable.

- **Most of the run would sit at very high LR.** With a linear warmup to 5e-2 over 3000 steps, the LR passes 1e-3 (the reference value) at step 60. It passes 5e-3 at step 300 and 1e-2 at step 600. After warmup, the cosine decay keeps the LR above 1e-3 until about 91% of the run. Nearly the whole run would be 5-50× above any LR we have evidence for.
- **The warmup changes when it fails, not whether.** A longer warmup does not make 5e-2 stable. It only moves the point where the run diverges later, to somewhere between a few hundred and a few thousand steps.
- **A failure would show up early, so a probe is cheap.** Set `lr.rate: 3.125e-3` and `warmup: 3000`, but cap `max_steps` at about 4000 so the probe stops early. Keep `lr.iterations` at 60000 so the LR curve matches the full run. A few hours on the debug queue is enough. Watch for loss spikes, NaN, or a train loss that is worse than t1's at the same step.
- **Expected outcome:** instability somewhere between about 5e-3 and 1e-2. The probe would tell us the stability limit. It would not give a better model.
- **Recommended full run:** 2e-3 (`rate 1.25e-4`) with `warmup 2000-3000`, plus the matched gfs-hrrr scratch run at the same LR.

**Decision:** we will try 5e-2 anyway, as **stage2a_t3** (`stage2a_t3_bo.yaml`).
- The yaml is a copy of `stage2a_t2_bo_1st.yaml`: a fresh fork of stage1b 61e40fef with trainable parameters 8/8/8/8.
- It changes `lr.rate: 3.125e-03` (5e-2 effective) and `warmup: 3000`. Output goes under `stage2a_t3/`.
- `max_steps` stays at 60000, so the LR schedule is the real full-run one.
- The job is submitted with a short walltime and ends by itself. Check for loss spikes or NaN vs t1 at the same steps.
- If it looks good, resume with the usual 5 edits (`run_id`, `fork_run_id: null`, `load_weights_only: false`, `transfer_learning: false`, `warm_start: null`).

## 8. t3 result (job 59454952, 2026-10-07): diverged at effective LR ≈ 1.25e-2

From the progress-bar step loss in `/pscratch/sd/b/bhuang95/nested-eagle/slurm/stage2a_t3_bo.59454952.out`:

- **Epoch 0 (steps 0-726, LR ramping 0 → 1.2e-2) was stable and *faster* than t1.**
  Epoch-0 train loss was 0.0761 vs 0.1056 for t1, and val loss was 0.0507 vs 0.0658.
  Step loss fell smoothly to about 0.042.
- **Blow-up at global step ≈ 761 (LR ≈ 1.27e-2).** The loss crept up from 0.041 to 0.052, then went
  0.093 → 0.30 → 1.3 → 11.5 → 26.6. It was **NaN from step ≈ 776** (LR ≈ 1.3e-2). The job then
  burned about 1 h computing NaN until it was cancelled.
- **Conclusion:** the stability edge is about 1e-2. LRs of 3e-3 to 8e-3 were crossed without trouble,
  and the higher LR made the model learn faster early on. That is evidence that 1e-3 is
  below the optimum for this budget.

### Recommended t4

- Peak **5e-3** (`rate: 3.125e-4`), which is about 0.4× the divergence LR.
  The cosine schedule keeps the LR near its peak for thousands of steps, so the run needs a margin.
- Keep `warmup: 1000`, the same as t1 and the reference, so LR is the only knob that changes.
  t3 ramped 0 → 5e-3 in 300 steps without trouble; a 1000-step ramp is >3× gentler (see §9.1).
- Run the **matched gfs-hrrr scratch control at 5e-3** with the same warmup.
- Fallback if t4 spikes: 3e-3 (`rate: 1.875e-4`).

## 9. stage2a_t4: peak LR 5e-3, warmup 1000 (set up 2026-10-07)

### 9.1 Why these settings (evidence from t3, and its limits)

The t3 warmup was linear: LR = 5e-2 × step / 3000. It reached 5e-3 at step 300, 1e-2 at step 600 and
1.2e-2 at step 726. The step loss from the progress bar of job 59454952:

| step | effective LR | step loss (approx.) |
|---|---|---|
| 300 | 5.0e-3 | 0.064 |
| 400 | 6.7e-3 | 0.053 |
| 500 | 8.3e-3 | 0.048 |
| 600 | 1.0e-2 | 0.050 |
| 726 | 1.2e-2 | 0.042 |

- From step 300 to 726 the loss kept falling. No step rose more than 15% above the one before it.
- **Mild warning signs:** the loss stopped improving and rose about 10-15% around steps 404-412 (LR ≈ 6.8e-3)
  and again around steps 583-598 (LR ≈ 1e-2), then recovered both times.
- **What t3 supports:** *passing through* 5e-3 while the LR is rising caused no instability. A
  1000-step ramp to 5e-3 raises the LR by 5e-6 per step, compared with 1.7e-5 per step in t3. That is more
  than 3× gentler, so the warmup itself is very unlikely to cause trouble.
- **What t3 does NOT show:** that 5e-3 is stable when *held* for a long time.
  1. t3 spent only a few dozen steps at each LR. The cosine schedule keeps t4 near its peak for thousands of steps.
     Divergence can build up gradually: in t3 the loss crept upward for about 10 steps before the blow-up.
  2. The loss surface usually gets sharper as training goes on (progressive sharpening), so the highest stable LR
     goes down. An LR that is fine at step 400 may not be fine at step 5000.
  3. This is one run with one seed.
- So the peak is set at **about 0.4× the divergence LR** (5e-3 vs ≈ 1.25e-2) rather than just below 1e-2.
  Warmup stays at 1000, the same as t1 and the scratch reference, so **peak LR is the only knob that differs from t1**.

### 9.2 Configuration

| item | value |
|---|---|
| yaml | `stage2a_t4_bo.yaml` (fresh-fork copy kept as `stage2a_t4_bo_1st.yaml`) |
| derived from | `stage2a_t3_bo.yaml`; only the LR, warmup, experiment name and root changed |
| init | fork of stage1b `61e40fefaa664a9c8261eb57063971c2` (`last.ckpt`), `transfer_learning: true`, `load_weights_only: true` |
| trainable_parameters | 8/8/8/8 |
| `lr.rate` | `3.125e-04` → **effective max LR 5e-3** (16 × rate) |
| `lr.warmup` | 1000 |
| `lr.min` / `iterations` | 3e-7 / `${training.max_steps}` = 60000 |
| output root | `$SCRATCH/nested-eagle/0.25deg-06km/production/era5-gfs-hrrr/stage2a_t4/` |
| mlflow experiment | `0.25deg-06km production era5-gfs-hrrr stage2a_t4` |
| submit script | `slurm/submit_stage2a_t4_bo.sh` (16 nodes × 4 GPU, 12 h walltime) |
| log | `/pscratch/sd/b/bhuang95/nested-eagle/slurm/stage2a_t4_bo.<jobid>.out` |

At about 36 min per epoch, the full run of about 83 epochs takes about 50 h. Expect 4-5 resubmits.
To resume, make the usual 5 edits to `stage2a_t4_bo.yaml`: `run_id`, `fork_run_id: null`,
`load_weights_only: false`, `transfer_learning: false` and `warm_start: null`.

### 9.3 What to watch, and decision rules

- **First 3-4 epochs:** watch the step loss, especially after step 1000, when the LR reaches its peak and stays near it.
  A steady upward creep, or jumps like t3's at steps 750-760, means stop the job.
  Don't let it run on NaN: t3 wasted about 1 h after its loss went NaN.
- **If it is unstable:** make t5 = 3e-3 (`rate: 1.875e-4`) with warmup 1000.
- **If it is stable:** compare with t1 (1e-3) and the reference at matched epochs and at the end.
  Compare with t3's epoch 0 (train 0.0761, val 0.0507) to see the early-speed gain.
- **Still to do:** run the matched gfs-hrrr from-scratch control at 5e-3 / warmup 1000. Without it, a t4 gain
  can't be credited to pretraining (see §4).

### 9.4 Status

- [x] Submitted 2026-10-07: job id 59492299 (PENDING at the time of writing)
- [ ] Stable past step ~3000?
- [ ] Final train/val `multi_dataset_loss`: ____ / ____

## 10. LR sweep t5/t6/t7 (submitted 2026-10-09)

Each yaml is a copy of `stage2a_t4_bo.yaml` (fresh-fork mode). Only `lr.rate`, the experiment name and the output root changed.
Fresh-fork copies are kept as `stage2a_t{5,6,7}_bo_1st.yaml`. Submit scripts are `slurm/submit_stage2a_t{5,6,7}_bo.sh`
(copies of the t4 script with a 4 h walltime). Warmup is 1000 for all.

Pretrained sweep at tp=8, warmup 1000: 5e-4 (t2), 1e-3 (t1), 3e-3 (t5), 5e-3 (t4), 7e-3 (t6), 1e-2 (t7).

Why each LR (all relative to the t3 divergence edge of about 1.25e-2):

- **t5, 3e-3 (about 0.24× the edge).** The safe point of the sweep, and the fallback if t4 at 5e-3 becomes unstable once it is held at its peak.
  t1, t5 and t4 (1e-3, 3e-3, 5e-3) are roughly evenly spaced on a log scale, so they show whether the gain is still growing at 5e-3 or has flattened out.
- **t6, 7e-3 (about 0.55× the edge).** Aggressive, but it has a real chance of finishing 60k steps. If the final loss keeps improving from 5e-3 to 7e-3,
  the optimum sits near the stability edge. If it gets worse, the optimum lies between 3e-3 and 5e-3.
- **t7, 1e-2 (at the edge).** A probe for where training breaks when the LR is *held* near its peak. t3 only passed through 1e-2 for a few dozen steps
  (and already wobbled at steps 583-598). With the cosine schedule, t7 stays near 1e-2 for thousands of steps, and the highest stable LR usually
  falls as training goes on. Expected outcome: NaN within the first few thousand steps, with no final loss to compare.
- Warmup is kept at 1000 for all runs. t3 crossed these LRs with a ramp more than 3× steeper, so warmup isn't the limiting factor.
  Keeping it at 1000 also leaves peak LR as the only knob that differs from t1.
- The matched gfs-hrrr scratch control is deferred. The plan is to map stage2a's LR sensitivity first, then run scratch at the best LR.

Walltime: the submit scripts use 4 h, which is about 6 epochs at about 36 min/epoch. A full 83-epoch run therefore needs about 14 submissions,
each with the 5 resume edits. To cut down on resubmits, raise the walltime of whichever runs survive (t4 originally used 12 h).

- t4 epoch 0 (job 59492299, before the node failure): val 0.0492 (t1 0.0658, t3 0.0507); step loss ≈ 0.041 at step 726.
  Watch for hangs after an NCCL error: that job sat hung for about 7 h until the node-failure cancel.
- t7 (1e-2) is expected to diverge. t3 blew up at ≈ 1.27e-2 and was already wobbling at ≈ 1e-2. Cancel it at the first
  sustained loss creep or NaN; don't resubmit it unless it is clearly stable past about 5k steps.
- Resume procedure is unchanged: the usual 5 edits (`run_id`, `fork_run_id: null`, `load_weights_only: false`,
  `transfer_learning: false`, `warm_start: null`).

| run | job | stable past 3k? | final train / val |
|---|---|---|---|
| t5 3e-3 | 59599636 | yes (TIMEOUT at step ≈ 4356, 6 epochs, no spikes) | ep6: 0.0237 / 0.0294 |
| t4 5e-3 | 59599236 | yes (TIMEOUT at step ≈ 4300, 5 epochs, no spikes) | ep5: 0.0242 / 0.0300 |
| t6 7e-3 | 59599637 | **no: NaN between steps 2999 and 3099** (LR ≈ 7e-3, held at peak), epochs 5-6 NaN; last clean ckpt epoch_003-step_002904 | ep4: 0.0260 / 0.0319 |
| t7 1e-2 | 59599638 | | |

### 10.1 Assessment (2026-10-13): can t4/t5 beat the reference?

Val at matched epochs (ref / t1 / t5 / t4 / t6): ep1 0.0869/0.0658/0.0533/0.0506/0.0495; ep4 0.0407/0.0368/0.0320/0.0309/0.0319;
ep5 0.0377/0.0343/0.0311/0.0300/NaN. t4's epoch-5 val is about where t1 was at epoch 8 and the reference at epoch 10.

- An early lead doesn't predict the final loss here: t1 led the reference by 9% at epoch 5 and ended tied. So 5-6 epochs can't show a final gain.
- t6 diverged while held at 7e-3. The highest stable LR when held at peak is therefore below 7e-3, and t4 (5e-3) is about 0.7× that. Cosine keeps t4 above 4e-3 until step ≈ 18k, so its risk isn't over.
- Speed peaks around 5e-3: t6 was already no better than t4 at epoch 4.
- Recommendation: continue t5 (safest) and, if budget allows, t4 too. Don't resume t6. Any final gain needs the matched scratch control at the same LR before it can be credited to pretraining.

### 10.2 Re-assessment (2026-10-15): t5 at 13 epochs, t4 at 12 (jobs 59744094/59744095 TIMEOUT)

Val at ep11: ref 0.0292, t1 0.0272, t5 0.0253, t4 0.0251. t4 and t5 are indistinguishable from ep8 on, so the speed gain saturates by 3e-3.
How far ahead (the epoch at which the ref reaches the same val): t5 is 4.2 epochs ahead at ep4, 7.3 at ep8 and 9.4 at ep12. t1 was about 2 ahead early, but it ended behind (ref reached 0.0204 at ep69, t1 only at ep81).
**Key diagnostic: val at matched TRAIN loss.** Every run lies on the same train→val curve as the ref, within ±2% (t5 ep12 −2.3%, t4 −1.5%; t1 at ep20/40/60/81 −1.2%/0/−0.5%/−0.1%).
So pretraining plus higher LR moves the model along the *same* trajectory faster; it doesn't reach a better-generalizing solution. The final val is set by the final train loss, and the annealed tail sets that.
Expectation: t5 run to 83 epochs probably ends within about 1-2% of the ref (0.0203). A clear win is unlikely, though the −2.3% at ep12 is the first hint of a deviation.
Options: (a) continue t5 only, as the definitive test (~70 epochs ≈ 42 h); (b) more useful: rerun t5 with a shorter schedule (max_steps ≈ 30k) to show it matches the ref at about half the compute. Either way a final gain needs the 3e-3 scratch control before it can be credited to pretraining.
