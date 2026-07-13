# Chapter 3 — Post-hoc uncertainty quantification for the LSTM streamflow ensemble

**Consolidated 2026-07-08** from `continue.md`, `conversation.md`, `methods.md`, and
`ood_mean_evaluation_framing.md` (which this file replaces). The forward-looking content is
synthesized and de-duplicated below; the **dated methods decision log is preserved verbatim in
Appendix A** (append-only, per the project convention — do not rewrite past entries in place, add
dated revisions).

---

## 1. Project & current state

10-member LSTM ensemble predicts daily streamflow at ~269 BC/Alberta basins, 1980–2022, in physical
units (mm/day, specific runoff = m³/s × 86.4 / basin_area_km2). Members are **point predictors**
(mean only), differing by random init + SGD noise (a deep ensemble, Lakshminarayanan 2017).

**Architecture (current):** `StandardLSTM` — 1-layer `nn.LSTM`, hidden 256, dropout 0.4, ~272K
params. `h_T = out[:, -1, :]` (final-step hidden state, **pre-dropout**); `prediction =
fc(dropout(h_T))`, `fc = Linear(256, 1)`, no output activation. Sequence-to-one: a 365-day window →
next-day discharge. Recently swapped **EA-LSTM → StandardLSTM** (better in-distribution, justifiable
for Ch1, ~10× faster to train). Trained with **basin-averaged NSE\*** loss (per-basin
variance-normalized MSE).

**Splits (`config.py`):** train ≤2005 · val 2006–2012 · test 2013–2022. The two 2021 OOD events sit
in test.

**Ch3 is a post-hoc UQ paper:** extract a predictive variance *without retraining the mean*, off the
frozen hidden state / frozen mean.

---

## 2. The methodological decision the literature has to support

**Post-hoc, mean-frozen, heteroscedastic, distribution-light variance extraction — landing on a
heteroscedastic Student-t head as the "fork point."** `r | x ~ t_ν(0, σ(x))`, σ(x) read from `h`, ν
global or input-dependent.

### Hard constraints (the goal)
- **Extract a per-prediction σ²(x)** from the frozen hidden state, *after* the mean head, so each
  (station, day) gets a `(ŷ, σ²)` pair.
- **Do not change the mean.** Trained point predictions and checkpoints stay byte-identical.
- **σ² here is ALEATORIC** (input-conditional data noise), *not* epistemic. The 10-member ensemble
  spread is the epistemic piece. Total predictive variance = aleatoric (head) + epistemic
  (between-member). Do not conflate.

### Rationale chain
1. **Frozen mean.** Lineage: faithful heteroscedastic regression — Seitzer et al. (β-NLL, ICLR 2022)
   show Gaussian-NLL's inverse-variance gradient weighting degrades the mean; Stirn et al. (AISTATS
   2023) cure it by stop-gradient decoupling (provably faithful mean); Sluijterman et al.
   (Neurocomputing 2024) find post-warmup freeze ≈ joint. **We go further:** train entirely off the
   frozen hidden state + frozen mean — the conservative *endpoint* of the faithful-regression
   lineage, not an exotic choice. This also makes the "don't let variance corrupt the mean"
   machinery **moot** (the mean is immutable → plain NLL trains cleanly).
2. **Heteroscedastic** — residuals fan out with flow (verified empirically).
3. **Heavy-tailed / non-Gaussian** — standardized residuals are skewed and high-kurtosis (excess
   kurtosis ~78 low-flow → ~300 high-flow; skew flips sign by regime; lag-1 autocorr ~0.82). Plain
   Gaussian MVE is mis-specified in the tails → **Student-t**, which is a Gaussian **scale mixture**
   (N(0,s), s ~ Inverse-Gamma, marginalized; Andrews & Mallows 1974, West 1987) — Bayesian-native
   (a hierarchical prior on the variance), not a frequentist device.
4. **The fork.** Student-t is the simplest object that *frames both rabbit holes without entering
   either*:
   - enrich the mixing distribution (one IG → asymmetric finite/infinite mixture) ⇒ **CMAL/UMAL**
     (aleatoric; Klotz et al. 2022 HESS — already done in hydrology, so retrain into it only if a
     result *motivates* it);
   - make the **weights** uncertain instead of the scale ⇒ **Bayesian DL** (epistemic; the claimed
     "beyond GLUE for extremes" gap).
5. **More diagnostic than Gaussian.** A Gaussian fails *everywhere* here, so its failure says nothing
   about which fork. The t absorbs symmetric heavy tails; what *remains* unexplained is the signal —
   residual skew / regime-dependence the t can't hold → CMAL/UMAL; scale failing to inflate at the
   OOD events where the mean degrades → Bayesian (the **D-blind** signature, §6).

### Rejected alternatives (and why)
- **Gaussian MVE** (Nix & Weigend 1994) — the null baseline; fails everywhere, low diagnostic value.
  Kept only as the reference the t improves on.
- **Quantile regression + conformal (CQR, Romano 2019)** — trades the distributional assumption for
  an **exchangeability** assumption that breaks precisely at the OOD events; gives **marginal** (not
  conditional) coverage ("how often", not "why"); frequentist framing that doesn't compose with the
  Bayesian fork. (Open: does a pinball/conformal layer *approximate* CMAL cheaply? Skeptical.)
- **CMAL/UMAL** (Klotz 2022) — the **destination, not the tool**. Using it *as* the VE collapses the
  CMAL-vs-Bayesian framing.
- **Deep Evidential Regression** (Amini 2020) — gives a t-marginal via Normal-Inverse-Gamma (=
  scale mixture; the analytic core of our hierarchical prong), but its **epistemic** output is not a
  well-identified posterior (Meinert 2023 AAAI; Juergens/Meinert 2024 ICML — NIG overparameterized,
  "epistemic" is a convergence-speed proxy). May borrow the t-marginal; **decline** its epistemic —
  that is exactly the pseudo-Bayes the "beyond GLUE" novelty aims to surpass.
- **MC-dropout** (Gal 2016) — Klotz benchmark; crude variational epistemic, not aleatoric; baseline.

### Novelty positioning (post lit-check, 2026-06-19)
The **head form is no longer novel** — Pourkamali-Anaraki (TDistNN, NCA 2026, arXiv:2503.12354)
already swaps Gaussian NLL for a location/scale/df Student-t NLL (but end-to-end, no scale-mixture
justification). So novelty rests on: (a) the **post-hoc, mean-frozen, residual-trained**
configuration; (b) the **scale-mixture / hierarchical-variance** justification; (c) the
**hydrology-extremes** application; (d) the **CMAL-vs-Bayesian fork** framing. **GAP 2 unresolved:**
do *not* yet assert "formal Bayesian DL beyond GLUE for hydrological extremes is undone" — the
lit-check was GAP-1-concentrated; that bet needs a dedicated hydrology-venue check first.

---

## 3. Implementation

### Current approach — offline head on extracted hidden states
The mean ensemble is trained and frozen. Rather than an in-model variance path, we **extract the
frozen `h_T` to disk** and train a fully separate head offline:
- `--save_hidden` on a training/inference run writes `results_MVE/states/hidden_member_*.npy` +
  `hidden_index.json` (the (date, station) grid, 15706 × 269, float16).
- `src/build_index.py` pins the canonical grid and emits the **index contract**
  (`data/output/results_MVE/`): `index_dates.csv`, `y_obs_mm.npy`, `valid_mask.npy` (obs finite AND
  pred finite), `q_std.csv`, `meta.json`. All arrays align positionally `arr[date_i, station_j]`;
  station axis is sorted so `q_std` maps by position.
- `src/mve_*` (planned) + `run_variance_head.py` train the head off the contract; **HPC Job 2**
  (`hpc/variance/`) runs it as a single short GPU job, gated behind manual EXP5 validation of the
  ensemble (see the split hpc plan — Job 1 ensemble array → manual EXP5 check → Job 2 head).

### Training principles (constant across designs)
- **Freeze everything but the head;** hold the trunk in **`model.eval()`** so trunk dropout is OFF
  and `h` is deterministic (do NOT call `model.train()`). Backprop only into the head.
- Fit the head on **held-out** years (val 2006–2012) — residuals on the mean's training years are
  over-optimistic. Early-stop/select on val; **calibrate/report on test 2013–2022** (never used to
  fit either head).
- Save the head **separately** (e.g. `*_varhead.pth`) so mean `.pth` files are never overwritten.

### Loss & the variance target
Student-t NLL on **standardized** residuals `z = r / q_std` (mirrors NSE\*, keeps high-flow basins
from dominating; physical scale recovered later as `σ_phys = q_std · scale`). Standardized vs raw
mm/day was the key sub-decision — standardized is the default. σ² at the NLL optimum estimates
`E[(y−μ)²|x] = Var + bias²` and is **distribution-free**; "Student-t/Gaussian" is only an
interpretation overlay at inference. The model is a **two-moment estimator with a swappable
interpretation layer.**

### Superseded design (kept for the trail)
`continue.md`'s original plan attached the variance path **inside** the model: an `encode()` refactor
so `forward()` is unchanged, then `forward_with_var` computing `s = var_head(h.detach())`,
`var = softplus(s)+1e-6`, with a **Gaussian** NLL and load via `strict=False`. This was written for
the **EA-LSTM**. Superseded by (a) the EA-LSTM→StandardLSTM swap and (b) the extract-to-disk offline
head. The *principle* (frozen mean, head off `h`, held-out fit, calibration acceptance) carries over
unchanged; only the mechanism (in-model detached head + Gaussian → offline head on extracted `h` +
Student-t) changed. `h.detach()` still matters conceptually: it blocks weight-gradients at train
time only, so input-gradients at inference still flow through the frozen trunk (relevant if the
deferred sensitivity tool is revived, §5).

### Checklist / definition of done
- [ ] `src/mve_head.py`: `VarianceHead` (256 → hidden → t params) + Student-t NLL.
- [ ] `src/mve_dataset.py`: assemble tensors from the contract (`h`, `z = r/q_std`, split).
- [ ] `src/mve_training.py`: NLL loop (reuse device/checkpoint/early-stop helpers from `training.py`).
- [ ] `run_variance_head.py` (mirror `run_training.py`): load frozen mean, freeze, eval-mode loop,
      fit on val, save `*_varhead.pth` separately.
- [ ] `src/mve_inference.py`: per-cell σ_a, ν; combine with σ_e² → total predictive variance.
- [ ] **Calibration acceptance** (test years): coverage of intervals, **PIT on the full sample**,
      reliability — this is the acceptance test.
- [ ] `methods.md`/Appendix A: dated entry recording standardization choice, split, likelihood.

**Done =** a trained head (mean untouched, verified) whose test-year intervals are **calibrated**
(coverage ≈ nominal, flat-ish PIT) under a likelihood whose assumptions were checked — or a
documented decision to escalate to log-space / CMAL if the t fails calibration/shape checks.

---

## 4. Variance decomposition & interpretive backbone

### Law of total variance
`Var(y|x) = E_θ[Var(y|x,θ)] (aleatoric σ_a²) + Var_θ[E(y|x,θ)] (epistemic σ_e²)`.
Point-predictor members give **only** epistemic (between-member `σ_e² = 1/(M−1) Σ(f_m−μ̂)²`); the
per-member aleatoric term is empty (a point predictor has no opinion about its own noise). Aleatoric
is **backed out** of pooled residuals — hence the head.

### Corrected 1/M accounting (a bug this caught)
The residual about the **ensemble mean** already contains only `σ_e²/M` of epistemic (averaging M
members shrinks it):
- `T(x) = MSR − bias² = σ_a² + σ_e²/M`
- `σ_a² = T − σ_e²/M`  ← subtract **σ_e²/M, not full σ_e²** (earlier draft over-subtracted by
  `(M−1)/M · σ_e²`)
- reported total: `σ_tot² = σ_a² + σ_e² = T + (M−1)/M · σ_e²`

Members are positively correlated → true `Var(μ̂) ∈ [σ_e²/M, σ_e²]`; `A = T − σ_e²/M` is the
conservative (largest-aleatoric) edge, `A = T − σ_e²` the lower edge (the two bracket it). MSR is a
second moment (variance + bias²), not a variance.

*(Plotting note: in EXP5 cell 18 `bias²` is subtracted into `T` but not plotted — on a log axis it
dives to −∞ where the signed bias crosses zero (~83rd pct), a scale artifact, not variance. The bias
structure lives in cell 7.)*

### In-distribution finding
Epistemic **share** of total predictive variance collapses ~0.20 (low flow) → 0.02 (high flow);
aleatoric 0.79 → 0.98. Between-member spread captures almost nothing where flow is largest — the
in-distribution precursor to the OOD **D-blind** failure.

### Five interpretive layers
1. **What the spread samples:** members = an uncalibrated sample from a posterior over functions;
   they diverge where data didn't pin the fit (Fort et al. 2019, loss-landscape).
2. **σ̂_e² ≤ σ_e² (lower bound):** shared architecture/data/forcing → correlated errors → **no
   spread**. The ensemble can't see what all members get wrong together.
3. **Only epistemic is free; aleatoric needs the head** (the σ_a² = T − σ_e²/M argument).
4. **Agreement is ambiguous** (the trap): small σ̂_e² means *either* genuinely low epistemic
   uncertainty *or* shared overconfidence — between-member variance **cannot** distinguish them.
   Disagreement is evidence of uncertainty; agreement is **not** evidence of confidence.
5. **Calibration is the only bridge:** resolving (4) needs an external check against obs
   (coverage/PIT, stratified by regime; Ovadia et al. 2019 — UQ degrades under shift).

### Caveats that decide trustworthiness
- Gaussian NLL is misspecified (heteroscedastic + skewed + heavy-tailed + non-negative) → the t;
  escalate to log-space or CMAL if the t's residual (skew/regime shape) remains.
- **Aleatoric only** — combine with ensemble spread; report what each captures.
- Held-out fitting is mandatory, else σ² is optimistic.
- A frozen head can only reuse features the *mean* objective learned (not necessarily
  error-predictive) → start small MLP, escalate to a separate variance network if it underfits the
  empirical binned conditional residual variance.
- **Per-member vs ensemble VE (open):** prototype on **one** member, validate calibration, then
  decide whether per-member is worth it. Report a **robust** scale (MAD→σ, IQR) alongside SD given
  kurtosis ~300.

---

## 5. Experiment design — OOD probes & findings branches

### Probes — two orthogonal OOD events (2021, both in test)
- **2021 PNW heat dome** — temperature-extreme, precip ≈ 0. Probes the T / PET channel. Forcing was
  well-observed (T well-captured) → closest to a genuine "model can't."
- **Nov 2021 BC atmospheric-river flood** — precipitation-extreme. Probes the P channel. Orographic
  precip likely **under-captured** by gridded P → forcing under-capture is the credible cause.

Orthogonality is the point. Both run through the head + decomposition + calibration.

### IRF scrapped (attribution deferred)
Bayati et al. (2026) WRR Text S.3 shows their "IRF" is a **second trained surrogate model**
(LSTM-IRF, ~46,570 params, MSE to reproduce the value model via a state-dependent linear convolution)
— first-moment distillation, **not** post-hoc XAI, with **no second-moment analog** (σ² is a
residual-distribution property, not a routed/convolved quantity). Dropped the "IRF-on-σ vs IRF-on-μ"
novelty hook. **Reworked spine is IRF-free:** MVE variance head → aleatoric/epistemic decomposition →
calibration/PIT at the two OOD events → the D-blind risk finding. Attribution is **deferred,
evidence-gated**; the reserve candidate (not adopted) is perturbational/gradient temporal sensitivity
(∂μ/∂x, ∂σ²/∂x per forcing channel and lag) — genuine post-hoc XAI, native to both moments — revisited
only if EXP5 + calibration surface a specific P-vs-T-vs-PET question.

### Branches A–D (let the result pick the story)
- **A — both moments hold (calibrated).** Certifies the first two moments but **not** distributional
  shape (skew/tails untested) → motivates Ch4 tail analysis; don't overclaim the head's skill.
- **B — second moment fails by *shape*; σ stays interpretable.** Magnitude right, Gaussian coverage
  fails shape-specifically (under-covers the flood/upper tail) → log-transform → CMAL/UMAL → Ch4.
- **C — second moment fails by *uninterpretability*.** σ flat/muted everywhere (confirmed via
  linear/MLP/empirical-conditional-variance ladder) → the representation doesn't encode a variance
  regime → motivates building UQ into the objective (trained-in / Bayesian).
- **D — first moment fails (mean-regression / covariate misattribution).** σ inherits it
  (σ² = Var + bias²). Its **second-moment signature splits** (the key refinement):
  - **D-inherit** (benign): σ **spikes** where μ regresses; the uncertainty correctly tracks the
    mean's stress. Wrong mean, honest wide interval.
  - **D-blind** (dangerous): σ **fails to inflate** at the extreme — **confidently wrong where it
    matters most**. The aleatoric head is epistemically blind. Here the **ensemble earns its keep**:
    epistemic spread *should* inflate even when aleatoric σ doesn't → tri-part decomposition (μ
    regressed / aleatoric flat / epistemic caught it), motivating the Bayesian step for a concrete,
    demonstrated reason.

**Discriminators:** C vs D-blind = σ's **in-distribution** behavior (flat *everywhere* ⇒ C
representational; works in-dist, flat *only at OOD* ⇒ D-blind epistemic). Mean-regression is **D, not
C** — it inflates σ (Var + bias²), never flattens it. **Loss doesn't change the failure** — it only
changes the attractor (L2→mean, L1→median, pinball→quantile).

### Calibration / diagnostics (acceptance tests)
- Aggregate/central coverage **cannot** distinguish "calibrated" from "shape failure." Use **PIT on
  the full sample** (not the outside-CI subset — ill-posed).
- PIT shape: uniform = calibrated; symmetric ∪/∩ = dispersion (σ-magnitude) error; asymmetric tilt =
  skew or bias (disambiguate via mean residual ≈0 ⇒ skew, ≠0 ⇒ bias). **Stratify by flow regime** —
  regime-localized skew cancels in a pooled PIT.
- Given few OOD samples, **residual skew/kurtosis by flow bin** is more sample-efficient than tail
  coverage; use exceedance-balance (5/5 vs lopsided) on outside-CI points.
- Open sub-decision: **ν global vs input-dependent** — set by the conditional-kurtosis map (flat
  across regimes → global ν; ramps with flow → ν(x) or a CMAL trigger).

---

## 6. OOD evaluation — the two-claims framing

The OOD work carries **two separable claims**; conflating them is the trap.

| | **Claim 1 — tail/OOD CALIBRATION** | **Claim 2 — model ATTRIBUTION** |
|---|---|---|
| Statement | "the predictive distribution under-covers the tail at the OOD extremes" | "the model genuinely under-responds to OOD forcing (regresses to the mean)" |
| About | the **UQ** (is the interval honest?) | the **model** (is it at fault?) |
| Evidence bar | light — coverage/PIT at the events, obs caveat | heavy — must rule out forcing/rating-curve/timing |
| Home | **inside the MVE paper** (risk motivator) | **optional spin-off** (Frame-lineage DL eval) |
| Required for MVE paper? | yes | no |

### Claim 1 — tail calibration (STAYS in the MVE paper)
At the two 2021 extremes the predictive distribution **fails to cover the observed tail**. This is a
statement about the **interval**, not about *why the mean missed*, so it does **not** need the
attribution gauntlet. The arc: tail failure → UQ must be tail-aware → directs the
variance-extraction, the aleatoric/epistemic decomposition, and the risk analysis. Analyses (light,
in-paper): coverage/PIT at the event windows stratified by regime; ensemble-bracketing rate (report
honestly — recall the 13c→13d 80%→36% selection-bias correction); one limitation paragraph on
flood-stage rating-curve uncertainty. **Language stays at "intervals under-cover the observed tail"**
— it must NOT assert "the model regresses to the mean / is at fault" (that's Claim 2 and imports its
burden).

### Claim 2 — model attribution (OPTIONAL spin-off, not required)
Only if we want the mechanistic claim as a result in its own right. Full fairness burden:
- **2A — attribution of the gap:** rule out / size the rivals before claiming model fault — (1)
  forcing error (orographic AR precip under-captured; compare to station met obs); (2) rating-curve /
  obs error (gauges extrapolate at flood stage; the 132 mm/day AR "miss" could be partly bad obs);
  (3) timing/routing (window-max is timing-insensitive; corroborate with obs–sim cross-corr); (4)
  genuine under-response — the claim, credible only after 1–3.
- **2B — hydrological fairness (condition, don't pool):** heat dome — stratify by glacier
  %/elevation/snowpack; AR — by area, response time, antecedent saturation. Quantify per-basin
  OOD-ness vs 1980–2005 training support.
- **2C — literature grounding:** White et al. 2023 (heat dome, have it); **AR/flood reference still
  missing** (CW3E catalog or a Nov-2021 BC floods paper); reconcile with Frame et al. 2022 (§7);
  Bayati et al. 2026 (functional-realism lineage).
- **2D — statistical defensibility:** citation-grounded selection — AR = SW-BC corridor (~31 basins,
  22 precip-confirmed; more faithful to the named event than forcing-only); heat dome = whole domain
  (264/269 >1.5σ Tmax → selection-robust). Shrinkage slope with bootstrap CI + errors-in-variables/
  Deming (obs noise attenuates the slope). Spatial autocorrelation → effective n < n; pooled CIs
  anti-conservative (standing project caveat).

### The preliminary (jump-off) — EXP5 cells 13a–13d
Stay in EXP5 as cited motivation: 13a/13b single-station hydrographs (shape captured, magnitude
pulled to mean); 13c obs-selected (slopes 0.16/0.30, "80% obs>ensemble" — **demonstrably
selection-biased**); 13d forcing-selected (defensible): slopes 0.18 (AR)/0.38 (heat dome) robust,
"obs>ensemble" **36%**, median μ̂/obs ≈ **0.81** — the dramatic numbers were largely selection
artifact. The 13c→13d contrast is itself the "do-it-fairly" lesson.

### Relationship to the fork
Claim 1 (in-paper) is the **risk motivator** feeding the calibration/PIT step that decides the fork:
intervals under-cover the tail AND the ensemble under-covers at the extremes → epistemic/Bayesian
side; a heavier aleatoric shape would cover it → CMAL. Claim 2 (spin-off) is the **mechanistic
backing** for the epistemic story — strengthens the Bayesian motivation but is not required.

---

## 7. The Frame (2022) tension

**Frame et al. 2022 (HESS):** LSTMs *can* predict streamflow extremes exceeding the training-period
max, GIVEN adequate forcing, evaluated on **daily peaks**, EA-LSTM, CAMELS-US. Our events under-cover.
**Resolution:** Frame is about the *mean's capability*; Claim 1 is about the *interval's coverage* —
different objects, no direct collision.

| | AR flood | Heat dome |
|---|---|---|
| Input carried the signal? | Probably **NOT** (orographic precip under-captured) | **YES** (T well-observed) |
| Frame's claim applies? | No — outside his scope (assumes adequate forcing) | Yes — live |
| Most credible cause | Forcing under-capture | Genuine under-response / unrepresented process |
| Closest to "model can't"? | No | **Yes** |

Other reconciling differences: evaluation target (our window-max anomaly vs his daily peak —
apples-to-oranges); EA-LSTM→StandardLSTM swap (untested OOD consequence); domain (CAMELS-US vs
snow/glacier BC-AB). **NOT a difference — the loss:** Frame used the same NSE lineage, so NSE\* can't
be the anti-Frame lever. NSE\* **is** an L2-family loss (σ_b-normalized); within-basin same minimizer
as MSE; cross-basin it *downweights* high-σ_b basins — **more forgiving** there, which *licenses*
shrinkage rather than punishing it. (Gupta et al. 2009 variability-ratio α<1 is generic MSE/NSE
peak-damping, not NSE\*-specific.)

---

## 8. Venue / publication strategy

**Primary target: one of JGR-MLC / HESS / WRR.** The venue tracks the **kind of result**, not the
branch number. Decide the *lead framing* early but let the branch pick it — don't reverse-engineer the
science toward a venue.

**Spine:** method-is-the-star → **JGR-MLC**; UQ finding/breakdown with hydrological-risk implications
→ **HESS** (Klotz lineage); attribution (mis)realism to physical drivers → **WRR** (Bayati lineage).

**Branch → framing:** A → JGR-MLC (method demo). B → **HESS** (UQ-breakdown; strongest HESS branch),
secondary JGR-MLC. C → HESS (cautionary: post-hoc UQ can't be recovered from a mean-trained
representation). D → **WRR** (attribution-realism); D-inherit → WRR + strong JGR-MLC method angle;
**D-blind → HESS via risk** (confidently-wrong-at-the-extreme; best Ch4-risk motivator).

**Bridge tier (archival, ML-flavored):** JGR-MLC (AGU backing; a *substitute* for HESS/WRR, not an
add-on). AIES (AMS) — atmosphere-leaning, out-shadowed. Environmental Data Science (Cambridge) —
weakest. All ~2023-new (backing is brand/society, not yet impact factor) — check WoS/DOAJ before
committing.

**Workshop overlay (non-archival):** value highest when primary is HESS/WRR. ML4PS (NeurIPS) for A/B;
Climate Change AI (NeurIPS/ICLR/ICML) for D/D-blind. Non-archival + substantial journal extension
required. **NeurIPS ~Aug 2026** clashes with writing/immature results (risky); **ICLR ~Feb 2027 /
ICML ~May 2027** after the Dec draft (recommended; CCAI@ICLR'27 preferred). Preprints: ESS Open
Archive (pairs with AGU) or EarthArXiv.

**Timeline:** science over summer 2026; paper writing Sept–Dec 2026.

---

## 9. Literature

### Verified this run (authors/year/venue/DOI confirmed 2026-06-19)
- **Nix & Weigend 1994** (IEEE ICNN, 10.1109/ICNN.1994.374138) — the MVE head.
- **Andrews & Mallows 1974 / West 1987** — scale-mixture origin of the Student-t.
- **Seitzer 2022** (ICLR, arXiv:2203.09168) — β-NLL, Gaussian-NLL mean pathology.
- **Stirn 2023** (AISTATS, PMLR v206 / arXiv:2212.09184) — faithful heteroscedastic regression
  (stop-gradient → provably faithful mean).
- **Sluijterman 2024** (Neurocomputing, 10.1016/j.neucom.2024.127929) — optimal MVE training;
  post-warmup freeze ≈ joint.
- **Amini 2020** (NeurIPS, arXiv:1910.02600) — Deep Evidential Regression (NIG = scale mixture).
- **Meinert 2023** (AAAI, 10.1609/aaai.v37i8.26096) + **Juergens/Meinert 2024** (ICML,
  arXiv:2402.09056) — DER epistemic is a heuristic (not well-identified).
- **Pourkamali-Anaraki 2026** (NCA, 10.1007/s00521-026-12042-x / arXiv:2503.12354) — TDistNN
  (end-to-end Student-t output; head form no longer novel).
- **Huttel 2023** (arXiv:2308.10650, non-archival) — Bayesian evidential quantile regression (maps
  the "enrich the mixing → CMAL/UMAL" prong).

### Core ML-UQ reading list (pulled; not all DOI-verified — see below)
Lakshminarayanan et al. 2017 (deep ensembles + Gaussian head = literally our structure); Fort et al.
2019 (what the spread samples); Hüllermeier & Waegeman 2021 (aleatoric/epistemic split isn't unique —
depends on what you condition on); Ovadia et al. 2019 (calibration under shift); Kendall & Gal 2017
(aleatoric/epistemic via total variance in DL); Wilson & Izmailov 2020 (ensembles ≈ Bayesian
marginalization — the steel-man for treating σ_e² as epistemic; read against layer 2).

**Hydrology UQ lineage:** Klotz et al. 2022 HESS (benchmarks MC-dropout/ensembles/CMAL/UMAL;
https://hess.copernicus.org/articles/26/1673/2022/); Beven & Binley 1992 (GLUE, informal-Bayesian
incumbent); Frame et al. 2022 (LSTMs can hit unseen extremes, §7); Nearing et al. 2021 (ML's role);
Bayati et al. 2026 WRR (functional realism, 10.1029/2025WR040076, UBC EOAS).

**Events:** 2021 PNW heat dome — White et al. 2023, Nature Communications
(https://www.nature.com/articles/s41467-023-36289-3). **AR/Nov-2021 flood reference still missing**
(CW3E catalog or a Nov-2021 BC floods paper).

### Still UNVERIFIED — do not add to the bib until confirmed
Kendall & Gal 2017, Hüllermeier & Waegeman 2021, Lakshminarayanan 2017, Wilson & Izmailov 2020,
Daxberger 2021 (Laplace Redux), Kristiadi 2020 (last-layer), Bishop 1994 (MDN), Klotz 2022 (CMAL/UMAL
DOI), and all GAP-2 hydrology Bayesian-DL/GLUE references.

### Citation-accuracy flags
Takahashi et al. 2018 is a Student-t **VAE** for robust density estimation (IJCAI 2018), NOT a
regression head — cite accordingly. No single canonical "Student-t MVE neural net" landmark exists
the way Nix-Weigend owns Gaussian MVE.

---

## 10. Open questions (lit-review conversation asks)

Live questions for the ML-UQ positioning conversation (I'm strong on hydrology/stats, deliberately
building ML-UQ depth; conservative-by-default on stats; small-n alarms welcome for event-level work):
1. **Enrich the ML-UQ list** — canonical/recent work on post-hoc variance estimation, heteroscedastic
   NLL pathologies, ensembles-as-Bayesian, calibration under shift, heavy-tailed regression heads?
2. **Is "Bayesian DL beyond GLUE for extremes" genuinely under-explored** (GAP 2), or does SWAG /
   Laplace / deep-kernel / last-layer Bayesian already cover it in/near hydrology?
3. **Stop-point:** is a Student-t MVE *result alone* publishable (HESS/WRR vs a shorter AIES
   failure-paper), or does it need the CMAL/UMAL or Bayesian fork to advance past Klotz 2022?
4. **Pressure-test the rejections** — am I dismissing quantile/conformal too fast given its
   OOD-coverage literature? Evidential-regression-epistemic?
5. **Faithful-regression lineage** — is "frozen mean is more conservative" read correctly, and does
   going *beyond* Sluijterman (frozen hidden state too) have precedent or is it genuine novelty?

Also open (implementation sub-decisions): ν global vs input-dependent (§5); per-member vs ensemble VE
target (§4) — prototype one member first.

*(Bibliography deliberately light on the DL side until direction is locked. Don't assume a citation
exists; flag when one is needed.)*

---

## Appendix A — Methods decision log (verbatim, append-only)

> Preserved from `methods.md`. Per project convention, do **not** edit these entries in place — add
> dated revisions. Note the early entries (2026-05 → 2026-06-05) belong to the **EXP4 descriptive
> trend analysis**, a separate thread from the Ch3 MVE work above; kept here so the full trail lives
> in one place.

### Sanity-check basin selection — methods memo
**Purpose.** Select 6–10 basins to build intuition on the Mann-Kendall + Theil-Sen pipeline before
scaling to the full ~100-station dataset. Calibration of the test machinery, not a substantive
analysis.

**Selection criteria.** (1) *Typical-basin filter* — stations within the IQR (25th–75th) of the full
set on each of basin_area_km2, mean_elev, lat, lon (median-attribute basins are representative;
attribute-extreme basins tune intuition to edge cases). (2) *Glaciation contrast* — two clearly
separated groups: non-glaciated (glacier_pct == 0) and glaciated (glacier_pct >= 3%). (3) *Data
availability* — August missingness < 10% on daily Q_obs over 1980–2023, pre-imputation; winter gaps
ignored (analysis aggregates to August).

**What this set is NOT.** Not random, not stratified for inference, not "representative" beyond
attribute-typicality. Diagnostic only.

**What I'll do with the candidates.** Plot each station's August series and visually sort into
clearly-trending / clearly-noisy / ambiguous — cases to check MK fires when it should, doesn't when
it shouldn't, and is sensible in the middle.

### 2026-05-19 — Data units reconciliation
Diagnostic: at 08LB047, obs mean = 153, sim mean = 3.0, ratio ≈ 51. combined_streamflow.csv is in
m³/s (WSC), phase-split sim outputs in mm/day (LSTM). **Decision:** convert obs m³/s → mm/day via
86.4 / basin_area_km2; all downstream in mm/day. **Verification:** τ and p from MK are invariant under
positive monotonic scaling (unchanged); slope magnitudes change by the per-basin factor. **Action:**
re-run the six-basin sanity check in mm/day, confirm τ/p match, then scale to all stations.

### 2026-06-01 — Moving variance ported into EXP4a (obs-only)
Brought the EXP4b rolling-variance computation into EXP4a, **observed streamflow only** (no
sim/ensemble, so no obs-vs-sim compression narrative travels). Windows 10/15/20 yr, centered,
min_periods = window. Units: obs August mean m³/s → mm/day via 86.4 / basin_area_km2 (reuses the
2026-05-19 decision) for area-normalized cross-basin aggregation. Alternative (normalize by own
median², unitless) rejected to stay consistent with the project unit and EXP4b figures.
Deliverables (descriptive-first): per-station variance trajectories (15-yr, by glaciation) and
population-mean variance over time at three windows, glaciated vs non-glaciated. **Deliberately
omitted:** the EXP4b inferential layer (Theil-Sen variance slopes with block-bootstrap CIs,
Wilcoxon/MW-U/HC3 OLS, slope-tail investigation) — deferred under descriptive-first.

### 2026-06-03 — First inferential step: per-station relative slope vs 0, by glaciation class
Transition out of descriptive-first for one targeted question, at the user's direction (confirm with
Tyler/Valentina before it enters the shared narrative). Trigger: Cell 7 shows glaciated-class relative
Theil-Sen slopes below 0; Cell 7b formalizes "is the per-class location < 0?" Framing: individual
series noisy/non-linear, but Theil-Sen robust, decrease identifiable at the glaciation-class level;
non-glaciated expected NOT to support it. **Method:** one-sample Wilcoxon signed-rank vs 0, one-sided
(alternative='less'), exact where scipy allows. Reported alongside: sign test (binomial on fraction
decreasing — no symmetry assumption); percentile bootstrap 95% CI on per-class median slope (B=10000,
seed 42). Run for all four classes incl. Non-glaciated as expected-null. **Alternatives considered:**
one-sample t-test (rejected — normality + small n + skew); Hodges-Lehmann with signed-rank CI (leans
on symmetry; deferred). **Caveat:** stations spatially autocorrelated → effective n < n → all p-values
and the bootstrap CI anti-conservative; reported directional, not calibrated. Honest fix = regional /
spatial-block resampling (not yet implemented). Raw p-values; no multiple-comparison correction.

### 2026-06-03 (addendum) — Cell 7d: MK trend of relative slope vs glacier cover
Second targeted step, raised by 7c: within glaciated basins (glacier_pct > 0), does the per-station
relative Theil-Sen slope trend monotonically with glacier cover? **Test:** Mann-Kendall
(pymannkendall original_test) on rel_slope ordered by glacier_pct — a rank test of monotonic
association (≡ Kendall tau), existence + direction, NOT magnitude. **original_test, NOT Hamed-Rao:**
the AR(1) correction targets serial autocorrelation in a *time* series; here the ordering index is a
covariate, so AR(1) is not the relevant structure and applying it would be a misuse. Spatial
autocorrelation across basins left uncorrected → p directional only. **Confounding flagged:**
glacier_pct co-varies with elevation/area → a significant MK is association with cover, not evidence
glaciation is causal (earth-process read → Tyler). Non-glaciated excluded.

### 2026-06-05 — Cell 7c-modelled: sim relative trend vs glacier cover
Recreated the obs 7c scatter on MODELLED flow. Sim Theil-Sen workflow built in-cell: load the 10
phase-split members (mm/day, dated), average to the daily ensemble-mean series, take the August annual
mean per station-year, then a per-station Theil-Sen slope as %/yr of that station's median August flow
(units cancel → comparable to obs). **Choice:** ensemble MEAN series for the trend — the
conditional-mean trend, right aggregate for a slope (contrast variance, where averaging suppresses
signal → EXP4b used per-member). Full 1980–2022 to mirror obs; NOTE this mixes in-sample (≤2005) and
out-of-sample years — comparability is the reason, the in-sample portion is flagged. **Deferred:** the
obs-vs-sim comparison re-opens the shelved EXP4b compression theme → group/PI decision, not folded
into current-phase output (CLAUDE.md principle 5).

### 2026-06-19 — Scrap surrogate-IRF from the Ch3/MVE plan; defer attribution layer
Trigger: Bayati et al. (2026) WRR Text S.3 — their "IRF" is a SECOND TRAINED SURROGATE MODEL
(LSTM-IRF, ~46,570 params, Adam), fit by MSE to reproduce Q_LSTM-V[t] via a state-dependent linear
convolution of forcings (Eq. 1). Model distillation into a unit-hydrograph / linear-response form,
NOT a post-hoc XAI lens, reconstructing the FIRST MOMENT only. **Decision:** drop surrogate-IRF (and
the "IRF-on-σ vs IRF-on-μ" novelty hook). Two reasons it cannot transfer to σ²: (1) no second-moment
analog of Eq. 1 — Q ≈ a convolution of forcings (unit-hydrograph theory), so a learned impulse kernel
reconstructs it and kernels are interpretable; σ²(x) = E[(y−μ)²|x] is a residual-distribution
property, not routed — imposing the convolution structure gives uninterpretable kernels; (2) it is a
whole separate trained model per moment (heavy + theoretically empty per (1)); and we swapped
EA-LSTM → StandardLSTM, so replicating their Fig-2b architecture would be from-scratch regardless.
**Replacement:** NONE now — attribution DEFERRED and evidence-gated (user direction). Reworked spine
is IRF-free: MVE head → aleatoric/epistemic decomposition (LTV) → calibration/PIT at the two OOD
events → confidently-wrong-at-extreme (D-blind). D-inherit/D-blind needs only σ-at-event vs
in-distribution (head + per-member spread). **Reserve (NOT adopted):** perturbational/gradient
temporal sensitivity (∂μ/∂x, ∂σ²/∂x per channel and lag) — genuine post-hoc XAI, native to both
moments (h.detach() blocks weight-grads at train time only; input-gradients at inference still flow
through the frozen trunk). Revisit only if EXP5 Part 1/2 + calibration surface a specific
P-vs-T-vs-PET question. **Caveat:** scrapping IRF removes the JGR-MLC "transferable IRF diagnostic"
venue angle; venue framing now rests on decomposition/calibration/risk (HESS/WRR).

### 2026-06-19 — MVE variance-extraction method: heteroscedastic Student-t (the fork-point VE)
Decision: the post-hoc variance head (frozen mean, trained off the frozen hidden state) uses a
heteroscedastic STUDENT-T likelihood, r | x ~ t_ν(0, σ(x)) with σ(x) read from h and ν
global-or-input-dependent. This is the "simple MVE first" step; NOT the destination. **Why Student-t
not the alternatives:** a Student-t is a Gaussian SCALE MIXTURE (N(0,s), s ~ Inverse-Gamma,
marginalized; Andrews & Mallows 1974, West 1987) — for a Bayesian a hierarchical prior on the
variance, marginalized, so Bayesian-native not a frequentist device; it sits at the FORK (enrich the
mixing → CMAL/UMAL aleatoric; make the WEIGHTS uncertain → Bayesian DL epistemic) — the simplest
object framing both without entering either; MORE DIAGNOSTIC than Gaussian (Gaussian fails everywhere
— cell 10: standardized-residual skew flips sign by regime, excess kurtosis 78 low → 300 high — so
its failure says nothing about which rabbit hole; a t absorbs symmetric heavy tails, and what REMAINS
is the signal → skew/regime → CMAL/UMAL; scale failing to inflate at OOD where the mean degrades →
Bayesian, the D-blind signature); freezing the mean makes the Sluijterman (2024) / β-NLL (Seitzer
2022) "don't let variance corrupt the mean" machinery MOOT (mean immutable → plain NLL). **Apparatus:**
aleatoric = Student-t head (post-hoc, frozen h); epistemic = 10-member ensemble spread (free, no
head); both read AT heat-dome and AR-flood. Publishable claim is the epistemic one: if ensemble spread
(the current DL-hydrology epistemic proxy, an ensemble descendant of GLUE's informal Bayes) fails to
inflate at extremes where the mean is most wrong, the field-standard epistemic approximation is shown
inadequate for extremes → motivates FORMAL Bayesian DL beyond GLUE. **Rejected (this turn):**
Gaussian MVE (null baseline, fails everywhere, low diagnostic value); QR + conformal / CQR (trades the
distributional assumption for exchangeability that breaks at the OOD events; marginal coverage;
frequentist); CMAL/UMAL (the destination, not the tool — using it AS the VE collapses the framing);
Deep Evidential Regression (t-marginal but epistemic not well-identified — Meinert 2023; may borrow
the t-marginal, not the epistemic); MC-dropout (crude variational epistemic; baseline). **Open
sub-decisions:** (1) ν global vs input-dependent — set by the conditional-kurtosis map (flat → global
ν; ramps → ν(x) or CMAL trigger); (2) per-member vs ensemble VE target (continue.md §6.5) — prototype
per-member on one member first; (3) variance fragile under kurtosis ~300 → report a ROBUST scale
(MAD→σ, IQR) alongside SD. **Citations to add:** Andrews & Mallows 1974 / West 1987 (scale mixtures),
Kendall & Gal 2017 (heteroscedastic aleatoric deep regression), Lakshminarayanan 2017 (deep ensembles
= the epistemic), and for the Bayesian-VE option Daxberger 2021 / Kristiadi 2020 (last-layer /
Laplace). No canonical "Student-t MVE neural net" landmark — flagged for a lit check before the
novelty goes in an abstract.

### 2026-06-19 (addendum) — Literature gap-check results (verified)
Ran a fan-out + adversarially-verified lit check (23 sources fetched, 25 claims 3-vote-verified, 23
confirmed). **HEAD FORM NO LONGER NOVEL:** Pourkamali-Anaraki, TDistNN (NCA 2026,
10.1007/s00521-026-12042-x / arXiv:2503.12354) explicitly swaps Nix-Weigend Gaussian NLL for a
location/scale/df Student-t NLL — BUT end-to-end (not post-hoc/frozen) and no scale-mixture
justification. Novelty must rest on (a) the post-hoc/mean-frozen/residual-trained configuration, (b)
the scale-mixture / hierarchical-variance justification, (c) the hydrology-extremes application, (d)
the CMAL-vs-Bayesian fork. **FROZEN-MEAN DESIGN STRONGLY LITERATURE-BACKED:** Gaussian NLL's
inverse-variance gradient weighting degrades the mean (Seitzer β-NLL, ICLR 2022, arXiv:2203.09168);
cure = decouple variance from mean via stop-gradient → provably faithful mean (Stirn, AISTATS 2023,
PMLR v206 / arXiv:2212.09184); freezing the mean is the limiting case (inherits the guarantee);
Sluijterman 2024 finds post-warmup freeze vs joint no substantial difference → position "mean
immutable" as the conservative ENDPOINT of the faithful-regression lineage. **DER** (Amini, NeurIPS
2020) is the theoretical precedent for the t-marginal via Normal-Inverse-Gamma (= scale mixture); its
single-pass EPISTEMIC claim is refuted as a heuristic (Meinert, AAAI 2023, 10.1609/aaai.v37i8.26096;
Juergens/Meinert, ICML 2024, arXiv:2402.09056 — NIG overparameterized, "epistemic" a convergence-speed
proxy) → our conservative design (frozen mean, NO epistemic claim from the head) is defensible by
explicitly DECLINING DER's contested claim. **NO NAMED POST-HOC FROZEN-MEAN SECOND-MOMENT EXTRACTOR**
in the verified set → the exact configuration is under-occupied (absence-of-evidence caveat). **GAP 2
UNRESOLVED** — do NOT yet put "formal Bayesian DL beyond GLUE for hydrological extremes has not been
done" in an abstract; the run was GAP-1-concentrated and no verified claim covered formal Bayesian DL
(Laplace/last-layer/VI/MCMC/SWAG) for rainfall-runoff extremes vs GLUE; needs a dedicated
hydrology-venue follow-up. **Citation flags:** Takahashi et al. 2018 is a Student-t VAE (IJCAI 2018),
NOT a regression head; Huttel et al. (arXiv:2308.10650, 2023) is non-archival but maps the
"enrich the mixing → CMAL/UMAL" prong. **VERIFIED refs to add:** Amini 2020; Meinert 2023;
Juergens/Meinert 2024; Seitzer 2022; Stirn 2023; Sluijterman 2024; Pourkamali-Anaraki 2026; Nix &
Weigend 1994; Andrews & Mallows 1974 + West 1987; Huttel 2023. **STILL UNVERIFIED (do not add):**
Kendall & Gal 2017, Hüllermeier & Waegeman 2021, Lakshminarayanan 2017, Wilson & Izmailov 2020,
Daxberger 2021, Kristiadi 2020, Bishop 1994, Klotz 2022, and all GAP-2 hydrology Bayesian-DL/GLUE refs.

### 2026-06-19 (scope split) — OOD "bridge" pulled out of EXP5 into its own analysis
Decision: the OOD mean-behavior work (EXP5 cells 13a–13d) promoted to a STANDALONE first-moment
analysis; EXP5 keeps 13a–13d only as the cited jump-off. Rationale: making the model evaluation FAIR
at extremes (separating genuine under-response from forcing / rating-curve / timing error) + the
hydrological + literature grounding is a research thread in its own right, and it is the Branch-D GATE
for the UQ work. Carried result: robust anomaly-shrinkage (slope ~0.18–0.38, forcing-selected) +
moderate epistemic under-coverage (~36% obs above ensemble) — both pending fair attribution before any
claim. *(Superseded by the rescope below — kept for the trail.)*

### 2026-06-19 (rescope) — OOD work: tail-CALIBRATION stays in the MVE paper; attribution is an OPTIONAL spin-off
Supersedes the (scope split) entry — that over-split. The OOD piece carries TWO separable claims:
**CLAIM 1 — tail/OOD CALIBRATION** (a property of the UQ): "the predictive intervals under-cover the
tail at the OOD extremes." Native UQ/risk content, the paper's risk motivator; does NOT need the
forcing/obs/timing attribution gauntlet (it tests the INTERVAL's coverage, not why the mean missed) →
STAYS IN the MVE paper. **CLAIM 2 — model ATTRIBUTION** (a property of the model): "the model genuinely
under-responds to OOD forcing (regression to the mean)." Frame-lineage DL-evaluation result carrying
the full fairness burden (forcing / rating-curve / timing) → SEPARABLE as an OPTIONAL spin-off, NOT
required for the MVE paper. **The trap:** do not let the MVE paper ASSERT model-failure/regression
using these events while skipping attribution — that imports Claim 2's burden into Claim 1's section.
Keep MVE-paper OOD language to CALIBRATION ("intervals under-cover the observed tail"), with a
flood-stage rating-curve caveat; defer "why the mean missed" explicitly.

# chapter3.md — updates 2026-07-13

**How to use this file.** These are splice-in blocks, formatted to respect your append-only
convention rather than rewrite the consolidated body in place. Block A **appends** to Appendix A
(do not touch existing entries). Blocks B–D are dated addenda / a refreshed reference list for the
living sections (§9, §2.5, §8). I chose an update pack over regenerating the full ~8k-word file
deliberately — to avoid transcription drift in your precise technical wording (the 1/M accounting,
the kurtosis figures, the decomposition algebra). Say the word if you'd rather I emit a single fully
merged chapter3.md instead.

---

## BLOCK A — append to Appendix A (append-only log)

### 2026-07-13 — GAP-2 resolved (verified hydrology-venue lit check); aleatoric head pivot; workshop plan; Frame correction

**GAP-2 verdict (resolves the 2026-06-19 addendum's "GAP-2 UNRESOLVED").** Ran the dedicated
hydrology-venue check the earlier GAP-1-concentrated run deferred. Verdict:

- **BROAD claim is FALSE — do not assert "formal Bayesian DL for streamflow is undone."** Formal
  variational inference over LSTM weights already exists in hydrology: **Li et al. 2021** (WRR
  57, 10.1029/2021WR029772) implement stochastic VI over LSTM weights as a residual-error model on
  process-based models (2 Chinese catchments, no OOD/extremes); **Li et al. 2022** (VB-LSTM, J.
  Hydrol., PII S0022169421012713) use variational-Bayes LSTM as a multi-model ensemble combiner
  (beats BMA; no extremes focus). A reviewer will produce both immediately.
- **NARROW gap is REAL and under-explored:** the *conjunction* of (post-hoc, mean-frozen config) ×
  (hydrological OOD extremes) × (explicit ensemble-spread-vs-formal-weights epistemic-honesty
  comparison). No verified paper occupies that triple. Confidence: high the broad claim is false;
  moderate-high the narrow gap is open. **Caveat:** did not exhaustively sweep 2024–2026
  preprints (EarthArXiv/EGUsphere) — "open" = *not contradicted*, not *proven empty*.
- **Structural finding for framing:** formal MCMC/HMC in hydrology targets *low-dimensional physical
  parameters* (e.g. DREAM/Vrugt; HMC + stochastic-rain-model, HESS 27:2935, 2023), NOT NN weights —
  weight-space MCMC is infeasible at LSTM scale. That seam is *why* DL-hydrology uses
  dropout/ensembles/mixture heads instead of formal weight posteriors; it is a methodological reason,
  not an oversight, and it is the bridge this chapter sits on.
- **Klotz 2022 already reports the D-blind precursor:** UMAL and MC-dropout are overconfident /
  too-narrow at high flow. Position our finding as extending that in-benchmark observation to a
  formal ensemble-vs-weights statement at named OOD events.

**Aleatoric head: PIVOT from Student-t → CMAL/UMAL or log-scale (skew is first-order).** Supersedes
the 2026-06-19 "heteroscedastic Student-t as the fork-point VE" decision *for the aleatoric-first
step*. Rationale: standardized-residual skew is first-order (not merely heavy tails); a symmetric
Student-t cannot hold regime-dependent skew → this is Branch B firing, i.e. following the plan, not
abandoning it. Sub-decisions recorded:
- **Log ≠ clean with a frozen physical-space mean.** Residuals `r = y − μ̂` are signed (can't log);
  modelling `log y | x` has location `E[log y|x]`, which by Jensen ≠ `log μ̂`. A log-space likelihood
  centred on `log μ̂` is biased; un-biasing reintroduces a fitted location (breaks "frozen") or needs
  a smearing correction to defend. **CMAL/single-ALD models skew natively around frozen μ̂ in
  physical space** — no transform, no Jensen gap. On frozen-mean compatibility the native asymmetric
  head is strictly cleaner than log → prefer CMAL/ALD as the *first* move (reverses the "log first"
  ordering that was casually listed).
- **CMAL mean-freezing subtlety:** a K-component mixture has K locations → aggregate mean is
  emergent, so a full mixture can quietly un-freeze the mean. A **single asymmetric Laplacian**
  (location = frozen μ̂, fit scale + asymmetry τ off `h`) preserves the clean frozen-mean story and
  still delivers skew. Verify frozen-mean mechanics per family (UMAL differs: sampled τ,
  quantile-regression basis).
- **Novelty after pivot:** loses the scale-mixture / hierarchical-variance justification (t-specific)
  and the "simplest object framing both forks" device. Survives: (a) the post-hoc / mean-frozen /
  residual-trained *config* (Klotz did CMAL/UMAL end-to-end, not this); (b) using the Klotz-best
  skew-aware aleatoric head *hardens* the epistemic claim — aleatoric flatness at OOD can no longer
  be dismissed as head misspecification. Epistemic fork untouched.

**Workshop (D-blind slice): linear Gaussian head is acceptable.** For the 4-page workshop paper only,
fit a linear (escalate to small MLP if it underfits) Gaussian-NLL head off frozen `h`. Justification:
D-blind is a *moment-level* claim, and the Gaussian NLL σ² optimum estimates
`E[(y−μ̂)²|x] = Var + bias²`, which is **distribution-free** — skew misspecification corrupts
PIT/coverage, not the moment. Constraint: **make no PIT/coverage claim with the Gaussian head**
(that would invite "your head isn't skew-aware"; save calibration for the CMAL/log HESS paper). The
headline (ensemble spread σ_e², head-free) is robust regardless. The skew-aware head is a HESS
concern, not a workshop blocker.

**Workshop D-blind — analysis order (recorded so the 4-pager turns around fast once a venue confirms):**
(1) in-distribution competence check = the load-bearing control: show linear-head σ tracks the
empirical binned conditional residual variance in-dist → separates Branch C (flat everywhere =
representation/head too weak) from D-blind (works in-dist, fails only at OOD); (2) confirm heat-dome
mean regression (have: μ̂/obs ≈ 0.81, forcing-selected, robust); (3) read μ̂ shrinkage, σ_e
(head-free), σ_a (head) at event vs in-dist — D-blind = μ̂ regresses AND σ_e narrow AND σ_a fails to
inflate; (4) clincher figure = standardized `z = (y−μ̂)/σ_total` at events (huge |z|) vs well-behaved
in-dist z. **Mechanism to state explicitly:** head trains on ~unbiased in-dist residuals → learns
σ ≈ Var, no bias² term; at OOD reads a frozen `h` that may not encode the novelty → structurally
cannot anticipate OOD bias² (aleatoric blindness expected by construction). Non-obvious ML-relevant
finding = *ensemble* blindness (independently-init members agree where all wrong) → empirical
rebuttal to "deep ensembles ≈ Bayesian marginalization." Lead with that.

**Station constraint (drives event usage):** heat dome = all 269 (selection-robust, 264/269 >1.5σ
Tmax) → the load-bearing event, anchors the workshop; AR flood = ~70 spatially-blocked stations
(SW-BC corridor, orographic precip under-captured) → statistically weaker + causally ambiguous +
carries the attribution burden → **defer to HESS**, do not anchor the workshop on it.

**Venue / workshop strategy (supersedes the §8 timeline's "HESS by Dec" pressure framing):**
- **HESS has no submission deadline** — a January submission costs nothing but coursework overlap.
  The real constraint is personal bandwidth: draft *before* the 4–5-course winter term. Treat
  "done before January" as a strong preference protected by starting to write in September (off the
  August framing pass), NOT by compressing the end. Failure mode to avoid = rushing to beat a
  self-imposed date and drawing major revisions.
- **NeurIPS'26 workshop (ML4PS/CCAI)** is the only fixed external date (contributed-paper deadline
  ~late-Aug/early-Sept; notifications before Sept 29; conference Sydney Dec 6–12). Workshop slate
  being confirmed as of 2026-07-13; ML4PS CFP not yet posted. **Opportunistic shot:** submit the
  4-page D-blind (heat-dome) abstract in August *iff* the result is clean — it is a subset of
  analysis being done anyway, and writing it crystallizes the HESS framing early. **Decouple
  acceptance from attendance:** the accepted paper is the NSERC-wrap deliverable; present
  remotely / at the Paris–Atlanta satellite / via co-author rather than flying to Sydney into the
  HESS-writing window. MSc already secured (UBC geophysics), so the workshop is pure
  enrichment/exposure — strictly droppable behind the paper.
- **Fallback:** a spring-2027 workshop (e.g. CCAI@ICLR'27) off the finished draft — **but ICLR'27
  location is UNVERIFIED**; the "west-coast US, easy/cheap" assumption is not confirmed (ICLR 2026
  was Rio; one low-quality source hints ICLR'27 = Singapore). Check iclr.cc before relying on it.
- ML exposure comes from the workshop; keep the big paper at **HESS** (D-blind/risk story, Klotz
  lineage). JGR-MLC only if the result turns method-dominant (Branch A), which the CMAL/log pivot
  makes less likely.

**Frame (2022) characterization — correction for §7.** Frame is **not an ML skeptic**; co-authors are
the pro-LSTM neural-hydrology camp (Kratzert/Klotz/Gauch/Nearing/Gupta), and the paper is a *rebuttal*
to skeptics: it opens by naming "a concern among hydrologists" that DL can't extrapolate, then
largely refutes it, and finds that adding mass-balance constraints (MC-LSTM) *hurt* at extremes
(pure ML > physics-constrained ML — the opposite of skepticism). What is true and usable: his
headline extrapolation metric is the absolute percent bias of the largest annual peak-flow event,
binned by return period, across ~498 CONUS/CAMELS basins — an aggregate **point-prediction** skill
metric with **no uncertainty/coverage evaluation**. §7 reconciliation therefore stands on firmer
ground: Frame's object = the *mean's* capability; our Claim 1 object = the *interval's* coverage —
different objects, no collision. Also note his "extremes" = magnitude/return-period extremes within
the record; our 2021 events = regime-novelty (heat-dome PET, orographic-AR precip) — same word,
different extrapolation axis. Present Frame as the optimistic pole of a live debate (2025 HESS
bounded-ceiling paper is the counterweight), not as settled fact.

---

## BLOCK B — replace §9 "Literature" with the refreshed version below

### 9. Literature

#### Verified (authors/year/venue/DOI or arXiv confirmed 2026-06 → 2026-07)

*Faithful-regression & MVE lineage (unchanged, previously verified)*
- Nix & Weigend 1994, IEEE ICNN, 10.1109/ICNN.1994.374138 — MVE head.
- Andrews & Mallows 1974 / West 1987 — scale-mixture origin of the Student-t.
- Seitzer 2022, ICLR, arXiv:2203.09168 — β-NLL, Gaussian-NLL mean pathology.
- Stirn 2023, AISTATS, PMLR v206 / arXiv:2212.09184 — faithful heteroscedastic regression.
- Sluijterman 2024, Neurocomputing, 10.1016/j.neucom.2024.127929 — post-warmup freeze ≈ joint.
- Amini 2020, NeurIPS, arXiv:1910.02600 — Deep Evidential Regression (NIG = scale mixture).
- Meinert 2023, AAAI, 10.1609/aaai.v37i8.26096 + Juergens/Meinert 2024, ICML, arXiv:2402.09056 — DER epistemic is a heuristic.
- Pourkamali-Anaraki 2026, NCA, 10.1007/s00521-026-12042-x / arXiv:2503.12354 — TDistNN (head form no longer novel).
- Huttel 2023, arXiv:2308.10650 (non-archival) — Bayesian evidential quantile regression.

*Hydrology UQ lineage (GLUE → formal Bayes → Bayesian DL) — NEW this run*
- Beven & Binley 1992, Hydrol. Process. 6:279–298 (GLUE; DOI 10.1002/hyp.3360060305, standard — not re-fetched).
- Vrugt et al. 2009, SERRA 23, 10.1007/s00477-008-0274-y — DREAM (formal MCMC) vs GLUE debate.
- Nott et al. 2012, WRR, 10.1029/2011WR011128 — GLUE ≈ Approximate Bayesian Computation.
- Jin et al. 2010, J. Hydrol. 383(3–4):147–155, 10.1016/j.jhydrol.2009.12.028 — GLUE vs formal Bayes, conceptual model.
- HMC + stochastic-rain-model 2023, HESS 27:2935 — formal HMC on *physical parameters* (not NN weights).
- Li et al. 2021, WRR 57, 10.1029/2021WR029772 — **formal SVI over LSTM weights** (residual-error; 2 catchments; no OOD).
- Li et al. 2022, J. Hydrol., PII S0022169421012713 — VB-LSTM, **formal VI** (ensemble combiner; no extremes); *DOI 10.1016/j.jhydrol.2021.127046 to confirm*.
- Klotz et al. 2022, HESS 26:1673–1693, 10.5194/hess-26-1673-2022 — CMAL/UMAL + MC-dropout; overconfident at high flow.
- Frame et al. 2022, HESS 26:3377–3392, 10.5194/hess-26-3377-2022 — mean extrapolation (has corrigendum); pro-LSTM, point-prediction metric, no UQ.
- "Unveiling the limits of deep learning in hydrological extrapolation" 2025, HESS 29:5871 — bounded-output ceiling counterpoint; *authors to confirm*.
- Nearing et al. 2021, WRR 57 e2020WR028091, 10.1029/2020WR028091 — role of hydro science + ML.
- Nearing et al. 2024, Nature 627:559–563, 10.1038/s41586-024-07145-1 — global extreme-flood prediction (extremes, not UQ-focused).
- Willard et al. 2025, JGR-MLC, 10.1029/2025JH000732 — ML ensembles + proper scoring rules for UQ.

*General ML-UQ (importable; NO streamflow application found — flag as import, not hydrology precedent)*
- Lakshminarayanan et al. 2017, NeurIPS, arXiv:1612.01474 (10.48550/ARXIV.1612.01474) — deep ensembles.
- Fort et al. 2019, arXiv:1912.02757 — ensembles loss-landscape (what the spread samples).
- Maddox et al. 2019, NeurIPS, arXiv:1902.02476 — SWAG (no hydro use found).
- Daxberger et al. 2021, NeurIPS 34:20089, arXiv:2106.14806 — Laplace Redux (no hydro use found).
- Hüllermeier & Waegeman 2021, Mach. Learn. 110(3):457–506, 10.1007/s10994-021-05946-3 — aleatoric/epistemic split.
- Ovadia et al. 2019, arXiv:1906.02530 (10.48550/arXiv.1906.02530) — UQ degrades under shift.

*Events*
- White et al. 2023, Nat. Commun. 14:727, 10.1038/s41467-023-36289-3 — 2021 PNW heat dome. **AR / Nov-2021 BC flood reference still missing** (open fork: meteorological AR-characterization source for Claim 2A vs flood-impact source vs CW3E AR-intensity catalog).

*Attribution / functional-realism lineage*
- Bayati et al. 2026, WRR, 10.1029/2025WR040076 (UBC EOAS) — functional realism (surrogate-IRF scrapped from Ch3, see Appendix A 2026-06-19).
- Kirchner 2024, HESS 28(19):4427–4454, 10.5194/hess-28-4427-2024 — ERRA (ensemble rainfall-runoff analysis).

#### Still UNVERIFIED — do not add as fact until confirmed
Kendall & Gal 2017; Wilson & Izmailov 2020 (arXiv:2002.08791 — the "ensembles ≈ Bayesian
marginalization" claim our D-blind result argues against; verify before citing); Kristiadi 2020
(last-layer Laplace, arXiv:2002.10118); Izmailov 2021 (HMC for NNs, arXiv:2104.14421); Gal &
Ghahramani 2016 (MC-dropout, arXiv:1506.02142); Bishop 1994 (MDN); Ghobadi & Kang 2022 (Water
14:3672 — confirm whether *formal* Bayesian or dropout-flavoured before using as a formal-DL precedent).

#### Did not find (flag as absence-of-evidence, not confirmed-absent)
Deep kernel learning and explicit functional-space priors applied to rainfall-runoff / streamflow.

#### Citation-accuracy flags (unchanged)
Takahashi et al. 2018 is a Student-t **VAE** (IJCAI 2018), NOT a regression head. No canonical
"Student-t MVE neural net" landmark exists the way Nix–Weigend owns Gaussian MVE.

---

## BLOCK C — append to §2.5 (Novelty positioning) as a dated addendum

> **Update 2026-07-13 (post GAP-2 hydrology-venue check + aleatoric pivot).** GAP-2 is resolved (see
> Appendix A 2026-07-13): the *broad* "formal Bayesian DL for streamflow is undone" claim is FALSE
> (Li et al. 2021 WRR; Li et al. 2022 J. Hydrol. are formal VI over LSTM weights) — do not assert it;
> the *narrow* gap (post-hoc mean-frozen config × hydrological OOD extremes × ensemble-vs-formal
> epistemic-honesty comparison) is real and under-explored (confidence moderate-high; 2024–26
> preprints not exhaustively swept). Aleatoric head pivots Student-t → CMAL/single-ALD (skew is
> first-order = Branch B; log rejected for frozen-mean/Jensen incompatibility). Novelty now rests on
> (a) the post-hoc/mean-frozen/residual-trained config, (b) the CMAL-best skew-aware head *hardening*
> the epistemic claim, (c) hydrology-extremes application, (d) the ensemble-vs-formal fork — NOT on
> the (now-dropped) Student-t scale-mixture justification.

---

## BLOCK D — append to §8 (Venue strategy) as a dated addendum

> **Update 2026-07-13.** HESS has **no submission deadline** → "by Christmas" is a self-imposed
> bandwidth goal (draft before the 4–5-course winter), not a wall; a January submission is fine, and
> rushing into major revisions is the real cost to avoid. Workshop route (ML exposure, NSERC wrap,
> MSc already secured so purely additive): NeurIPS'26 ML4PS/CCAI, contributed-paper deadline
> ~late-Aug/early-Sept, Sydney Dec 6–12 — submit the heat-dome D-blind 4-pager in August *iff* the
> result is clean, and **decouple acceptance from attendance** (present remote/satellite/co-author,
> keep the December week for the paper). Fallback = spring-2027 workshop (CCAI@ICLR'27) off the
> finished draft — **ICLR'27 location unverified; do not assume west-coast US / cheap** (ICLR 2026 =
> Rio; check iclr.cc). Big paper stays HESS unless the result turns method-dominant (→ JGR-MLC).