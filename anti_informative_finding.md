# Research memo — non-informative epistemic collapse under shift (SHELVED, develop later)

**Date written:** July 16, 2026
**Status:** Shelved. Operational version → HESS/CCAI now. Foundational version → develop with a methods co-supervisor (Pleiss / Sutherland), target EIML/AABI later.
**Why this note exists:** the foundational claim is potentially strong but not yet defensible. This captures *what I saw and why I thought it mattered* so the reasoning doesn't go cold before I have the firepower to make the claim rigorously. Do not let this decay into "I vaguely remember an interesting result."

---

## 1. The observation (state precisely; refine as understanding sharpens)

On a frozen mean streamflow predictor + deep ensemble, with a post-hoc variance head (Gaussian / asymmetric-Laplace tails), decomposing predictive uncertainty into aleatoric + epistemic via law-of-total-variance:

- Under the **2021 AB/BC heat dome** (a genuine covariate/distribution shift), the **ensemble-derived epistemic signal went NON-INFORMATIVE** — it stopped carrying information about the model's epistemic state / correctness exactly in the OOD regime where it is most relied upon.

**Precise quantity + measurement (FILL IN / SHARPEN — this is the crux):**
- What exactly is the quantity? (ensemble disagreement / LTV epistemic term / MI-style term / variance of the variance-head across ensemble members?)
- What does "non-informative" mean operationally? Candidate definitions to pin down:
  - variance/disagreement *decouples from error* (calibration curve flattens; corr(uncertainty, error) → 0), OR
  - epistemic term collapses toward ~0 even as error rises, OR
  - variance stays ~constant in- vs out-of-distribution (uninformative rather than misleading).
- Pick ONE, define it, measure it. Right now this is the weakest-specified part and the first thing to fix.

## 2. The novelty claim (the reason this is more than trivial)

- **Trivial / known (CITE, don't re-prove):** deep ensembles are a crude, *non-Bayesian* θ-sample (SGD modes of a non-convex loss, not posterior samples). Lakshminarayanan et al. 2017 §2; Wilson & Izmailov; Fort et al. (mode connectivity). NOT my contribution.
- **The actual claim:** my failure mode appears **distinct from Wilson–Izmailov "Dangers of BMA under covariate shift."** W-I characterize *confidently wrong* (low predictive variance + high error). Mine is *non-informative* (signal decouples from epistemic state) — arguably a **different failure regime**, not an instance of theirs.
- **Contribution framing = extend the taxonomy, NOT refute W-I.** "W-I characterize confident-wrongness; I document a distinct non-informative regime, and when you get which." Collegial + defensible. Do NOT frame as "W-I incomplete/wrong" (hostile-reviewer bait; W-I's people review EIML/AABI).
- **Weaker-but-safe fallback claim:** "under these conditions the failure manifests as non-informativeness rather than confident-wrongness — here's the regime map." Still novel, easier to defend. Know which claim the evidence actually supports and claim exactly that.

## 3. Why it's SHELVED (what must be nailed before firing at EIML/AABI)

1. **Precise, operational definition of "non-informative"** that is measurably NOT "confidently wrong" (see §1). One sentence, one metric.
2. **Evidence ruling out the W-I mechanism** — show it's decoupling, not just variance-collapse-in-disguise. The variance-head-across-every-ensemble sweep is likely the vehicle: does non-informativeness persist across configs in a way characteristically different from confident-wrongness?
3. **Resolve the decomposition-artifact vulnerability (DEEPEST RISK).** The LTV aleatoric/epistemic split is Hüllermeier-lineage, and Hüllermeier himself critiques whether that decomposition cleanly identifies epistemic uncertainty. A reviewer can say: "your epistemic estimate is non-informative because the decomposition doesn't identify epistemic uncertainty in the first place — measurement artifact, not ensemble failure." Must show the non-informativeness is a property of the ENSEMBLE, not of the chosen decomposition.

## 4. Who to develop it with, and why

- **Pleiss** — reliable DL / calibration / GP-Bayesian. Owns the "is the uncertainty informative/calibrated/decoupled-from-error under shift" question. Co-author name adds credibility to a foundational claim vs W-I. (Also prospective MSc co-sup → double win.)
- **Sutherland** — kernel two-sample testing / distinguishing distributions = the machinery to turn "signal goes non-informative" into a *principled* statement about detectability, and to rule out the decomposition-artifact objection rigorously. Worth a research conversation as a methods collaborator even without taking her course.
- Bring it to them as a live open problem ("I think I have a failure mode distinct from W-I; help me make it defensible") — good research-taste signal + relationship-building.

## 5. Staging (how this result travels)

- **NOW — operational version → HESS (journal) + CCAI/TCCML (workshop):** the concrete in-domain failure + full ensemble treatment + variance-head sweep. Carries on *operational consequence* alone (hydrology relies on ensemble spread naively — my PI's own reaction is proof it's news to the applied field). Does NOT require winning the W-I taxonomy fight. In HESS: CITE the crude-θ mechanism (Lakshminarayanan/W-I), DEMONSTRATE the operational failure in-paper (it's load-bearing → must be shown, not outsourced to a workshop citation). Motivates the Bayesian-DL direction = thesis groundwork.
- **LATER — foundational version → EIML/AABI, with Pleiss/Sutherland:** the W-I-distinct-failure-mode claim, fired only once §3 (1–3) are airtight and a methods co-author is attached. Higher ceiling, PhD-community identity-building. This is the loaded chamber I'm deliberately not firing until it's aimed.

## 6. To preserve (so "later" ≠ "re-derive from scratch")

- [ ] Exact experiments/configs that produced the non-informativeness, versioned + runnable.
- [ ] The variance-head-across-every-ensemble sweep data.
- [ ] Plots showing the decoupling/collapse, with the heat-dome window marked.
- [ ] This memo, updated whenever understanding sharpens (esp. §1 definition).

## 7. One-line reminder to future me

I am NOT claiming a novel mechanism (ensembles-are-crude is textbook). I AM claiming a demonstrated operational consequence (now, applied venues) and — pending §3 — a failure regime distinct from Wilson–Izmailov (later, foundational venues, with a methods co-author). Cite the mechanism; prove the consequence; extend the taxonomy, don't refute it.