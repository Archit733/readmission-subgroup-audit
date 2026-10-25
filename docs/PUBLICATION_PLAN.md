# Publication assessment and plan

> Written on 2026-09-29, before the research extension. The extension steps it recommends were carried out in
> [`research/`](../research/); the resulting paper is [`research/paper/preprint.pdf`](../research/paper/preprint.pdf).

## Verdict

As it stands, not suitable for a main-track journal or conference. With about 2 to 3 months of extension, a workshop
or Findings-track paper is realistic, and a full conference paper is possible.

## Why it is not enough yet

| Gap | Reviewer objection |
|---|---|
| Novelty | Age and subgroup performance gaps in readmission models are well known. Bootstrap CIs and patient-level splits are good practice, not a new method. |
| One dataset | UCI Diabetes-130 covers 1999-2008, diabetic patients only, and is heavily used. |
| No external validation | The gap is not shown to appear anywhere else. |
| One seeded split | No variance across splits. |
| Unexplained cause | Two mechanisms were tested and both failed. Honest, but the paper has a finding without an explanation. |

## What is publishable

The strength is rigor, not the headline result:

1. Leakage quantified: 0.688 to 0.676 from a patient-level split.
2. A proxy-based mechanism (r = -0.80 over 9 points) retracted after direct testing.
3. The subgroup gap shown at the operating point: 64% vs 23% of readmissions caught.
4. A monitoring result: subgroup alerts lag, and small subgroups lack the sample size to alert at all.

That framing suits negative-result and reproducibility venues.

## Venues

| Tier | Needs | Venue |
|---|---|---|
| Now | Public repo, 4-page write-up | ML4H Findings track (non-archival; accepts negative results and reproducibility studies); health-ML or fairness workshops at major ML conferences; arXiv preprint |
| After extension | Second and third dataset (MIMIC-IV, eICU; both need PhysioNet credentialing), repeated splits, mitigation experiments (reweighting, group-specific thresholds) | CHIL; ML4H proceedings (Impact and Society, or Applications and Practice); MLHC |
| Journals | Extended version | JAMIA Open, Journal of Biomedical Informatics, BMC Medical Informatics and Decision Making, PLOS Digital Health. npj Digital Medicine would be a stretch without clinical validation. |

## Deadlines checked (as of 2026-09-29)

* ML4H 2026 (Sydney, Dec 6-7): proceedings deadline was Sept 10 and has passed. A separate Findings track exists
  (non-archival, 4 pages); whether it has a later deadline was not confirmed. Check the call for papers.
* CHIL: the call page reachable was the 2025 cycle (early-February deadline, 8-10 pages, three tracks). 2027 dates
  were not confirmed.
* MLHC and journal timelines were not checked.

Sources: https://ml4h.ahli.cc/submit/call-for-papers/ , https://chil.ahli.cc/submit/call-for-papers/ ,
https://mlhc.org/paper-submission

## Extension steps, in order

1. Repeat the audit on MIMIC-IV. If the age gap appears in a second, modern dataset, "one old dataset" becomes a finding.
2. Add repeated splits (for example 20 seeds) so intervals also reflect split variance.
3. Try to close the gap. Either outcome is reportable.
4. Keep the retracted-mechanism story.
