# 5-minute presenter script

About 140 words per minute, roughly 700 words. Text in [brackets] is a slide cue.

**[Slide 1: Title. 0:00 to 0:30]**
Hi everyone. My project asks a simple question. When a hospital model says it's 68% accurate, is it 68% accurate for
everyone? I'll show you it isn't, how I proved that, and how I turned it into something a hospital team could
actually run.

**[Slide 2: The problem. 0:30 to 1:15]**
Hospitals want to predict which patients will be readmitted within 30 days, so they can send follow-up care to the
people who need it. These models are judged with one number, the AUC. Mine scores 0.676. But a single number is an
average. If the model works well on some patients and fails on others, the average hides it. And in healthcare, the
group where it fails is the group that gets missed. So I treated the audit as the product. The model is
deliberately ordinary.

**[Slide 3: Data and method. 1:15 to 2:15]**
I used a public dataset of about 102,000 hospital encounters from 71,000 patients. Three rules guided the method.
First, no leakage. About 30% of encounters come from repeat patients. If the same person appears in both training
and testing, the score is inflated. I split by patient instead, and the AUC fell from 0.688 to 0.676. That lower
number is the honest one. Second, every subgroup claim needs uncertainty. I use bootstrap confidence intervals. I
only call something a real gap if the intervals don't overlap. And if a group is too small, I say "inconclusive",
never "no gap". Third, I check my own math. Every subgroup score is recomputed with a separate library, Fairlearn,
and the build fails if they disagree.

**[Slide 4: The finding, AUC by age. 2:15 to 3:15]**
Here is the result. For patients aged 20 to 30, AUC is 0.776. For patients aged 90 to 100, it is 0.550, barely
better than a coin flip. The confidence intervals don't overlap, so this is a real gap. Gender and race show no
meaningful gap between the large groups.
[Next slide: recall chart]
Here is why it matters in practice. Suppose the hospital flags the top 20% highest-risk patients. For the youngest
group, that catches 64% of actual readmissions. For the oldest group, it catches 23%, and random flagging would
catch 20%. So for the oldest patients, the flag is nearly useless.

**[Slide 5: What I got wrong. 3:15 to 3:55]**
I want to be honest about one thing. I first thought I'd found the cause: patients with more diagnoses. That
correlation was strong, but it was based on nine points and it went through age as a proxy. So I tested it
directly, with finer bins, and then with the Charlson comorbidity index. Neither shows a clean pattern. I retracted
the claim. The cause of the age gap is still unexplained, and my report says so.

**[Slide 6: Making it usable. 3:55 to 4:40]**
Then I built what a hospital would need around the audit. A prediction service where every score comes with a
reliability note. If you score a 92-year-old, it tells you this model is less reliable for that age band. A
dashboard to explore all of this. And monitoring. On a simulated data stream, input drift is caught first, and
performance alerts come later, because real readmission outcomes arrive 30 days late. It has 92 automated tests and
an auto-generated model card.

**[Slide 7: Limits and close. 4:40 to 5:00]**
The limits: the data ends in 2008, there's no external validation, and this is a research tool, not for clinical
use. The takeaway is simple. Before trusting a model's average, ask who it fails, and prove it with uncertainty.
Thank you. Happy to take questions.

## Delivery notes

* If you run long, cut "Gender and race show no meaningful gap between the large groups" and shorten the
  "Second, every subgroup claim..." detail to one line.
* Say 0.776 and 0.550, then pause for a beat. 64% vs 23% is the line people remember.
* Do not skip slide 5. The retracted claim is the most credible moment in the talk.
* Figures: slide 4 uses `reports/figures/fig1_auc_by_age.png` and `fig2_recall_at_top20.png`; slide 6 can use a
  screenshot of the dashboard's "Score a patient" tab and `fig4_monitoring_simulation.png`.

## Likely questions

* **Why not fix the old-patient gap?** The cause is unexplained. Two candidate mechanisms were tested and both were
  negative, so any fix would be a guess. Measuring and flagging comes first.
* **Is 0.55 vs 0.78 just noise?** No. Both bands have hundreds of patients and the intervals do not overlap. Small
  groups are labelled inconclusive for exactly that reason.
* **Why is AUC only 0.68?** Readmission is hard to predict from this kind of data. The project is about the audit,
  not chasing a higher score.
* **Is it clinically usable?** No: the data is old and unvalidated externally, and it is labelled a research tool.
* **What next?** External validation, more candidate mechanisms, repeated splits.
