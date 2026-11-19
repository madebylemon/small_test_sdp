# Group 57 — Mini Experiment Speaking Notes

**Topic:** Scoring Multi-Agent Workflows with Fewer Test Runs by Reusing a Parent's Repeated Runs  
**Source:** Preliminary offline experiment reported by Antigravity on November 19, 2026. These results have not been independently reproduced from the implementation in this workspace.

## Slide 1 — Can parent history make child evaluation cheaper?

“So this is the basic idea behind our research. AFlow makes new workflows by editing existing ones. In this example, Round 2 is an ensemble solver and Round 3 adds a Programmer checker. The parent has already been evaluated five times per question. Instead of throwing away those results, we want to see if they tell us which questions will be unpredictable in the child, so we can spend testing runs more intelligently.”

## Slide 2 — A small offline replay

“We started with historical GSM8K data rather than paying for new model calls. We selected five actual parent-child relationships with different kinds of edits, tested the full hundred questions and a smaller thirty-question subset, and compared uniform testing with parent-guided methods and a simplified child-pilot baseline. For each question, we split the five existing runs into three available runs and two held-out reference runs. So we're comparing the methods at the same evaluation budget.”

**If asked:** Ten train/reference splits are combinations of the same five historical runs—not ten independent repeats. Our two-run reference is noisy.

## Slide 3 — Parent history helps, but the pilot still matters

“At 150 runs across the hundred-question benchmark, uniform testing had RMSE .01076. Basic parent guidance gave .01068, which is a small improvement. The trust-adjusted version gave .00991, about a 7.9 percent reduction. But the simplified child-pilot method actually had the lowest error at .00956, so we can't say our method has beaten the pilot yet. The parent-child variance correlation was stronger for some small edits, but almost disappeared after a much larger mutation. That suggests the kind of edit really matters.”

**If asked:** The smaller 30-question example at 75 runs showed parent-only RMSE .03081 versus uniform .03408, about a 9.6% reduction; the child pilot was .03004. None of these results establish statistical significance.

## Slide 4 — What we learned and next steps

“Our feasibility study suggests parent history can tell us something useful about which questions will be unstable in a child workflow. But the gains were mixed, and the reference score is based on only two held-out runs per question. We also need to evaluate more parent-child pairs and compare against a faithful implementation of Speculative Evaluation. So our question for Prof. Zhang is whether we can get a small amount of independent model evaluation time to validate this properly.”

## Cautions for questions from the professor

- **7.9%** is the reported RMSE improvement of **trust-calibrated parent guidance** at 150 runs—not basic P-PGA.
- **0.7%** is the approximate improvement of **basic P-PGA** at 150 runs over uniform testing.
- The **child-pilot baseline** used here is simplified, not the entire published Speculative Evaluation method.
- Results are from historical offline replay, not new live inference.
- Five equal observed scores don't prove a question is truly deterministic.
- The 150-run protocol is 70% fewer **rollouts** than 500, not proof of equal precision or 70% fewer actual dollars.
- The 5 chosen pairs and overlapping held-out splits do not justify broad statistical generalization.
