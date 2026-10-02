# Manuscript correction from the recovered fixed-noise records

The full original fixed-noise study was recovered from
NewIEEEPro/tnnls_step3_outputs/. Every rounded trajectory-MSE and delay-RMSE
median in Table S2 agrees with these records. P3 acceptance also agrees:
75/105 evaluations, with zero observed accepted-large-error events.

P3 improves over P4 in **four of five initial fixed-noise pairs**, not all
five. The fresh-noise confirmation remains **15/15**, with unchanged effect
sizes.

## Evidence

| Optimization seed | P3/P4 delay-RMSE ratio | P3 improves |
|---|---:|---|
| 20260719 | 0.515114 | Yes |
| 20260720 | 0.590085 | Yes |
| 20260721 | 0.325167 | Yes |
| 20260722 | 0.285225 | Yes |
| 20260723 | 1.234533 | No |

Exact values are retained in step3_principal_runs.csv. Regenerating manuscript
artifacts also writes the paired values and comparison JSON. The ratio of
marginal median delay RMSEs is 0.431672 (rounded to 0.432). The median paired
ratio is 0.515114, a separate statistic.

## Main article, Section 10.1

Replace the initial five-seed paragraph with:

~~~latex
In the initial five-seed ablation, P3 achieves median delay RMSE
\(1.313\times10^{-2}\), compared with \(3.041\times10^{-2}\)
for rollout-only P4, and improves delay recovery in four of the five
paired runs. P1, which also uses profile initialization, reaches
\(1.593\times10^{-2}\), whereas P2 does not consistently improve on P4.
These results identify the anchor objective, rather than profile
initialization, as the component selected for fresh-noise confirmation.
~~~

## Supplement, Section S.VIII-A

Replace the sentence claiming all five P3 improvements with:

~~~latex
P3 improves delay RMSE over P4 in \(4/5\) paired runs and has a
ratio of marginal median delay RMSEs of \(0.432\).
~~~

Keep Table S2 run counts at five. Only the number of P3/P4 paired improvements
changes. No Figure 2 caption change is needed for this count correction.

