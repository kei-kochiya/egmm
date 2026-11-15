# Addendum to the holdout pre-registration (written 2026-11-15, after the holdout was run)

`HOLDOUT_PREREGISTRATION.md` is left exactly as committed with the library freeze (commit `7d6b9c9`). This
addendum records, without editing it, where the executed study departs from that text. The same list is in
the paper (Section "Pre-Registered Holdout", paragraph "Deviations from the pre-registration").

1. **Two datasets were already cached.** The registration says the library was frozen "before any of the
   datasets below was downloaded". `eeg-eye-state` and `isolet` were already in the local OpenML cache from a
   dataset search on the day before registration. No fit or analysis of them exists in the repository before
   the registration; the paper reports every holdout result also without them (post-hoc robustness table, column "Excl. pre-cached").
   The accurate statement is: library code frozen and committed before any holdout dataset was fitted, and
   before 14 of the 16 were downloaded.
2. **`dionis` was re-run.** The harness stopped on `dionis` when raw scikit-learn failed to fit (singular
   covariances at `reg_covar=1e-6`). Per-fit failure recording and a single-dataset re-run option were added
   to the harness (not to the library), and `dionis` was re-run for all methods with the same seeds.
3. **Failed-cell rule fixed after the failure.** A method/dataset cell whose fit failed is excluded only from
   that method's comparisons (here: raw `SklearnGMM` on `dionis`).
4. **No primary hypothesis or margin.** The registration fixed datasets, rules, methods, and statistics but
   named no primary hypothesis or equivalence margin, so the holdout supports out-of-sample estimation rather
   than a confirmatory test. Every exclusion in the paper's robustness table is a post-hoc sensitivity
   analysis next to the registered all-dataset analysis.
5. **Post-hoc follow-up runs (added 2026-11-15, after this addendum).** The holdout was later re-timed against
   stronger batch-EM baselines and used for three post-hoc ablations (`experiments/next/`, stage `holdout`;
   library unchanged, NLLs identical to the registered runs). `volkert` and `isolet` were also used to diagnose
   the diagonal-covariance losses. These analyses are labelled post hoc in the paper and are not part of the
   registered evidence; `volkert` and `isolet` can no longer serve as unseen data for a diagonal-mode rule.
