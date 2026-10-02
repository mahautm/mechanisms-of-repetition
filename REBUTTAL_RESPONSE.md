# Response to Reviewer

We thank the reviewer for the detailed and constructive critique. We address the core points below.

## Head ablation mechanics

The reviewer is correct to flag this — we found and fixed a genuine bug. In the original implementation, the ablation hook was a `forward_hook` registered on the attention module itself, slicing the attention module's *output*. Since that output has already passed through the output projection (`o_proj`/`out_proj`/`c_proj`/`dense`/`wo`), which is a full `d_model x d_model` mixing matrix, a contiguous slice of it blends contributions from every head rather than isolating one.

We corrected this: the hook is now a `forward_pre_hook` registered on the output projection submodule, so it intercepts the projection's *input* — the true per-head concatenation — and zeros a contiguous `head_dim`-sized slice *before* the projection mixes heads together. We verified the fix with a smoke test (`EleutherAI/pythia-70m`, 3 head-ablation policies, 48 samples each) and are rerunning the full ablation suite on the corrected hook for the revision. We will add this implementation detail explicitly to the paper and release the hook code.

## Prompt construction and circularity

Natural and ICL prompt sets are rebuilt independently at each checkpoint from a fixed raw-text pool (prompts are re-classified into conditions using that checkpoint's own greedy generations, not selected once from a final model and reused). This avoids the specific circularity concern the reviewer raised.

The length and construction differences between the two conditions are intentional rather than an oversight: shorter, unconstrained generations were needed to avoid a ceiling effect in which very long generations trivially reach a repetitive cycle regardless of condition, which would wash out the comparison we are trying to make. This rationale was not stated in the paper; we will add it explicitly, together with matched-length and matched-structure control conditions, so readers can distinguish confounds we designed around from ones we overlooked.

## Statistics

We agree the significance claims were underspecified. We are now computing per-head z-tests against a null derived from resampled baselines, with both Bonferroni and Benjamini-Hochberg correction across heads and checkpoints, and will report effect sizes with confidence intervals and paired tests in the revision.

## Overclaiming

We will soften "innate," "default-mode," and "causally proven" throughout, and reframe the newline/structural-token analysis as correlational rather than causal.

## Reporting inconsistencies

The Table 1 (1000-token) vs. Section 3.1 (256-token) discrepancy, and the mismatch with Appendix E's repetition definition, were real errors. We have corrected the definition in the paper and re-verified that the reported results are not materially affected by the fix; we will state this explicitly in the revision along with the corrected numbers.

## Generality

We will promote Pythia-70M and OLMo-1B from appendix to main-paper figures alongside Pythia-1.4B, and add non-greedy decoding checks (temperature, top-k, repetition penalty).
