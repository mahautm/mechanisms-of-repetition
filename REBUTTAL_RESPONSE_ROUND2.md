We thank the reviewer and report the requested results.

Head ablation. The hook is a forward pre-hook on the attention output projection (dense / o_proj). It zeroes the ablated head's slice of the projection input, before heads are mixed:

    def hook(module, args):
        x = args[0].clone(); d = x.size(-1) // n_heads
        x[..., h*d:(h+1)*d] = 0
        return (x, *args[1:])

Results are for Pythia-1.4B (4-bit weights; fp16 reruns in progress). For each checkpoint we give the baseline repetition rate (%, 95% CI) and the mean change under ablation over all 384 heads (Δ, pp), for Natural and then ICL. The final number is how many heads remain significant after BH correction:

    step0       95.2 [93.6, 96.4], -0.67 | 86.6 [76.4, 92.8], +2.69 | 0
    step1       88.6 [86.3, 90.6], -0.15 | 82.7 [75.8, 87.9], -0.38 | 0
    step1000    99.8 [99.3, 99.9], -0.01 | n = 4                    | 0
    step36000   99.9 [99.3, 100],  -0.25 | 92.8 [87.2, 96.0], +1.00 | 0
    step71000   99.7 [99.0, 99.9], -0.48 | 90.2 [83.9, 94.2], +1.50 | 0
    step143000  99.5 [98.8, 99.8], -0.43 | 91.4 [85.3, 95.1], -0.60 | 1 (L2H5, Natural, -2.5 [-3.7, -1.3])

Exact McNemar tests on per-prompt outcomes likewise find no significant head at step0. Per-head paired confidence intervals from these reruns will accompany the final results; the other checkpoints are running.

Prompt populations. Originally, 32-token Minipile excerpts from a fixed pool were assigned per checkpoint according to that checkpoint's greedy continuation. Natural prompts were the excerpt plus the continuation up to cycle onset, so their length varied; ICL prompts were the excerpt duplicated. In the paired analysis, each source s gives Natural = s[0:64] and ICL = s[0:32] repeated twice. The two are length-matched, involve no behavioural selection, and use the same 500 sources at every checkpoint. At step0, ICL − Natural is +2.0 pp [-0.8, 4.8] for baseline repetition (p = .21) and -1.5 pp [-4.1, 1.0] for the ablation effect (p = .70). No head differs between conditions after correction. The other checkpoints are running.

Direction of causal effects. Under Equation (1), Δ > 0 indicates a suppressing head and Δ < 0 a facilitating head. Single-head effects are small and not significant, with one exception: L2H5 facilitates Natural repetition. The majority signs are descriptive only: facilitating for Natural (345/384 heads at step71000) and suppressing for ICL (289/384). We therefore withdraw the claim that the circuit promotes ICL repetition.

Learned lens and initialization. The lens clarification will follow. Pythia step0 is the official "step0" revision, saved before any optimizer update; all step0 runs were verified from logs to load this revision. On the paired prompts, repetition at initialization is as follows:
- 51 independent official Pythia (GPT-NeoX) initializations, spanning 14M–410M with 10 PolyPythia seeds each, plus 1.4B: 64–81%, increasing with size, and 94% at 1.4B.
- The official OLMo-2-1B step0: 39–43%.
- A default (non-training) initialization of Llama-3.2-1B, five seeds: about 1%, so the effect is architecture-dependent.

We agree this is a pre-training behavioral observation that depends on architecture, not evidence for a dedicated circuit.
