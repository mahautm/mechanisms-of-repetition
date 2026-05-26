# Untrained Model Repetition Experiment

## Objective
Investigate the phenomenon where completely untrained (step 0, randomly initialized) transformer models exhibit degenerate repetition under greedy decoding. 
This experiment will:
1. Empirically verify that this happens across several standard model architectures when weights are randomly initialized.
2. Track the evolution of the output probability distribution during generation.
3. Test hypotheses about the theoretical causes of this repetition.

## Hypotheses for Untrained Repetition

Is this known? Yes. The phenomenon of repetition in greedy decoding is broadly known ("The Curious Case of Neural Text Degeneration", Holtzman et al. 2019), but in *untrained* networks specifically, it often relates to architectural priors, rank collapse, and initialization dynamics.

Here are a few specific hypotheses for why an untrained transformer collapses into repetition:

1. **Attractor Dynamics in the Transformer Equation**: 
   A transformer's forward pass can be modeled as an unrolled, interacting dynamical system. With random weights, the repeated application of residual connections, self-attention, and MLPs across multiple layers tends to cause the hidden states to contract towards a fixed point or a low-dimensional "attractor" manifold. Once the state enters this attractor sequence (often heavily influenced by the immediately preceding tokens), the output logits become stationary, causing the `argmax` operation in greedy decoding to repeatedly select the exact same token over and over.

2. **Rank Collapse / Attention Smoothing**: 
   Without training to separate attention heads to look for specific features, random attention matrices tend to act as uniform smoothing operators over the context. Especially with causal masking, the representations of all tokens can quickly collapse into a highly correlated, rank-deficient state (Dong et al., 2021). When all context embeddings look the same, the predicted next token is identical at every step.

3. **Narrow Cone Projection and Anisotropy**: 
   Randomly initialized high-dimensional weights, combined with LayerNorm centering, can project hidden states into a narrow "cone" in the embedding space. This structurally restricts the output logits, heavily biasing the model to unconditionally output from a very small subset of the vocabulary. If one token gets a slight advantage, it dominates all context windows.

## Plan
- **Empirical Check**: Initialize standard architectures (e.g., GPT-NeoX/Pythia, LLaMA, GPT-2) with completely random weights (`from_config`).
- **Distribution Tracking**: Generate text greedily and track the top-1 logit, entropy, and max probability of the output distribution over time.

## Mathematical Formulation of Untrained Repetition

We can formalize the inevitability of repetition in an untrained transformer using the concept of **Rank Collapse** driven by the exponential convergence of row-stochastic matrices (as analyzed theoretically by Dong et al., 2021).

### 1. The Transformer Forward Pass as a Markov Chain
Let $X^{(\ell)} \in \mathbb{R}^{N \times d}$ be the hidden states of $N$ tokens at layer $\ell$.
A simplified pure self-attention update (omitting the skip connection for pure analysis) computes:
$$ X^{(\ell+1)} = P^{(\ell)} X^{(\ell)} W_V^{(\ell)} $$
where $W_V^{(\ell)}$ is the value weight matrix and $P^{(\ell)} \in \mathbb{R}^{N \times N}$ is the attention matrix:
$$ P^{(\ell)} = \text{softmax}\left( \frac{X^{(\ell)} W_Q^{(\ell)} (X^{(\ell)} W_K^{(\ell)})^T}{\sqrt{d}} \right) $$

Crucially, $P^{(\ell)}$ is a **row-stochastic matrix** (its rows sum to 1 and are non-negative). 

### 2. Contraction of the Hidden States
Repeated multiplication by row-stochastic matrices strongly contracts the distances between the rows of $X$.
If we strip away $W_V$ for a moment, the product of many random stochastic matrices $P^{(L)} P^{(L-1)} \dots P^{(1)}$ converges exponentially fast to a rank-1 matrix $\mathbf{1} \pi^T$, where $\mathbf{1}$ is the all-ones vector. 

This means that at deep layers, the embeddings for all tokens become identical:
$$ \lim_{L \to \infty} X^{(L)} = \mathbf{1} v^T $$
Therefore, for any two positions $i, j$ in the context window:
$$ x_i^{(L)} \approx x_j^{(L)} \approx v $$

### 3. The Stationary Distribution and Greedy Repetition
At generation step $t$, the predicted distribution for the next token is determined by the final hidden state of the last token in the sequence context $x_t^{(L)}$:
$$ p(t_{t+1}) = \text{softmax}(W_U x_t^{(L)}) $$

Because of the rank collapse, after a certain depth $L$, the hidden state $x_t^{(L)}$ collapses to the fixed vector $v$ *regardless* of the input context $x_1 \dots x_t$.
Thus, the predicted output distribution becomes constant and stationary:
$$ p(t_{t+1}|t_1 \dots t_t) \approx \text{softmax}(W_U v) = \mathcal{C} $$

Under **greedy decoding**, the predicted token is always simply the argmax of this constant distribution:
$$ t_{t+1} = \text{argmax}(\mathcal{C}) $$
Since this $\text{argmax}$ does not change as $t$ increases, the model deterministically emits the exact same token $t^*$ forever:
$$ t_{t+1} = t_{t+2} = t_{t+3} = \dots = t^* $$

### 4. Skip Connections and the Lipschitz Bound
While modern transformers include skip connections ($X^{(\ell+1)} = X^{(\ell)} + \text{Attn}(X^{(\ell)})$), random initializations can still be highly contractive depending on the initialization variance.
If the spectral norm of the residual branch is small (controlled by the initialization scale $\sigma$), the layer-to-layer transition $F(x) = x + f(x)$ maintains a Lipschitz constant $K < 1$ with respect to the distances between token representations, enforcing the Banach Fixed-Point Theorem across the sequence dimension. The representations still collapse into a narrow cone, triggering the same stationary logit behavior and, therefore, an infinite repetition period.

### 5. Theoretical Bounds on Onset and Period

Given this dynamical systems perspective, we can establish bounds on the structural characteristics of the repetition:

#### A. Repetition Period Bound
Because greedy decoding is deterministic, the generation process creates a deterministic map over the token vocabulary $V$. 
1. **Absolute Upper Bound**: If the model has an effective non-decaying memory of $W$ tokens (e.g., restricted by an attention window, or if it considers the full KV cache), the maximum possible period before repeating a state is $O(V^W)$.
2. **Contractive Effective Bound**: Due to the rank collapse property established in Sections 2 and 4, the deep continuous mapping $x_{t+1} = F(x_t)$ contracts towards a low-dimensional manifold. If the effective Lipschitz constant $K_{gen}$ of the step-to-step generator is $< 1$, the Banach Fixed-Point Theorem guarantees a **unique fixed point**. 
   Therefore, under sufficient contraction (which is common in untrained initializations), the expected theoretical period is exactly **1** limit cycle: a single repeating token (or a very short 2-token cycle if the mapping oscillates around the fixed point).

#### B. Onset Time (Tokens Until Repetition)
How many tokens $T$ does it take for the model to "lock into" the repetition? 
The state space of the logits collapses at a rate proportional to the contraction factor $\lambda$ of the causal attention mixing across time steps. 
Let $\Delta_0$ be the initial maximum distance between the true continuous fixed-point $v^*$ and the initial token representations. We want to find the step $T$ where the distance to the fixed point is less than $\epsilon$, where $\epsilon$ is the margin required such that the discrete `argmax` operation no longer changes.

Using the exponential contraction bound:
$$ \Delta_T \leq \lambda^T \Delta_0 $$
Setting $\Delta_T < \epsilon$, we can bound the onset step $T$:
$$ T \leq \left\lceil \frac{\log(\Delta_0 / \epsilon)}{|\log \lambda|} \right\rceil $$

**Interpretation**: 
- If the initialization variance is small, $\lambda \ll 1$, meaning $|\log \lambda|$ is large, and $T$ is very small (immediate repetition lock-in).
- If the variance is very large, $\lambda$ might approach or exceed 1 initially (chaotic regime), delaying the onset $T$ until the sequence length forces a dominant dependency, or preventing collapse entirely until $T \approx \text{context\_window\_size}$.


### 6. Predicted Impact of Key Variables

To empirically validate these bounds, we can systematically vary four key axes and predict their effects:

1. **Input Tokens (Prompt Length/Content)**: 
   *Prediction*: Varies the initial distance $\Delta_0$ to the attractor. Because of the strong $L$-layer contraction per generated token $T$, changing the prompt might shift the exact onset by a token or two, but it should **not change the final repeating token** (the fixed point) or the period if the attractor is unique.

2. **Initialization (Variance/Scale)**: 
   *Prediction*: Directly controls the contraction factor $\lambda$. 
   - **Small scale**: $\lambda \ll 1 \Rightarrow$ Fast onset $T$ (collapses almost instantly).
   - **Large scale**: $\lambda \to 1 \Rightarrow$ Delayed onset $T$ or completely chaotic generation.

3. **Generation Length (Number of Tokens)**: 
   *Prediction*: If the generation length is too short, we miss the onset $T$. However, once $T$ is reached (the sequence enters the limit cycle), extending generation infinitely will simply loop the sequence endlessly since the state maps to itself (period = 1). The generated sequence is deterministic after $T$.

4. **Model Size (Depth and Width)**: 
   *Prediction (Depth)*: The state passes through $L$ layers per generated token. Contraction per token is effectively proportional to $\lambda^L$. **Deeper models will collapse into repetition much faster** (smaller $T$) due to exponential bounding.
   *Prediction (Width/Heads)*: Wider models (larger $d$) might resist rank collapse longer geometrically, but more attention heads average out the variance faster (law of large numbers), pushing the attention matrix toward uniformity sooner. Overall, larger models (where depth scales up) generally collapse significantly faster because the exponential depth effect overpowers width.

### 7. Bridging the Gap: Why Trained Models Still Repeat Under Uncertainty

While training specifically optimizes the weights to break this rank collapse (by separating attention heads and scaling up MLP norms to push representations apart), trained models still famously fall into repetition traps—especially when uncertain or processing out-of-distribution (OOD) text. 

Your intuition strongly aligns with the dynamical systems view: **when structured activations fail, the model defaults back to its architectural prior (the untrained contractive state).**

#### A. Reverting to Uniform Attention (The Rank Collapse Returns)
In a healthy trained state, attention heads find specific matching features, resulting in a high variance $Q \cdot K^T$ matrix (sharp, peaky softmax). 
However, under textual uncertainty or OOD prompts, the queries do not strongly match any keys in the context. The dot products in $Q \cdot K^T$ remain small. Consequently, the `softmax` operation produces a near-uniform distribution over the context. 
As proven in Section 2, uniform-like stochastic matrices are aggressive smoothing operators. The representations of all tokens are averaged together, deleting sequence-specific information and causing contextual rank collapse.

#### B. The Lipschitz Norm Drop
Trained models escape the Banach Fixed-Point attractor by ensuring the residual updates $f(x) = \text{Attn}(x) + \text{MLP}(x)$ have high enough norms (Lipschitz constant $K \geq 1$) to prevent the hidden state from contracting. 
When the model is "uncertain," it fails to confidently activate its learned non-linear features. The norm of the residual updates drops ($||f(x)|| \ll ||x||$). The layer transition $x_{\ell+1} = x_\ell + f(x_\ell)$ becomes dominated by the identity mapping, pulling the effective sequence-to-sequence Lipschitz constant back below 1. The network re-enters the contractive regime and falls toward a fixed-point attractor.

#### C. The Self-Reinforcing Trap
Once uncertainty triggers a partial collapse, the model outputs a "safe," high-frequency, generic token (the center of the cone). On the next generation step, this generic token provides even *less* distinct information to the attention heads, ensuring the dot products remain low and the residual norms remain small. The rank collapse becomes self-reinforcing, locking the trained model into the exact same degenerative limit-cycle (period = 1 or 2) observed in randomly initialized networks.



## 8. Experimental Design: Proving the "Uncertain Attractor" in Trained Models

To convince a rigorous reviewer that *Natural Repetitions* in fully-trained models are caused by this exact attractor mechanism, we need to show the temporal collapse of internal representations right as the repetition begins.

### The Phenomenon to Capture
In our workspace, "Natural" repetitions are defined as cycles occurring when the model generates unconditionally from plain text (e.g., Minipile), as opposed to few-shot (ICL) prompts. We need to track the model token-by-token and show that the transition from "healthy text" to "repetition trap" exactly coincides with a drop in Lipschitz norms and attention rank collapse.

### What to Measure (The Metrics)
At every generation step $t$, across all layers $\ell$, we measure:
1. **Residual Update Norm Ratio**: $\frac{||\text{Attn}(x_t^\ell) + \text{MLP}(x_t^\ell)||}{||x_t^\ell||}$. 
   *Hypothesis*: This ratio will drop significantly just before onset, pushing the effective Lipschitz constant $K < 1$.
2. **Attention Entropy / Uniformity**: The entropy of the attention probability distribution $H(P^\ell)$.
   *Hypothesis*: Entropy will spike toward $\log(\text{context\_size})$ (approaching uniform attention) as the model becomes uncertain, smoothing out the sequence.
3. **Contextual Rank Collapse (Cosine Similarity)**: The average cosine similarity between the current hidden state $x_t^L$ and the previous $k$ hidden states $x_{t-1}^L, \dots, x_{t-k}^L$.
   *Hypothesis*: As the attractor takes over, this similarity will asymptotically approach $1.0$, indicating that the sequence representations have collapsed into a single fixed vector.

### Experimental Protocol
1. **Find the Prompts**: Use a trained model (e.g., Pythia-70m or Pythia-1.4B) and sample generations from Minipile prefixes until we harvest ~100 trajectories that cleanly fall into a repetition loop (extracting the exact `onset` step for each).
2. **Harvest a Control Group**: Sample 100 trajectories that *never* repeat within the same generation length.
3. **Align and Contrast (The Reviewer Convincer)**: 
   - Align all repeating trajectories around $t = \text{Onset Token}$.
   - Plot the three metrics from $(t - 20)$ to $(t + 20)$.
   - Overlay the control group metrics for baseline scale.

### Why this convinces a reviewer:
Reviewers are highly skeptical of theoretical claims without mechanistic evidence. If we show a clear, measurable state-change—specifically that **attention flattens, residual norms crater, and token states merge** starting 3-4 tokens *before* the discrete repetition loop locks in—it proves the deterministic dynamical systems collapse is the underlying causal driver of the bug. It changes the narrative from "the model predicts the same token" to "the model structurally lost the ability to differentiate sequence states."
