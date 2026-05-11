# Time Series Forecasting: Foundational Papers

This note covers four papers that form the historical and conceptual backdrop for modern transformer-based time series forecasting. They are not implemented in this project but are essential context for understanding why PatchTST, iTransformer, and TimeMixer are designed the way they are.

---

## Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

**Authors:** Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang  
**Venue:** AAAI 2021 (Best Paper Award)  
**arXiv:** 2012.07436

### Problem

Standard transformers have O(L^2) time and memory complexity in sequence length L. For long-horizon forecasting (predict 720 steps from 1000+ steps of history), this becomes prohibitive. The encoder-decoder architecture also forces a step-by-step decoding strategy at inference time, accumulating errors across each predicted step.

### Core Contributions

**ProbSparse Self-Attention.** The observation is that self-attention weight distributions follow a long-tail pattern: a small number of queries (the "active" ones) dominate the attention map, while the rest produce near-uniform distributions that contribute negligible information. Informer exploits this by sampling only a subset of queries -- those with high "sparsity measurement" (KL divergence between their attention distribution and a uniform prior) -- and computing full attention only for those. This reduces time and memory complexity from O(L^2) to O(L log L).

**Self-Attention Distilling.** After each encoder layer, the sequence is halved via a 1D max-pooling operation with stride 2. This progressive downsampling reduces the total space complexity to O((2 - epsilon) * L * log L) and forces the model to prioritize the most informative representations.

**Generative Style Decoder.** Instead of decoding step by step (which accumulates error), the decoder receives a start token concatenated with placeholder zero vectors of length equal to the prediction horizon. It produces the entire output in a single forward pass, eliminating sequential error propagation.

### Why It Matters for This Project

Informer introduced the ETT (Electricity Transformer Temperature) dataset, which is the benchmark used throughout Week 4. Its split protocol (chronological, 12/4/4 months for train/val/test on ETTh1) became the standard that PatchTST and iTransformer both follow. Understanding Informer's limitations -- particularly that ProbSparse attention still does not solve the fundamental problem of treating individual time steps as tokens -- explains why PatchTST's patching strategy was such a significant advance.

### Limitations (Important)

Subsequent work, including the DLinear paper, demonstrated that Informer's improvements over naive Transformer baselines were less impressive once controlled properly. The ProbSparse approximation sacrifices some accuracy for efficiency, and the generative decoder, while faster, imposes structural constraints on what the model can learn. Informer's architectural ideas were influential but its empirical claims were partially revisited.

---

## Are Transformers Effective for Time Series Forecasting?

**Authors:** Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu  
**Venue:** AAAI 2023 (Oral)  
**arXiv:** 2205.13504

### Problem

This paper asks a direct and uncomfortable question: given that transformers for time series (Informer, Autoformer, FEDformer) all modify the attention mechanism to handle long sequences, do these modifications -- or indeed the attention mechanism itself -- actually help? Or is the improvement simply coming from the direct multi-step forecasting formulation that these models happen to use?

### Core Contributions

**The Linear Baseline Family.** The authors propose three minimalist models:

- **Linear**: a single linear layer mapping the input sequence directly to the prediction horizon, with no other components.
- **NLinear**: same as Linear, but first subtracts the last observed value from the entire input sequence (a simple normalization for distribution shift), adds it back after the linear layer.
- **DLinear**: decomposes the input into trend and seasonal components using a moving average kernel (borrowed from Autoformer), then applies one linear layer to each component independently, and sums the outputs.

**The Main Finding.** DLinear outperforms or matches Informer, Autoformer, and FEDformer on most ETT, Weather, Electricity, Traffic, and Exchange benchmarks at long prediction horizons. A single linear layer with decomposition beats years of attention mechanism engineering.

**Why Transformers Struggle.** The authors argue that transformers are permutation-invariant by design: attention computes pairwise similarities between tokens regardless of their order. Temporal ordering is crucial in time series, and point-wise attention between individual time steps loses this ordering information. The architectural modifications in Informer and its descendants mitigate this problem partially but do not solve it.

### Why It Matters for This Project

DLinear is the sanity check baseline in Week 4. If PatchTST does not outperform a linear model with decomposition, the implementation is wrong. More importantly, the DLinear paper explains *why* PatchTST works: by grouping time steps into patches before computing attention, PatchTST gives the model local temporal structure, partially restoring the ordering information that point-wise attention destroys. The DLinear paper is the conceptual bridge between "attention doesn't work for time series" and "patched attention does."

### Note on Subsequent Debate

The DLinear finding was contested. A HuggingFace analysis showed that under controlled conditions (same parameter count, univariate setting), Autoformer outperforms DLinear. PatchTST then definitively demonstrated that transformer-based models can surpass DLinear when patching is applied. The current consensus is that DLinear exposed real architectural flaws in pre-PatchTST transformers, but transformers are not inherently unsuitable for time series.

---

## Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

**Authors:** Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long  
**Venue:** NeurIPS 2021  
**arXiv:** 2106.13008

### Problem

Long-term time series contain complex trend and seasonal components that interact in non-trivial ways. Standard transformers treat the entire raw series as input without decomposing these components, making it harder to model their distinct dynamics. Additionally, point-wise self-attention between individual time steps treats each timestep as an independent token, losing the sub-series structure that is characteristic of temporal data.

### Core Contributions

**Progressive Decomposition.** Rather than decomposing the series once as pre-processing, Autoformer embeds decomposition as an inner block within each encoder and decoder layer. At each layer, a moving average operation extracts the trend component; the remainder is the seasonal component. This allows decomposition to interact with the learned representations across layers rather than being fixed at the input.

**Auto-Correlation Mechanism.** This replaces self-attention entirely. The key insight comes from the Wiener-Khinchin theorem in stochastic process theory: the auto-correlation function of a stationary process captures its periodic structure. Auto-Correlation computes similarity between the input series and its time-lagged versions (using Fast Fourier Transform for efficiency), producing O(L log L) complexity. The top-k lag correlations are selected, and the corresponding sub-series are aggregated by rolling (cyclic shifting). This is fundamentally different from point-wise attention: it captures sub-series-level dependencies rather than token-to-token interactions.

**Reported Results.** Autoformer achieves a 38% relative MSE reduction over prior state of the art across six benchmarks. On the ETT dataset with the input-96-predict-336 setting, MSE drops from 1.334 (previous best) to 0.339.

### Why It Matters for This Project

Autoformer's decomposition idea is the direct ancestor of TimeMixer's PDM blocks. The moving average trend extraction in DLinear also comes from Autoformer. Understanding Autoformer's progressive decomposition architecture makes TimeMixer's multiscale extension of the same idea much easier to follow.

### Limitations

Auto-Correlation complexity is O(L log L) but the constant factor is larger than standard attention for typical sequence lengths. More significantly, the DLinear paper showed that a simple linear model with Autoformer's own decomposition could match or exceed the full Autoformer on several benchmarks, suggesting that the Auto-Correlation mechanism itself may not be the primary source of improvement.

---

## Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

**Authors:** Jiehui Xu, Haixu Wu, Jianmin Wang, Mingsheng Long  
**Venue:** ICLR 2022 (Spotlight)  
**arXiv:** 2110.02642

### Problem

Unsupervised anomaly detection in time series is difficult because anomalies are rare, unlabeled, and must be distinguished from complex normal temporal patterns. Reconstruction-based approaches (train a model on normal data, flag high-reconstruction-error points as anomalous) are the dominant paradigm, but reconstruction error alone conflates anomaly difficulty with forecasting difficulty: a hard-to-predict normal point and a genuine anomaly produce similar reconstruction errors.

### Core Observation

Because anomalies are rare, an anomalous time point cannot build meaningful long-range associations with other points in the series (there are too few similar anomalies for the model to attend to). Instead, anomalous points can only attend meaningfully to their immediate neighbors. Normal points, by contrast, can attend broadly across the series, capturing periodic and trend dependencies. This implies an inherent difference in the attention distribution shape between normal and anomalous points.

### Core Contributions

**Association Discrepancy.** For each time point, the Anomaly Transformer computes two association distributions:

- **Series-Association**: the standard learned self-attention weights from the raw series. Reflects what the model actually attends to.
- **Prior-Association**: a Gaussian kernel centered on each time point. Reflects the expected local concentration around an anomaly.

The Association Discrepancy is the KL divergence between these two distributions. For normal points, series-association is spread broadly (diverse attention), while prior-association is locally concentrated. The KL divergence is large. For anomalous points, series-association collapses to local neighbors (because there is no global structure to attend to), bringing it close to the Gaussian prior. The KL divergence is small. Large discrepancy = normal. Small discrepancy = anomalous.

**Minimax Training Strategy.** Training with only reconstruction loss would make the series-association collapse everywhere (minimizing reconstruction does not require large discrepancy). The minimax strategy addresses this: the minimize phase drives the prior-association to approximate the series-association; the maximize phase pushes them apart, constrained by the reconstruction loss. This amplifies the distinguishability of the association discrepancy between normal and anomalous points.

**Evaluation.** Achieves state-of-the-art results on six benchmarks spanning server machine data (SMD), Mars Science Laboratory (MSL), Soil Moisture Active Passive (SMAP), and PSM datasets.

### Why It Matters for This Project

The anomaly detection module in Week 4 uses reconstruction error from PatchTST as the anomaly signal -- a simpler approach. The Anomaly Transformer represents the principled alternative: rather than using a forecasting model's error as a proxy, it trains an architecture whose attention mechanism is explicitly designed to distinguish normal from anomalous association patterns. Understanding this contrast sharpens the interpretation of Week 4's anomaly results: reconstruction error correlates with anomaly presence, but it is a noisy signal. The Anomaly Transformer's association discrepancy is a cleaner signal precisely because it is designed for detection, not forecasting.

### Limitations

The adjustment operation used for evaluation in the original paper (flagging an entire anomaly window as detected if any point within it is flagged) inflates precision and recall scores compared to point-wise evaluation. This has been noted in subsequent work. Additionally, the method requires the anomaly-to-normal ratio to be approximately reflected in the training data distribution, which may not hold in all industrial settings.
