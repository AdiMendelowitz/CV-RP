# Time Series Forecasting: Foundational Papers

This note covers four papers that form the historical and conceptual backdrop for modern transformer‑based time series forecasting. They are not implemented in this project but are essential for understanding why PatchTST, iTransformer, and TimeMixer are designed the way they are.

---

## Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

**Authors:** Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang  
**Venue:** AAAI 2021 (Best Paper Award)  
**arXiv:** 2012.07436

### Problem

Standard transformers have `O(L^2)` time and memory complexity in sequence length L, which becomes prohibitive for long‑horizon forecasting with long look‑backs. The canonical encoder–decoder architecture also encourages step‑by‑step decoding at inference time, causing error accumulation across the predicted steps.

### Core contributions

**ProbSparse self‑attention.**  
Empirically, many queries in self‑attention have nearly uniform attention distributions; only a subset of “active” queries concentrate mass on a few keys. Informer scores queries using a sparsity measurement (based on KL divergence to a uniform prior) and computes full attention only for the top queries, reducing complexity from `O(L^2)` to `O(Llog L)`.

**Self‑attention distilling.**  
After each encoder layer, Informer halves the sequence length using 1D max‑pooling with stride 2, progressively shrinking the time dimension. This further reduces memory and encourages the model to retain only the most informative temporal representations.

**Generative‑style decoder.**  
Instead of autoregressively predicting one step at a time, the decoder takes a start token plus zero‑filled placeholders of length equal to the prediction horizon and outputs all forecast steps in a single forward pass. This avoids sequential error accumulation.

### Why it matters for this project

Informer introduced the ETT datasets (including ETTh1) and a now‑standard chronological split protocol (e.g., 12/4/4 months for train/validation/test on ETTh1). PatchTST and iTransformer adopt these benchmarks and splits, so their results are interpreted in an Informer‑defined ecosystem. Informer also shows that **efficiency tricks on attention alone** do not fix the deeper issue of treating individual time steps as tokens, which motivates PatchTST’s patching approach.

### Limitations (important)

Later work, especially the DLinear paper, showed that once baselines are controlled and evaluation is consistent, Informer’s gains over simpler models are smaller than originally claimed. ProbSparse trades some accuracy for efficiency, and the generative decoder constrains how outputs are produced. Informer’s ideas were influential, but its empirical edge has been partially re‑evaluated.

---

## Are Transformers Effective for Time Series Forecasting?

**Authors:** Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu  
**Venue:** AAAI 2023 (Oral)  
**arXiv:** 2205.13504

### Problem

This paper asks whether existing transformer variants for long‑term time‑series forecasting actually benefit from their complex attention mechanisms, or whether most gains come from **direct multi‑step (DMS) forecasting** rather than the architecture itself.

### Core contributions

**Linear, NLinear, DLinear.**  
The authors introduce three minimalist baselines:

- **Linear:** a single linear layer mapping the look‑back window directly to the forecast horizon.  
- **NLinear:** same as Linear, but subtracts the last observed value from the input as a simple shift‑normalisation, then adds it back at the output.  
- **DLinear:** decomposes the input using a moving‑average kernel into trend and seasonal components, applies one linear layer to each component independently, and sums their outputs (borrowing decomposition from Autoformer).

**Main finding.**  
Across benchmarks (ETT, Weather, Electricity, Traffic, Exchange), DLinear often matches or outperforms Informer, Autoformer, and FEDformer at long horizons. A properly tuned linear model with decomposition can beat several years of transformer architecture engineering.

**Why transformers struggle (as implemented then).**  
The paper argues that tokenising individual time steps and using point‑wise attention over them makes transformers effectively **permutation‑equivariant in time**, with positional encodings only partially restoring order. Since temporal order is fundamental, this configuration wastes capacity; many architectural tweaks do not address this core issue.

### Why it matters for this project

DLinear is the **sanity‑check baseline** in Week 4: if PatchTST cannot beat DLinear on ETT, something is wrong with the implementation or training. Conceptually, DLinear explains *why* PatchTST works: by forming **temporal patches**, PatchTST restores local temporal structure and produces tokens that are more compatible with attention than single time‑step snapshots. DLinear is the bridge from “transformers seem ineffective” to “transformers can be effective with the right tokenization.”

### Note on subsequent debate

DLinear’s conclusions sparked debate. For example, later analyses (including independent HuggingFace reproductions) show that under equal conditions (e.g., univariate setting, matched parameter counts) some transformer models like Autoformer do outperform DLinear. PatchTST then showed that patched transformers can decisively surpass DLinear on many benchmarks. The emerging consensus is that DLinear exposed real flaws in older transformer designs and benchmarks, but did not prove transformers inherently unsuitable for time series.

---

## Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

**Authors:** Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long  
**Venue:** NeurIPS 2021  
**arXiv:** 2106.13008

### Problem

Long‑term series combine trend and seasonal components that interact at multiple scales. Standard transformers ingest the raw series without explicit decomposition, making it harder to model trend vs seasonal dynamics separately, and their point‑wise self‑attention again treats each time step as an independent token.

### Core contributions

**Progressive decomposition blocks.**  
Autoformer embeds a moving‑average decomposition block inside each encoder and decoder layer:

- A moving average extracts a **trend** component.  
- The residual is treated as a **seasonal** component.  
- Decomposition happens at each layer, so trend/seasonal separation evolves with the learned representation rather than being fixed preprocessing.

**Auto‑Correlation mechanism.**  
Instead of standard self‑attention, Autoformer introduces Auto‑Correlation, inspired by stochastic process theory and the Wiener–Khinchin theorem.

- It computes correlations between the series and its time‑lagged versions using FFTs, identifying top‑k lag indices with strong periodic structure.  
- Sub‑series at those lags are cyclically shifted and aggregated, yielding sub‑series‑level dependencies with `O(Llog L)` complexity.

**Reported results.**  
Autoformer reports about 38% relative MSE improvement over prior SOTA across six benchmarks (Wu et al. 2021, calculated from Table 1 averages across datasets). For example, on ETTh1 with input 96, predict 336, MSE drops from approximately 1.334 (prior SOTA) to 0.339 (Autoformer), per Table 1 in the paper.

### Why it matters for this project

Autoformer’s **embedded decomposition** is the direct predecessor of TimeMixer’s multi‑scale mixing blocks; TimeMixer essentially generalises “trend/seasonal decomposition per layer” into a multi‑resolution architecture. DLinear’s moving‑average trend extraction also comes directly from Autoformer’s decomposition idea. Understanding Autoformer makes it much easier to see what TimeMixer changes (no attention, multi‑scale mixing) and what it reuses.

### Limitations

Although Auto‑Correlation has `O(Llog L)` complexity, its constant factors can be higher than standard attention at moderate sequence lengths. DLinear further showed that applying Autoformer‑style decomposition followed by simple linear layers can match or surpass the full Autoformer in some settings, suggesting that decomposition, not the Auto‑Correlation module, may be the key contributor.

---

## Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

**Authors:** Jiehui Xu, Haixu Wu, Jianmin Wang, Mingsheng Long  
**Venue:** ICLR 2022 (Spotlight)  
**arXiv:** 2110.02642

### Problem

Unsupervised anomaly detection in time series is challenging: anomalies are rare, unlabeled, and must be separated from complex but normal dynamics. Reconstruction‑based methods (train on “normal,” flag high reconstruction error) mix up “hard‑to‑predict normal points” with true anomalies, since both can yield large errors.

### Core observation

Because anomalies are rare, a truly anomalous point has few similar points in the series and cannot form strong long‑range associations. It tends to attend to local neighbours only. Normal points, in contrast, form rich associations across the series (periods, trends, repeated patterns). This implies that **attention distributions** for normal vs anomalous points have qualitatively different shapes.

### Core contributions

**Association discrepancy.**  
For each time index, Anomaly Transformer defines two association distributions:

- **Series‑association:** learned self‑attention weights over all time steps (what the model actually attends to).  
- **Prior‑association:** a local Gaussian kernel centred on that time step (what we expect for a purely local, anomaly‑like pattern).

The **association discrepancy** is the KL divergence between these distributions.

- Normal points: series‑association is broad and multi‑modal, prior‑association is narrow; divergence is large.  
- Anomalous points: series‑association collapses to local neighbours, becoming similar to the Gaussian prior; divergence is small.

Thus **small** discrepancy indicates anomalies; **large** discrepancy indicates normality.

**Minimax training.**  
A naive reconstruction‑only objective can drive series‑association to collapse, reducing discrepancy everywhere. The authors propose a minimax scheme:

- Minimise phase: make prior‑association approximate series‑association.  
- Maximise phase: push them apart, under a constraint imposed by reconstruction loss.

This amplifies the distinguishability of anomalies in terms of association discrepancy.

**Evaluation.**  
Anomaly Transformer achieves state‑of‑the‑art results on six unsupervised anomaly benchmarks, including SMD, MSL, SMAP, and PSM, across service monitoring and space/earth exploration data.

### Why it matters for this project

Your Week 4 anomaly detection module uses **reconstruction error from PatchTST** as the anomaly score — simple and generic. Anomaly Transformer illustrates a more principled alternative: design the attention mechanism to make **normal vs anomalous association patterns** explicitly separable. This contrast clarifies how to interpret your results: reconstruction error is a useful but noisy proxy, whereas association discrepancy is a targeted signal for detection.

### Limitations

The original paper uses an **adjusted evaluation** strategy that credits a detection if any point within an anomaly window is flagged, which inflates precision and recall compared to strict point‑wise evaluation. Later work has highlighted this discrepancy. In addition, performance depends on the training data reflecting a realistic anomaly‑to‑normal ratio; strong distribution shifts in anomaly frequency can degrade performance.
