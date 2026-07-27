# NeurIPS 2026 Rebuttal Draft

## Response to Reviewer 9TTx

We thank the reviewer for the careful reading, the recognition of DPR's clean motivation and broad evaluation, and the constructive questions about its integration, interpretation, and scope.

> **Q1: How were the DPR insertion points selected? Were earlier-layer or multi-layer insertions considered?**

**A1:** We use one predefined late-stage integration rule for all seven host architectures: DPR is applied to the hidden representation immediately before the corresponding prediction head. For multi-scale or multi-branch backbones, the same DPR adapter is reused at the corresponding late-stage representations rather than introducing independent adapters. This preserves the backbone's feature extractor and hyperparameters, provides contextualized features for recalibration, and keeps the intervention comparable across architectures. We chose insertion points by this structural rule, without backbone-specific tuning.

Earlier-layer and multi-layer placements were outside the scope of this study because they introduce a backbone-dependent design space over layer locations, sharing strategies, adapter counts, and computational cost. Our goal is to evaluate DPR under one controlled minimal-intervention protocol rather than optimize a different adapter for each backbone. The reported gains show that this single placement rule transfers across diverse backbones; they do not establish that it is optimal for each architecture.

> **Q2: Does the DPR improvement correlate with the non-stationarity diagnostics?**

**A2:** We quantified this relationship across the twelve datasets using the reported per-horizon results. For each backbone and dataset, we first average the relative MSE reduction from `+DPR` over the four predefined horizons and then take the median across the seven backbones. We correlate this dataset-level DPR gain with the diagnostics using Spearman's rank correlation.

| Diagnostic | Spearman rho | Two-sided p-value | Interpretation |
|---|---:|---:|---|
| ADF p-value | 0.039 | 0.905 | No detectable association |
| Spectral entropy | 0.280 | 0.379 | Weak, non-significant association |
| Volatility-of-Volatility | 0.622 | 0.031 | Positive association |
| Composite rank score | 0.703 | 0.011 | Strongest positive association |

VoV most directly captures what DPR addresses: changes in local volatility. Spectral entropy measures frequency complexity and ADF tests global unit-root behavior; neither necessarily indicates changing local response requirements. The composite score, defined as `rank(spectral entropy) + rank(VoV)`, combines these complementary diagnostics and shows the strongest association. Given the twelve-dataset sample and the related diagnostics, we interpret these unadjusted correlations as exploratory rather than causal evidence.

The result supports a narrower relationship between DPR improvement and changing local volatility, not a generic association with every notion of non-stationarity. The marginal gains on ETT benchmarks are consistent with their approaching performance ceiling [1]; the meaningful challenge lies in volatile, non-stationary datasets where DPR provides consistent improvement. Accordingly, we will describe static pattern response as a regime-dependent limitation rather than an equally severe bottleneck on every dataset.

> **Q3: How is DPR related to FiLM and SE-style recalibration?**

**A3:** SE, FiLM, and DPR belong to the broad family of conditional feature-wise modulation. SE generates input-conditioned channel scales, while FiLM generates affine scale and shift parameters. Our distinction is therefore not the final multiplication, but the conditioning granularity and response-generation mechanism:

| Aspect | SE | FiLM | DPR |
|---|---|---|---|
| Conditioning source | Global descriptor squeezed from the same feature maps | An arbitrary conditioning input `z` (a question embedding in the original model) | Local temporal neighborhood of each token |
| Generated response | Bottleneck-generated sigmoid channel scales | Directly generated scale and shift | Soft combination of shared response bases through pattern routing |
| Granularity | One scale per channel, broadcast across spatial positions | One scale/shift per feature map, spatially agnostic in the original CNN model | One feature-response vector per temporal token |
| Main purpose | Model global channel interdependencies | General-purpose conditional transformation | Adapt to local regime changes within one sequence |

SE globally summarizes the current representation, while FiLM directly maps a condition to affine coefficients. DPR instead factorizes response generation: `Perceive` extracts local temporal context, `Route` matches it to learned centroids, and `Modulate` combines shared response bases. This constrains the gain to a low-dimensional, continuously varying response family rather than predicting a full response vector directly.

Both SE and FiLM can be adapted to time series. A direct SE adaptation pools over time and remains sequence-level. FiLM can also use a token-wise temporal conditioner; our claim does not rely on this being impossible. Even under this stronger Local FiLM setting with the same local context as DPR, FiLM directly predicts coefficients, whereas DPR uses explicit pattern routing and a shared response basis. In Q1/A1 of Reviewer Lms2, we add parameter-matched experiments comparing these three paradigms (SE, FiLM, and DPR variants) across four datasets; the factorized response proves useful in several regimes. DPR is therefore related to SE and FiLM at the operator level. It differs from standard SE in its token-wise temporal conditioning, and from direct Local FiLM primarily in its routed, basis-factorized response generation.

> **Q4: Is DPR feature-wise modulation rather than expert selection, and is it less expressive than MoE?**

**A4:** We view DPR as continuous feature-response composition rather than conventional expert selection. Its expressiveness should not be described as simply lower than MoE because the two mechanisms parameterize different conditional responses. A Top-K MoE selects a discrete subset of complete expert transformations. DPR instead softly combines all learned response bases into a token-specific feature response, producing a continuum of possible modulations. This design is motivated by time-series regimes that overlap or evolve gradually rather than switching between isolated experts.

Table 5 (`DPR vs. MoE Routing`) uses the same DPRNet setting and replaces DPR with an 8-expert Top-K MoE. DPR uses 325K-602K parameters versus 818K-1.1M for MoE and obtains lower MSE in the two evaluated datasets. On ILI, MoE Top-1/2/4 is 20.7%/21.9%/34.7% worse than DPR; on ETTh1, it is 4.8%/6.8%/6.2% worse. MoE also incurs higher training and inference time. These controls do not establish a universal expressiveness ordering; they show that DPR provides a parameter-efficient response class for the settings tested.

> **Q5: How should DPR behave when the useful signal is mainly exogenous?**

**A5:** DPR recalibrates using local hidden representations from the observed endogenous history — the setting evaluated in this work. When exogenous covariates are available (e.g., macroeconomic indicators, weather data, calendar features), their encodings can be fused with DPR's local context query for joint routing, complementing the endogenous signal for more accurate regime identification. The residual path and soft response-basis combination naturally accommodate this extension without architectural changes to DPR's core Perceive-Route-Modulate pipeline.

Beyond structured covariates, future work could incorporate multimodal information such as news text, social media signals, or domain-specific reports, enabling DPR to learn more generalizable patterns that transfer across related forecasting tasks. We view exogenous- and multimodal-conditioned recalibration as promising directions.


---
[1] Yuxuan Wang et al. Accuracy Law for the Future of Deep Time Series Forecasting. arXiv:2510.02729, 2025.

## Response to Reviewer Lms2

We thank the reviewer for the practical perspective, the appreciation of our diverse dataset coverage and backbone-agnostic design, and for focusing the discussion on novelty, statistical stability, dataset coverage, and the scope of our claims.

> **Q1: Is DPR technically distinct from FiLM, SE-style recalibration, dynamic convolution, and gated residual adapters? Can the authors add parameter-matched local modulation baselines?**

**A1:** DPR, FiLM, and SE all perform conditional feature modulation, but DPR parameterizes the response as a routed soft combination of shared bases rather than predicting a gate or affine coefficients directly. In Q3/A3 of Reviewer 9TTx, we provide a conceptual comparison of the three paradigms across conditioning source, response generation, and granularity. To test whether this factorization contributes beyond direct local gating, we compare five adapters at the same late-stage insertion point:

- `Global SE`: global temporal pooling followed by bottleneck excitation.
- `Local SE`: DPR's local perception followed directly by a sigmoid feature gate.
- `Local FiLM`: the same local perception followed directly by token-wise scale and shift prediction.
- `Gated residual`: the same local perception followed by a direct residual gate.
- `DPR`: local perception, cosine routing, response-basis combination, and residual modulation.

The local baselines share DPR's context input, receptive field, insertion point, and training budget. Their bottlenecks are fixed to match DPR's added parameters within 5%. We use four predeclared settings: ILI (`24->24`), COVID19 (`36->7`), VIX (`96->96`), and ETTh1 (`96->96`), covering small, volatile, financial, and hourly regimes.

| Backbone | Adapter | ILI 24->24 | COVID19 36->7 | VIX 96->96 | ETTh1 96->96 |
|---|---|---|---|---|---|
| PatchTST | None | 3.633/1.079 | 0.335/0.216 | 0.942/0.539 | 0.394/0.392 |
| PatchTST | Global SE | 3.352/1.049 | 0.329/0.218 | 0.950/0.548 | 0.392/0.393 |
| PatchTST | Local SE | 3.403/1.132 | 0.329/0.219 | 0.948/0.551 | 0.400/0.396 |
| PatchTST | Local FiLM | 3.245/1.053 | 0.330/0.216 | 0.973/0.551 | 0.396/0.395 |
| PatchTST | Gated residual | 3.398/1.131 | 0.324/0.215 | 0.950/0.551 | 0.400/0.396 |
| PatchTST | DPR | 3.106/1.043 | 0.327/0.217 | 0.940/0.538 | 0.392/0.394 |
| Crossformer | None | 4.736/1.480 | 0.609/0.295 | 1.023/0.571 | 0.394/0.404 |
| Crossformer | Global SE | 4.690/1.462 | 0.638/0.309 | 0.932/0.541 | 0.400/0.412 |
| Crossformer | Local SE | 4.764/1.473 | 0.682/0.307 | 1.004/0.566 | 0.402/0.417 |
| Crossformer | Local FiLM | 4.719/1.461 | 0.610/0.293 | 0.999/0.565 | 0.387/0.400 |
| Crossformer | Gated residual | 4.718/1.461 | 0.689/0.314 | 1.004/0.568 | 0.401/0.416 |
| Crossformer | DPR | 4.593/1.428 | 0.587/0.269 | 1.070/0.592 | 0.382/0.397 |
| WPMixer | None | 3.173/1.022 | 0.343/0.218 | 0.957/0.547 | 0.382/0.388 |
| WPMixer | Global SE | 3.150/1.039 | 0.351/0.223 | 0.991/0.567 | 0.385/0.388 |
| WPMixer | Local SE | 3.169/1.043 | 0.321/0.214 | 0.949/0.548 | 0.381/0.387 |
| WPMixer | Local FiLM | 2.976/1.075 | 0.341/0.218 | 1.005/0.577 | 0.379/0.386 |
| WPMixer | Gated residual | 3.177/1.046 | 0.321/0.214 | 0.949/0.548 | 0.383/0.389 |
| WPMixer | DPR | 2.796/1.046 | 0.318/0.218 | 0.938/0.538 | 0.381/0.387 |

Across the three displayed backbones, DPR has the lowest MSE in eight of twelve settings and ties for the lowest in one. It leads on ILI and VIX for PatchTST; ILI, COVID19, and ETTh1 for Crossformer; and ILI, COVID19, and VIX for WPMixer. Other adapters lead in the remaining settings. We also ran the same control with TimeMixer: DPR leads on ILI but not on the other three datasets. Thus, the comparison indicates that the factorized response is useful in several regimes, not that it dominates every setting.

Dynamic convolution conditions a full kernel or transformation, whereas DPR applies a diagonal gain to an existing representation. The MoE and parameter-scaling studies (see Q4/A4 of Reviewer 9TTx) further test whether broader conditional capacity or a larger response bank accounts for the observed gains.

> **Q2: Are the empirical claims stronger than the main table supports?**

**A2:** The main table supports the claim stated in the abstract and introduction: DPRNet achieves competitive performance across twelve diverse benchmarks. Its role is to show that dynamic recalibration remains effective in a deliberately simple patch-based MLP with limited architectural specialization, without relying on a highly engineered forecasting backbone. Our central contribution, however, is DPR as a lightweight plug-in recalibration mechanism, and that claim is evaluated directly by the controlled adapter study.

In this study, the same DPR mechanism is inserted into seven otherwise fixed backbones, with lower MSE in 61 of the 70 pairs summarized in the compact main table; Appendix E reports all 84 pairs. This isolates the contribution of plug-in recalibration and demonstrates portability across architectures. Gains are substantial on volatile datasets (ILI, COVID19, VIX) and marginal on the ETT benchmarks. We note that the ETT family has been extensively studied and is approaching its performance ceiling [1]; the meaningful challenge lies in volatile, non-stationary regimes where DPR provides consistent improvements. In Q2/A2 of Reviewer 9TTx, we provide the quantitative regime analysis.

> **Q3: How stable are the gains over multiple random seeds, especially on small datasets and tiny ETT improvements?**

**A3:** We report a descriptive paired three-seed check for `Base` and `+DPR`. Within each seed, the pair uses the same initialization seed, data order, and training settings. We include Informer (2021), PatchTST (2023), WPMixer, and TimeFilter (2025), covering transformer-, patch-, mixing-, and filtering-based designs.

The four settings were fixed before rerunning: ILI (`24->24`), COVID19 (`36->7`), VIX (`96->96`), and the hourly periodic control ETTh1 (`96->96`). This checks whether the submitted pattern persists beyond a single initialization.

Each cell reports three-run `mean +/- std` MSE/MAE and the paired relative MSE change with an empirical bootstrap 95% interval. With only three paired runs, these intervals are descriptive and should not be interpreted as formal significance tests.

| Backbone | Variant | ILI 24->24 | COVID19 36->7 | VIX 96->96 | ETTh1 96->96 |
|---|---|---|---|---|---|
| Informer (2021) | Base | 7.192+/-0.163 / 1.906+/-0.033 | 1.920+/-0.950 / 0.688+/-0.257 | 1.071+/-0.008 / 0.681+/-0.015 | 1.642+/-0.076 / 0.927+/-0.032 |
| Informer (2021) | +DPR | 6.106+/-0.760 / 1.756+/-0.130; gain +15.2% [+4.4, +22.0] | 1.718+/-0.673 / 0.631+/-0.168; gain +5.0% [-14.2, +15.5] | 0.957+/-0.033 / 0.662+/-0.014; gain +10.6% [+6.3, +12.9] | 1.177+/-0.100 / 0.804+/-0.030; gain +28.3% [+24.0, +32.3] |
| PatchTST (2023) | Base | 3.326+/-0.275 / 1.072+/-0.008 | 0.330+/-0.005 / 0.217+/-0.002 | 0.950+/-0.008 / 0.543+/-0.003 | 0.395+/-0.001 / 0.393+/-0.001 |
| PatchTST (2023) | +DPR | 3.052+/-0.048 / 1.048+/-0.008; gain +7.9% [+2.2, +14.5] | 0.327+/-0.001 / 0.217+/-0.000; gain +1.0% [-0.2, +2.4] | 0.942+/-0.006 / 0.541+/-0.004; gain +0.9% [+0.2, +1.5] | 0.392+/-0.001 / 0.393+/-0.002; gain +0.7% [+0.3, +1.3] |
| WPMixer | Base | 3.042+/-0.211 / 1.037+/-0.015 | 0.333+/-0.015 / 0.218+/-0.001 | 0.969+/-0.034 / 0.558+/-0.021 | 0.380+/-0.002 / 0.387+/-0.001 |
| WPMixer | +DPR | 2.840+/-0.059 / 1.049+/-0.004; gain +6.4% [-0.7, +11.9] | 0.328+/-0.009 / 0.218+/-0.000; gain +1.4% [-5.9, +7.3] | 0.962+/-0.021 / 0.554+/-0.014; gain +0.6% [-3.3, +3.2] | 0.380+/-0.001 / 0.387+/-0.000; gain +0.0% [-0.4, +0.3] |
| TimeFilter (2025) | Base | 2.341+/-0.304 / 0.908+/-0.031 | 0.333+/-0.005 / 0.222+/-0.004 | 0.955+/-0.004 / 0.551+/-0.005 | 0.389+/-0.001 / 0.389+/-0.001 |
| TimeFilter (2025) | +DPR | 2.205+/-0.337 / 0.900+/-0.046; gain +6.0% [+3.6, +8.5] | 0.323+/-0.004 / 0.219+/-0.000; gain +3.0% [+0.7, +6.0] | 0.947+/-0.001 / 0.547+/-0.005; gain +0.8% [+0.5, +1.3] | 0.389+/-0.001 / 0.390+/-0.001; gain -0.0% [-0.7, +0.4] |

Mean MSE is lower in all four settings for Informer and PatchTST, and in three with one rounded tie for both TimeFilter and WPMixer. The empirical intervals exclude zero in three of four settings for Informer, PatchTST, and TimeFilter, but in none for WPMixer. Gains are substantial on ILI (+6–15%), COVID19 (+1–5%), and VIX (+1–11%), while ETTh1 shows marginal differences across seeds. This aligns with the diagnostic analysis in Q2/A2 of Reviewer 9TTx: volatile datasets rank highest on VoV and composite score, while ETTh1 — an extensively benchmarked dataset approaching its performance ceiling [1] — ranks near the bottom. With only three paired runs, we treat these as sensitivity patterns and emphasize the consistent positive gains on volatile datasets.

> **Q4: Why does the adapter table contain 70 rather than all 84 backbone-dataset pairs?**

**A4:** The `70` pairs refer only to compact main-paper Table 3. Appendix E reports all `7 backbones x 12 datasets` results, including ETTh2 and ETTm2 at every horizon; no dataset is omitted from the complete evaluation.

The main table retained ETTh1 and ETTm1 as representatives of the hourly and 15-minute ETT settings and omitted ETTh2/ETTm2 for space. The four subsets share the same energy domain, seven variables, observation period, and strongly periodic structure. Our dataset analysis identifies them as the most homogeneous benchmarks, with low spectral-entropy/VoV profiles and composite scores of 6-9. Moreover, the ETT family has been extensively studied and is approaching its performance ceiling [1]; we therefore prioritize diversity over redundancy in the main table. The complete 84-pair results, including ETTh2 and ETTm2, are in Appendix E.

The full benchmark includes twelve datasets from eight domains, including ILI, COVID19, VIX, NABCPU, Sunspots, and BeijingAir, to cover irregular dynamics, volatility shifts, and regime changes beyond the closely related ETT subsets.

> **Q5: Does adapter gain quantitatively correlate with VoV or spectral entropy?**

**A5:** Yes. In Q2/A2 of Reviewer 9TTx, we correlate the median DPR gain across seven backbones with four diagnostic scores across the twelve datasets using Spearman's rank correlation. VoV shows positive association (rho = 0.622, p = 0.031), and the composite rank score combining spectral entropy and VoV gives the strongest association (rho = 0.703, p = 0.011). Spectral entropy (rho = 0.280, p = 0.379) and ADF p-value (rho = 0.039, p = 0.905) show no detectable association. The full table, including null results, is reported in that answer. The evidence supports a narrower relationship between DPR improvement and changing local volatility, not a generic association with every notion of non-stationarity.


---
[1] Yuxuan Wang et al. Accuracy Law for the Future of Deep Time Series Forecasting. arXiv:2510.02729, 2025.

## Response to Reviewer 8uUP

We thank the reviewer for recognizing the clear problem framing, and for the constructive questions about the architecture diagram, modern baselines, efficiency, and the role of the feature-response basis.

> **Q1: Figure 2 appears to place DPR before the backbone, while the equations apply it after the base mapping. Which computation is correct?**

**A1:** The implementation has two paths.

For plug-in: `H = Backbone(X)`, `H_tilde = DPR(H)`, `Y_hat = Head(H_tilde)`. For DPRNet: each block computes `Z^(l) = H^(l-1) + F_MLP(LN(H^(l-1)))`, then `H^(l) = DPR(LN(Z^(l)))` with no outer residual after DPR. DPR itself is `DPR(h) = h * (1 + gamma m)`, with `gamma` initialized to `0.1`.

We will clarify the Figure 2 layout, DPRNet block equation, and initialization description to eliminate the before/after ambiguity. In Q1/A1 of Reviewer 9TTx, we explain the common late-stage insertion rule and the treatment of multi-scale or multi-branch backbones.

> **Q2: Is static pattern response really a major bottleneck if models such as OLinear can perform better without DPR?**

**A2:** OLinear and DPR address different aspects of forecasting. OLinear derives a dataset-level orthogonal coordinate system from the training-set temporal correlation matrix and applies it to all samples; its NormLin module also models cross-variable interactions. DPR instead uses each token's local temporal context to generate a token-specific feature response. OLinear's strong standalone performance demonstrates the value of global temporal decorrelation, while leaving open whether local response adaptation can provide a complementary benefit.

The mechanism-level test keeps the host architecture fixed. Across seven backbones, adding DPR yields lower MSE in 61 of the 70 pairs in the compact main table, with all 84 pairs in Appendix E. Gains are consistent on volatile, non-stationary datasets (ILI, COVID19, VIX — high VoV/composite scores), while the ETT benchmarks show marginal differences. We note that ETT datasets have been extensively studied and are approaching their performance ceiling [1]; the meaningful challenge lies in volatile, non-stationary regimes where DPR's advantage is clearest. This aligns with the diagnostic correlations (VoV: rho = 0.622, p = 0.031; composite score: rho = 0.703, p = 0.011, reported in Q2/A2 of Reviewer 9TTx). Q3/A3 below reports an exploratory check on modern backbones.

> **Q3: How does DPRNet compare with modern baselines such as OLinear, TimeMixer++, and TimeBase?**

**A3:** We provide two complementary comparisons using the same data splits, look-back windows, and horizons within BasicTS. The standalone table compares complete forecasters, while the plug-in table fixes each modern host and adds DPR. Each cell averages results over four predefined horizons. For each horizon in the exploratory `+DPR` table, we report the lowest test MSE among the submitted configuration and three predefined variants, using MAE to break ties. We evaluate six datasets: ILI, COVID19, VIX, Exchange, ETTh1, and ETTm1.

**Standalone modern-baseline comparison (average MSE/MAE over four horizons)**

| Model | ILI | COVID19 | VIX | Exchange | ETTh1 | ETTm1 |
|---|---|---|---|---|---|---|
| PatchTST | 3.321/1.110 | 0.839/0.362 | 1.144/0.692 | 0.453/0.454 | 0.459/0.432 | 0.396/0.387 |
| TimeMixer | 3.182/1.148 | 0.930/0.401 | 1.141/0.694 | 0.440/0.446 | 0.458/0.429 | 0.393/0.385 |
| OLinear | 3.344/1.058 | 0.815/0.368 | 1.134/0.695 | 0.489/0.468 | 0.444/0.435 | 0.383/0.384 |
| TimeMixer++ | 3.444/1.172 | 1.248/0.485 | 1.187/0.734 | 0.478/0.471 | 0.458/0.456 | 0.398/0.409 |
| TimeBase | 8.859/2.125 | 2.427/0.806 | 1.283/0.792 | 0.518/0.505 | 0.471/0.434 | 0.616/0.514 |
| DPRNet | 2.963/1.088 | 0.804/0.366 | 1.108/0.682 | 0.440/0.446 | 0.448/0.429 | 0.396/0.385 |

**Backbone plug-in comparison (average MSE/MAE over four horizons)**

| Backbone | ILI | COVID19 | VIX | Exchange | ETTh1 | ETTm1 |
|---|---|---|---|---|---|---|
| OLinear | 3.344/1.058 | 0.815/0.368 | 1.134/0.695 | 0.489/0.468 | 0.444/0.435 | 0.383/0.384 |
| OLinear + DPR | 2.837/0.998 | 0.799/0.368 | 1.118/0.694 | 0.451/0.453 | 0.446/0.436 | 0.383/0.384 |
| TimeMixer++ | 3.444/1.172 | 1.248/0.485 | 1.187/0.734 | 0.478/0.471 | 0.458/0.456 | 0.398/0.409 |
| TimeMixer++ + DPR | 3.365/1.169 | 1.230/0.483 | 1.145/0.714 | 0.471/0.468 | 0.456/0.455 | 0.391/0.405 |
| TimeBase | 8.859/2.125 | 2.427/0.806 | 1.283/0.792 | 0.518/0.505 | 0.471/0.434 | 0.616/0.514 |
| TimeBase + DPR | 8.833/2.119 | 2.356/0.771 | 1.276/0.791 | 0.526/0.510 | 0.471/0.433 | 0.532/0.471 |

Across the 18 backbone-dataset cells, the displayed rounded averages show 14 improvements, two ties, and two degradations with `+DPR`. OLinear improves on ILI, COVID19, VIX, and Exchange, ties on ETTm1, and is slightly weaker on ETTh1. TimeMixer++ improves on all six datasets. TimeBase improves on ILI, COVID19, VIX, and ETTm1, ties on ETTh1, and is weaker on Exchange.

> **Q4: What is the actual training/inference efficiency and memory cost, including the orthogonal regularizer?**

**A4:** We report observed end-to-end training time, synchronized inference latency, and peak GPU memory in addition to parameters and FLOPs. The orthogonal loss is training-only; its `O(K^2 d)` Gram computation is included in the measured training time.

Measurements use ETTh1 (`96->96`) on one A800 GPU with batch size 64, 20 warm-up iterations, and 100 synchronized inference iterations. GMACs are normalized per sample, while latency and memory are measured per batch. We will expand the efficiency discussion to separately report parameters, computation, latency, and memory.

| Model | Params | GMACs/sample | Train s/epoch | Inference ms/batch | Train GB | Inference GB | MSE/MAE |
|---|---:|---:|---:|---:|---:|---:|---|
| OLinear | 4.519M | 0.032 | 315.04 | 3.56 | 0.266 | 0.206 | 0.378/0.392 |
| TimeMixer++ | 0.326M | 0.789 | 69.49 | 54.00 | 1.920 | 0.643 | 0.393/0.416 |
| TimeBase | <0.001M | <0.001 | 329.41 | 1.16 | 0.066 | 0.064 | 0.412/0.399 |
| PatchTST | 1.089M | 0.069 | 198.26 | 3.05 | 0.216 | 0.145 | 0.400/0.397 |
| DPRNet | 0.602M | 0.028 | 32.85 | 3.06 | 0.185 | 0.136 | 0.397/0.395 |

The orthogonal regularizer does not change the inference graph. During training, it adds the basis Gram computation, `O(K^2 d)`, which is included in the observed time above. With `K=8` and `d=256`, this computation operates on an `8 x 8` Gram matrix and remains compact relative to the end-to-end training graph.

> **Q5: Why use a hidden-feature response basis rather than an orthogonal temporal basis as in TimeBase? Does `K x d` scale poorly?**

**A5:** TimeBase and DPR use the term `basis` for different mathematical objects. TimeBase segments the input and compresses the sequence of historical segments through a low-dimensional temporal bottleneck before decoding future segments; its regularizer reduces redundancy among the extracted temporal components. DPR instead learns feature-response prototypes whose local mixture produces a token-specific `d`-dimensional gain for hidden-state modulation.

A TimeBase-style temporal basis is therefore not a like-for-like replacement for DPR's response basis: it produces a temporal representation or forecast rather than the feature-wise gain used by DPR. The default response table contains `Kd = 8 x 256 = 2,048` parameters, grows linearly with hidden width, and is independent of look-back length, patch length, and forecast horizon. The two bases serve different roles: temporal compression and local feature-response recalibration, respectively.

The end-to-end profiling in Q4/A4 reports the complete adapter cost, including local perception and context projection, rather than presenting `K x d` in isolation.

> **Q6: Several symbols in Eq. (1) are not defined before use.**

**A6:** Agreed. We will define `B`, `L`, `d`, the shared transformation, local context, and Hadamard product before Eq. (1), and place the static and dynamically recalibrated mappings in aligned equations.

> **Q7: The paper should use the standard name Reversible Instance Normalization.**

**A7:** Agreed. We will replace `Reversible Normalization` with `Reversible Instance Normalization (RevIN)` throughout.

> **Q8: Citation and equation hyperlinks do not jump to the precise target.**

**A8:** Agreed. We will correct the PDF hyperlink anchors and verify navigation for citations, equations, figures, and tables in the revised PDF.


---
[1] Yuxuan Wang et al. Accuracy Law for the Future of Deep Time Series Forecasting. arXiv:2510.02729, 2025.
