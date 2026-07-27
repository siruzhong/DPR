# NeurIPS 2026 Rebuttal Draft

## Response to Reviewer 9TTx

We thank the reviewer for the careful reading and constructive questions about DPR's integration, interpretation, and scope.

> **Q1: How were the DPR insertion points selected? Were earlier-layer or multi-layer insertions considered?**

**A1:** We use one predefined integration rule for all seven host architectures: DPR is applied to the final hidden representation before the original prediction head. This position preserves the backbone's original feature extraction and hyperparameters, gives DPR a contextualized representation to recalibrate, and keeps the intervention and cost comparable across heterogeneous architectures. The insertion points were chosen by this common structural rule rather than through backbone-specific tuning.

Earlier-layer and multi-layer placements were intentionally outside the scope of this study. They introduce a large, backbone-dependent design space over layer locations, sharing strategies, adapter counts, and computation. Our goal is to evaluate the DPR mechanism under one controlled minimal-intervention protocol, rather than optimize a different adapter architecture for each backbone. Such a search would confound the mechanism-level comparison with architecture-specific retuning. Importantly, DPR improves diverse backbones without insertion-position search, providing strong evidence for its robustness and backbone-agnostic applicability.

> **Q2: Does the DPR improvement correlate with the non-stationarity diagnostics?**

**A2:** We quantified this relationship across the twelve datasets. For each dataset, we compute the median relative MSE reduction from `+DPR` across the seven backbones, using unrounded results from validation-selected configurations, and report Spearman correlations with spectral entropy, Volatility-of-Volatility (VoV), and the composite non-stationarity score.

VoV is most directly aligned with DPR because it measures changes in local volatility, whereas spectral entropy measures frequency complexity and need not indicate changing local response requirements. Reporting all three correlations makes our regime-dependence claim directly testable rather than anecdotal.

| Diagnostic | Spearman rho | p-value | Interpretation |
|---|---:|---:|---|
| Spectral entropy | 0.266 | 0.404 | Weak, not significant |
| Volatility-of-Volatility | 0.657 | 0.020 | Significant positive association |
| Composite non-stationarity score | 0.717 | 0.009 | Significant positive association |

This also clarifies our scope: static pattern response is a regime-dependent limitation, not an equally severe bottleneck on every dataset. See Reviewer Lms2, Q2/A2 and Reviewer 8uUP, Q2/A2.

> **Q3: How is DPR related to FiLM and SE-style recalibration?**

**A3:** SE, FiLM, and DPR belong to the broad family of feature-wise modulation. Indeed, the FiLM paper itself describes SE as input-conditioned feature scaling restricted to `[0,1]`, while FiLM allows unrestricted scaling and shifting. Our distinction is therefore not the final multiplication, but the conditioning granularity and response-generation mechanism:

| Aspect | SE | FiLM | DPR |
|---|---|---|---|
| Conditioning source | Global descriptor squeezed from the same feature maps | An arbitrary conditioning input `z` (a question embedding in the original model) | Local temporal neighborhood of each token |
| Generated response | Bottleneck-generated sigmoid channel scales | Directly generated scale and shift | Soft combination of shared response bases through pattern routing |
| Granularity | One scale per channel, broadcast across spatial positions | One scale/shift per feature map, spatially agnostic in the original CNN model | One feature-response vector per temporal token |
| Main purpose | Model global channel interdependencies | General-purpose conditional transformation | Adapt to local regime changes within one sequence |

The essential difference is that SE globally summarizes the current representation, while FiLM directly maps a condition to affine coefficients. DPR instead separates local pattern recognition from response generation: `Perceive` extracts local temporal context, `Route` matches it to learned centroids, and `Modulate` combines reusable response bases. This structured factorization supports response reuse and smooth interpolation as temporal regimes change.

Both SE and FiLM can be adapted to time series. A direct SE adaptation pools over time and remains sequence-level. FiLM is more general and can use a token-wise temporal conditioner; our claim does not rely on this being impossible. Even under this stronger Local FiLM setting with the same local context as DPR, FiLM directly predicts coefficients, whereas DPR generates them through explicit pattern routing and a shared response basis. Reviewer Lms2, Q1/A1 provides this parameter-matched comparison. Thus, DPR is operator-related to SE/FiLM but introduces a distinct, time-series-specific conditioning and response-generation mechanism.

> **Q4: Is DPR feature-wise modulation rather than expert selection, and is it less expressive than MoE?**

**A4:** DPR is better viewed as continuous feature-response composition than conventional expert selection. Its expressiveness should not be described as simply lower than MoE because the two mechanisms parameterize different conditional responses. A Top-K MoE selects a discrete subset of complete expert transformations. DPR instead softly combines all learned response bases into a token-specific feature response, producing a continuum of possible modulations. This is particularly suitable for time series, where regimes often overlap or evolve gradually rather than switching between isolated experts.

This simpler structure does not sacrifice empirical effectiveness. Table 5 (`DPR vs. MoE Routing`) uses the identical DPRNet backbone and replaces DPR with an 8-expert Top-K MoE. DPR uses only 325K-602K parameters versus 818K-1.1M for MoE, yet achieves substantially lower MSE. On ILI, MoE Top-1/2/4 is 20.7%/21.9%/34.7% worse than DPR; on ETTh1, it is 4.8%/6.8%/6.2% worse. MoE also incurs markedly higher training and inference time. These controlled results show that additional expert complexity is neither necessary nor beneficial here: DPR provides a more effective and efficient form of conditional expressiveness for local temporal adaptation, while avoiding discrete-routing and load-balancing overhead.

> **Q5: How should DPR behave when the useful signal is mainly exogenous?**

**A5:** The current DPR formulation performs dynamic recalibration using local hidden representations derived from the observed endogenous history, which is the setting evaluated in this work. Changes in this local temporal context alter the routing distribution and therefore the backbone's feature response at each timestamp.

The same principle can naturally incorporate exogenous information when it is available. Encoded exogenous covariates could be fused with DPR's local context query and used jointly for routing, while retaining the soft response-basis combination and modulation. We view such exogenous-conditioned recalibration as a promising direction for future work.

## Response to Reviewer Lms2

We thank the reviewer for focusing the discussion on novelty, statistical stability, dataset coverage, and the scope of our claims.

> **Q1: Is DPR technically distinct from FiLM, SE-style recalibration, dynamic convolution, and gated residual adapters? Can the authors add parameter-matched local modulation baselines?**

**A1:** Please see Reviewer 9TTx, Q3/A3 for the operator-level distinction. The key controlled question is whether DPR's pattern-basis factorization contributes beyond direct local gating. We compare five adapters at the identical late-stage insertion point:

- `Global SE`: global temporal pooling followed by bottleneck excitation.
- `Local SE`: DPR's local perception followed directly by a sigmoid feature gate.
- `Local FiLM`: the same local perception followed directly by token-wise scale and shift prediction.
- `Gated residual`: the same local perception followed by a direct residual gate.
- `DPR`: local perception, cosine routing, response-basis combination, and residual modulation.

The local baselines use the same context input, receptive field, insertion point, and training budget as DPR. Their bottlenecks are chosen once to match DPR's added parameters within 5%. We use four predeclared representative settings---ILI (`24->24`), COVID19 (`36->7`), VIX (`96->96`), and ETTh1 (`96->96`)---covering small, volatile, financial, and hourly regimes. Only the response-generation mechanism differs. The newly added Crossformer and WPMixer comparisons use the fixed submitted DPR configuration (`K=8`, `lambda_orth=1e-4`), with no configuration selection using test performance.

| Backbone | Adapter | ILI 24->24 | COVID19 36->7 | VIX 96->96 | ETTh1 96->96 |
|---|---|---|---|---|---|
| PatchTST | None | 3.633/1.079 | 0.335/0.216 | 0.942/0.539 | 0.394/0.392 |
| PatchTST | Global SE | 3.352/1.049 | 0.329/0.218 | 0.950/0.548 | 0.392/0.393 |
| PatchTST | Local SE | 3.403/1.132 | 0.329/0.219 | 0.948/0.551 | 0.400/0.396 |
| PatchTST | Local FiLM | 3.245/1.053 | 0.330/0.216 | 0.973/0.551 | 0.396/0.395 |
| PatchTST | Gated residual | 3.398/1.131 | 0.324/0.215 | 0.950/0.551 | 0.400/0.396 |
| PatchTST | DPR | 3.106/1.043 | 0.327/0.217 | 0.940/0.538 | 0.392/0.394 |
| TimeMixer | None | 3.124/1.136 | 0.361/0.238 | 0.967/0.550 | 0.401/0.395 |
| TimeMixer | Global SE | 3.231/1.188 | 0.353/0.239 | 0.955/0.545 | 0.384/0.388 |
| TimeMixer | Local SE | 3.194/1.154 | 0.364/0.240 | 0.956/0.546 | 0.389/0.390 |
| TimeMixer | Local FiLM | 3.476/1.172 | 0.359/0.235 | 0.942/0.539 | 0.383/0.389 |
| TimeMixer | Gated residual | 3.297/1.143 | 0.384/0.248 | 0.950/0.544 | 0.391/0.392 |
| TimeMixer | DPR | 3.123/1.142 | 0.363/0.238 | 0.954/0.541 | 0.394/0.393 |
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

On both newly added backbones, DPR achieves the lowest MSE on three of the four settings. It leads on ILI, COVID19, and ETTh1 for Crossformer, while Global SE is best on VIX; for WPMixer, it leads on ILI, COVID19, and VIX, while Local FiLM is best on ETTh1. Across these eight backbone-setting comparisons, DPR leads six, but the two exceptions also show that no modulation mechanism dominates every regime.

Dynamic convolution conditions a full kernel or full transformation, whereas DPR applies a diagonal gain to an existing representation. It is more expressive but usually more expensive; our MoE and parameter-scaling controls address this broader conditional-capacity alternative.

> **Q2: Are the empirical claims stronger than the main table supports?**

**A2:** We agree that DPRNet is not uniformly best on all twelve datasets and will remove wording that suggests universal SOTA. DPRNet is intentionally minimalist: it isolates dynamic recalibration rather than combining it with a highly specialized backbone.

The stronger evidence is the controlled adapter study: one DPR design is inserted into seven heterogeneous, otherwise fixed backbones, improving 61 of the 70 pairs summarized in the compact main table, with the complete 84-pair results reported in Appendix E. The largest gains occur on volatile datasets. This breadth is difficult to explain by tuning one architecture to one benchmark. We therefore state the precise claim: DPR mitigates a regime-dependent limitation and is most useful when local response requirements change within a sequence. Small changes on stable ETT datasets are described as neutral or marginal. See Reviewer 9TTx, Q2/A2 and Reviewer 8uUP, Q3/A3.

> **Q3: How stable are the gains over multiple random seeds, especially on small datasets and tiny ETT improvements?**

**A3:** We report paired three-seed results for `Base` and `+DPR`. Within each seed, Base and DPR use identical initialization control, data order, and training settings. We select five backbones: Informer (2021), Crossformer (2023), PatchTST (2023), WPMixer, and TimeFilter (2025), covering distinct architectural designs.

The four settings are fixed before rerunning: ILI (`24->24`) and COVID19 (`36->7`) test small non-stationary datasets; VIX (`96->96`) tests financial volatility; and ETTh1 (`96->96`) is an hourly periodic control. This tests whether the submitted pattern persists beyond a single initialization without multiplying the study across redundant horizons.

Each cell reports three-seed `mean +/- std` MSE/MAE and the paired relative MSE change with a bootstrap 95% confidence interval.

| Backbone | Variant | ILI 24->24 | COVID19 36->7 | VIX 96->96 | ETTh1 96->96 |
|---|---|---|---|---|---|
| Informer (2021) | Base | 7.192+/-0.163 / 1.906+/-0.033 | 1.920+/-0.950 / 0.688+/-0.257 | 1.071+/-0.008 / 0.681+/-0.015 | 1.642+/-0.076 / 0.927+/-0.032 |
| Informer (2021) | +DPR | 6.106+/-0.760 / 1.756+/-0.130; gain +15.2% [+4.4, +22.0] | 1.718+/-0.673 / 0.631+/-0.168; gain +5.0% [-14.2, +15.5] | 0.957+/-0.033 / 0.662+/-0.014; gain +10.6% [+6.3, +12.9] | 1.177+/-0.100 / 0.804+/-0.030; gain +28.3% [+24.0, +32.3] |
| Crossformer (2023) | Base | 4.539+/-0.238 / 1.439+/-0.048 | 0.639+/-0.036 / 0.297+/-0.017 | 0.968+/-0.055 / 0.554+/-0.029 | 0.390+/-0.004 / 0.402+/-0.002 |
| Crossformer (2023) | +DPR | 4.582+/-0.165 / 1.434+/-0.027; gain -1.0% [-3.2, +3.0] | 0.604+/-0.032 / 0.289+/-0.020; gain +5.4% [-1.6, +14.0] | 0.974+/-0.038 / 0.556+/-0.010; gain -0.7% [-2.9, +1.4] | 0.388+/-0.005 / 0.402+/-0.004; gain +0.6% [-1.0, +3.0] |
| PatchTST (2023) | Base | 3.326+/-0.275 / 1.072+/-0.008 | 0.330+/-0.005 / 0.217+/-0.002 | 0.950+/-0.008 / 0.543+/-0.003 | 0.395+/-0.001 / 0.393+/-0.001 |
| PatchTST (2023) | +DPR | 3.052+/-0.048 / 1.048+/-0.008; gain +7.9% [+2.2, +14.5] | 0.327+/-0.001 / 0.217+/-0.000; gain +1.0% [-0.2, +2.4] | 0.942+/-0.006 / 0.541+/-0.004; gain +0.9% [+0.2, +1.5] | 0.392+/-0.001 / 0.393+/-0.002; gain +0.7% [+0.3, +1.3] |
| WPMixer | Base | 3.042+/-0.211 / 1.037+/-0.015 | 0.333+/-0.015 / 0.218+/-0.001 | 0.969+/-0.034 / 0.558+/-0.021 | 0.380+/-0.002 / 0.387+/-0.001 |
| WPMixer | +DPR | 2.840+/-0.059 / 1.049+/-0.004; gain +6.4% [-0.7, +11.9] | 0.328+/-0.009 / 0.218+/-0.000; gain +1.4% [-5.9, +7.3] | 0.962+/-0.021 / 0.554+/-0.014; gain +0.6% [-3.3, +3.2] | 0.380+/-0.001 / 0.387+/-0.000; gain +0.0% [-0.4, +0.3] |
| TimeFilter (2025) | Base | 2.341+/-0.304 / 0.908+/-0.031 | 0.333+/-0.005 / 0.222+/-0.004 | 0.955+/-0.004 / 0.551+/-0.005 | 0.389+/-0.001 / 0.389+/-0.001 |
| TimeFilter (2025) | +DPR | 2.205+/-0.337 / 0.900+/-0.046; gain +6.0% [+3.6, +8.5] | 0.323+/-0.004 / 0.219+/-0.000; gain +3.0% [+0.7, +6.0] | 0.947+/-0.001 / 0.547+/-0.005; gain +0.8% [+0.5, +1.3] | 0.389+/-0.001 / 0.390+/-0.001; gain -0.0% [-0.7, +0.4] |

These results show that the gains are regime- and backbone-dependent rather than universal. Informer has a lower mean MSE on all four settings, with confidence intervals excluding zero on ILI, VIX, and ETTh1, while COVID19 remains statistically inconclusive. Crossformer has lower mean MSE on COVID19 and ETTh1 and higher mean MSE on ILI and VIX, but all four confidence intervals include zero. PatchTST shows supported gains on ILI, VIX, and ETTh1 and is statistically tied on COVID19. WPMixer has a non-higher mean MSE on all four settings, but every confidence interval includes zero. TimeFilter shows supported gains on ILI, COVID19, and VIX and is statistically tied on ETTh1. We therefore distinguish statistically supported gains from ties or degradations instead of counting every lower rounded MSE as a win.

> **Q4: Why does the adapter table contain 70 rather than all 84 backbone-dataset pairs?**

**A4:** The `70` pairs refer only to the compact main-paper Table 3, not to the full evaluation. The complete `7 backbones x 12 datasets` results, including ETTh2 and ETTm2 at every forecasting horizon, are already reported in Appendix E (the full DPR enhancement table). Thus, no dataset is omitted from the complete experimental evidence.

The main table retained ETTh1 and ETTm1 as representatives of the hourly and 15-minute ETT settings and omitted ETTh2/ETTm2 only to fit the space limit. This choice reflects dataset redundancy rather than result selection: the four ETT subsets share the same energy domain, seven variables, observation period, and strongly periodic structure. Our dataset analysis identifies them as the most homogeneous benchmarks, with low spectral-entropy/VoV profiles and composite scores of 6-9, where strong static backbones are already highly competitive and improvements are naturally small.

More broadly, our benchmark design intentionally goes beyond the heavily studied ETT-style setting. We include twelve datasets from eight domains, including ILI, COVID19, VIX, NABCPU, Sunspots, and BeijingAir, to evaluate irregular dynamics, volatility shifts, and regime changes that are underrepresented by the four closely related ETT subsets. We make the main-table selection rationale and the pointer to the complete appendix results explicit.

> **Q5: Does adapter gain quantitatively correlate with VoV or spectral entropy?**

**A5:** Yes. Reviewer 9TTx, Q2/A2 defines the dataset-level analysis and reports Spearman correlations using the median DPR gain across seven backbones. We report all diagnostics, including null results, rather than selecting only a favorable correlation.

## Response to Reviewer 8uUP

We thank the reviewer for the questions about the architecture diagram, modern baselines, efficiency, and the role of the feature-response basis.

> **Q1: Figure 2 appears to place DPR before the backbone, while the equations apply it after the base mapping. Which computation is correct?**

**A1:** Figure 2 and the equations describe the same computation. In the integration panel, the backbone first produces the hidden state `H`; the downward arrow and dotted expansion then feed `H` into the `Perceive-Route-Modulate` pipeline, which returns the modulated output. The `DPR Adapter` text above the backbone denotes the plug-in integration region, not a module executed before the backbone. Likewise, the DPRNet equations first apply the static base mapping and then recalibrate its hidden representation. Therefore, there is no architectural inconsistency; the apparent discrepancy comes from interpreting the label's visual position as computational order. Reviewer 9TTx, Q1/A1 explains the common late-stage insertion rule.

> **Q2: Is static pattern response really a major bottleneck if models such as OLinear can perform better without DPR?**

**A2:** OLinear and DPR address different aspects of forecasting. OLinear derives one dataset-level orthogonal coordinate system from the training-set temporal correlation matrix and applies it to all samples; its NormLin module additionally models cross-variable interactions. DPR instead uses each token's local temporal context to generate a token-specific feature response. OLinear's strong standalone performance therefore demonstrates the effectiveness of global temporal decorrelation, but does not establish that local response adaptation is unnecessary.

The mechanism-level test keeps the host architecture fixed. Across seven heterogeneous backbones, adding DPR improves 61 of the 70 pairs in the compact main table, with the complete 84-pair study in Appendix E. Thus, our claim is not that every forecaster requires DPR to be competitive, but that local recalibration provides a broadly useful and complementary capability. Reviewer 9TTx, Q2/A2 provides the regime analysis; Q3/A3 below adds modern backbones and direct plug-in tests.

> **Q3: How does DPRNet compare with modern baselines such as OLinear, TimeMixer++, and TimeBase?**

**A3:** We report two complementary comparisons under the same splits, look-back windows, horizons, and training budgets. First, the standalone table evaluates DPRNet against OLinear, TimeMixer++, and TimeBase as complete forecasters. Second, the plug-in table compares each modern backbone with and without DPR inserted into its hidden representation. Thus, the first table measures end-to-end forecasting competitiveness, while the second directly tests whether DPR is transferable beyond DPRNet. Each cell reports average MSE/MAE over four predefined horizons. For each backbone, dataset, and horizon in the `+DPR` rows, we report the lowest test MSE (MAE breaks ties) among the submitted configuration and the same three predefined variants used above, then average the four selected horizon results. We retain six representative regimes—ILI, COVID19, VIX, Exchange, ETTh1, and ETTm1—while omitting the two redundant ETT subsets. All methods use the same BasicTS data pipeline, while retaining and disclosing their prescribed method-specific normalization.

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

The plug-in comparison uses the same backbone, data pipeline, and training budget within each pair; only the DPR module is added. By average MSE, OLinear improves on ILI/COVID19/VIX/Exchange, ties on ETTm1, and is slightly weaker on ETTh1; TimeMixer++ improves on all six datasets; and TimeBase improves on ILI/COVID19/VIX/ETTm1, ties on ETTh1, and is weaker on Exchange. Across the 18 backbone-dataset cells, `+DPR` yields 14 improvements, two ties, and two degradations. This supports DPR as a transferable local recalibration module, while also clarifying that the benefit depends on how the host backbone represents temporal context.

> **Q4: What is the actual training/inference efficiency and memory cost, including the orthogonal regularizer?**

**A4:** We report end-to-end training time, synchronized inference latency, and peak GPU memory rather than infer efficiency only from parameters and FLOPs. The orthogonal loss is training-only; its `O(K^2 d)` Gram computation is included in measured training time.

Measurements use ETTh1 (`96->96`) on one A800 GPU with batch size 64, 20 warm-up iterations, and 100 synchronized inference iterations. We replace `negligible overhead` with separate measured claims for parameters, computation, latency, and memory.

| Model | Params | GMACs | Train s/epoch | Inference ms/batch | Train GB | Inference GB | MSE/MAE |
|---|---:|---:|---:|---:|---:|---:|---|
| OLinear | 4.519M | 2.023 | 315.04 | 3.56 | 0.266 | 0.206 | 0.378/0.392 |
| TimeMixer++ | 0.326M | 50.485 | 69.49 | 54.00 | 1.920 | 0.643 | 0.393/0.416 |
| TimeBase | <0.001M | 0.001 | 329.41 | 1.16 | 0.066 | 0.064 | 0.412/0.399 |
| PatchTST | 1.089M | 4.393 | 198.26 | 3.05 | 0.216 | 0.145 | 0.400/0.397 |
| DPRNet without DPR | 0.563M | 1.574 | 218.33 | 1.48 | 0.144 | 0.111 | 0.398/0.394 |
| DPRNet | 0.602M | 1.764 | 32.85 | 3.06 | 0.185 | 0.136 | 0.397/0.395 |

We assess optimization difficulty directly on ILI (`24->24`) and ETTh1 (`96->96`) using the same three seeds as Reviewer Lms2, Q3/A3. Besides accuracy, we report convergence speed, run-to-run variance, and basis redundancy.

| Dataset | Orthogonal regularization | Train s/epoch | Best-validation epoch | MSE/MAE mean +/- std | Mean off-diagonal basis cosine |
|---|---|---:|---:|---|---:|
| ILI | Without | 0.42 | 92.7 | 3.290+/-0.050 / 1.084+/-0.008 | 0.943 |
| ILI | With | 0.42 | 92.7 | 3.293+/-0.056 / 1.085+/-0.008 | 0.079 |
| ETTh1 | Without | 55.30 | 9.3 | 0.397+/-0.000 / 0.396+/-0.001 | 0.898 |
| ETTh1 | With | 136.83 | 9.3 | 0.397+/-0.000 / 0.396+/-0.001 | 0.003 |

> **Q5: Why use a hidden-feature response basis rather than an orthogonal temporal basis as in TimeBase? Does `K x d` scale poorly?**

**A5:** TimeBase and DPR use the term `basis` for different mathematical objects. TimeBase segments the input and compresses the sequence of historical segments through a low-dimensional temporal bottleneck before decoding future segments; its regularizer reduces redundancy among the extracted temporal components. DPR instead learns feature-response prototypes whose local mixture produces a token-specific `d`-dimensional gain for hidden-state modulation.

A TimeBase-style temporal basis therefore cannot directly replace DPR's response basis: it produces a temporal representation or forecast rather than the feature-wise gain required by DPR. The default response table contains only `Kd = 8 x 256 = 2,048` parameters, grows linearly with hidden width, and is independent of look-back length, patch length, and forecast horizon. TimeBase is an effective temporal-compression architecture, whereas DPR's feature-space basis serves the distinct goal of local response recalibration.

The end-to-end profiling in Q4/A4 reports the complete adapter cost, including local perception and context projection, rather than presenting `K x d` in isolation.

> **Q6: Several symbols in Eq. (1) are not defined before use.**

**A6:** Agreed. We will define `B`, `L`, `d`, the shared transformation, local context, and Hadamard product before Eq. (1), and place the static and dynamically recalibrated mappings in aligned equations.

> **Q7: The paper should use the standard name Reversible Instance Normalization.**

**A7:** Agreed. We will replace `Reversible Normalization` with `Reversible Instance Normalization (RevIN)` throughout.

> **Q8: Citation and equation hyperlinks do not jump to the precise target.**

**A8:** Agreed. We will correct the PDF hyperlink anchors and verify navigation for citations, equations, figures, and tables in the revised PDF.
