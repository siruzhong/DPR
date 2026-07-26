# Meta Review of Submission6876 by Area Chair QNFK
Metareview: This paper addresses feature processing in time series forecasting models. Whereas most architectures apply fixed feature transformations to all inputs, real-world time series are significantly more dynamic and heterogenous, motivating the paper's dynamic pattern recalibration (DPR) framework which routes each input token to learned response bases to produce dynamically recalibrated hidden features. All reviewers agree that the motivation of the paper is clear, and the proposed method is relatively simple and practical. However, several weaknesses were brought up; the following concerns, if appropriately addressed, would increase the likelihood of this paper being accepted:
- Clarifying the novelty of the method compared to existing conditional modulation mechanisms such as FiLM and SE-style calibration
- Improved empirical results, including error bars, more comprehensive backbone-dataset coverage, and more modern baseline methods
- More direct evidence demonstrating that the described "static pattern response" is actually an important bottleneck in time series forecasting
- A discussion of the computational tradeoffs

# Official Review of Submission6876 by Reviewer 9TTx
Summary:
This paper studies a common limitation in deep time series forecasting: most backbones apply fixed feature transformations to all temporal tokens, even when local dynamics change over time. The authors call this issue the "static pattern response."

To address it, the paper proposes Dynamic Pattern Recalibration (DPR), a lightweight adapter that follows a Perceive-Route-Modulate design. DPR extracts local context, softly routes each token over a learned basis of response patterns, and uses the resulting modulation vector to recalibrate hidden features. The method is evaluated both as a standalone minimalist model, DPRNet, and as a plug-in adapter for several forecasting backbones. Experiments across 12 datasets show consistent gains, especially on volatile and non-stationary time series.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
I found this to be a clear and convincing paper. The main idea is conceptually clean and practically useful: instead of only changing how temporal information is aggregated, the model should also adapt how each token's features are transformed under different local dynamics.

The main strengths are:

The motivation is strong. The "static pattern response" framing captures a real limitation of many forecasting backbones.
DPR is lightweight and easy to integrate. The Perceive-Route-Modulate pipeline is intuitive and does not require a major redesign of the host model.
The empirical evaluation is broad. The paper tests both DPRNet and DPR as an adapter across multiple architectures and datasets.
The comparison with parameter scaling and MoE-style routing is helpful. It supports the claim that the benefit comes from adaptive recalibration rather than simply adding parameters.
The paper is well organized and mostly easy to follow. The distinction between attention, MoE, and DPR is clear.
The main weaknesses are relatively minor:

Some claims about the generality of the static pattern response are mostly empirical. The appendix analysis is useful, but it is more explanatory than predictive, which is acceptable for this type of methodological paper.
The relation to earlier conditional modulation methods, such as FiLM or SE-style recalibration, could be discussed a bit more directly.
The adapter insertion strategy is described in the appendix, but the main paper could say more about why these insertion points were chosen.
The limitations section is reasonable, though it could more explicitly discuss cases where local endogenous signals are insufficient.
Overall, I think the paper makes a solid and practically useful contribution to time series forecasting. The method is not overly complex, the experiments are broad, and the results support the central claim.

Quality: 4: excellent
Clarity: 4: excellent
Significance: 4: excellent
Originality: 3: good
Questions:
Could the authors clarify how the DPR insertion points were chosen for different backbones? Were earlier-layer or multi-layer insertions considered?
The paper argues that DPR is especially useful for locally non-stationary datasets. Can the authors comment on whether the improvement magnitude correlates with the non-stationarity diagnostics reported in the appendix?
The comparison with MoE is helpful. Could the authors clarify whether DPR should be viewed mainly as feature-wise modulation rather than expert selection, and what this means for expressiveness?
How should DPR behave when the useful forecasting signal comes mainly from exogenous variables rather than the observed target history?
These questions are mainly intended to clarify the scope and interpretation of the method, especially the extent to which the adapter behavior is consistent across different backbones and dataset regimes.

Limitations:
Yes. The paper discusses the main limitation that DPR relies on endogenous temporal signals and may not anticipate external shocks before they appear in the observed sequence. A bit more discussion of exogenous-variable settings would be useful.

Rating: 5: Accept: Technically solid paper, with high potential value on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
NA

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes

# Official Review of Submission6876 by Reviewer Lms2
Summary:
The paper proposes Dynamic Pattern Recalibration (DPR), a lightweight adapter for time-series forecasting models. DPR follows a perceive-route-modulate design: multi-scale depthwise convolutions extract local context, a soft routing distribution assigns each token to learned response bases, and a Hadamard modulation vector recalibrates hidden features. The authors instantiate the mechanism in a simple patch-based MLP model called DPRNet and also insert DPR into several existing backbones. The evaluation covers twelve datasets, adapter studies across seven backbones, parameter-scaling comparisons, ablations, sensitivity analysis, efficiency plots, and qualitative routing visualizations.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths
The adapter is simple and practically appealing. A token-level modulation mechanism with identity initialization can be inserted into existing backbones without changing their core structure.
The paper evaluates beyond the most common long-horizon forecasting benchmarks. Including datasets such as ILI, COVID19, VIX, NABCPU, and Sunspots is valuable for testing irregular and volatile dynamics.
The backbone-agnostic experiment and ablations are useful. Applying DPR to several backbones, and testing multi-scale perception, orthogonal regularization, identity initialization, and routing type, gives a reasonable first check of the mechanism.
Main Weaknesses
The novelty relative to existing conditional modulation mechanisms is not fully established. DPR is close in spirit to FiLM, SE-style channel recalibration, dynamic convolution, gated residual adapters, and other feature-wise modulation methods. The related-work section mentions some of these, but the paper needs a sharper technical distinction and stronger empirical baselines using local conditional modulation. As written, the main novelty is framed through the term "static pattern response," but the actual mechanism is a familiar local gating/modulation design adapted to forecasting.

The main empirical claims are stronger than the table supports. DPRNet is clearly competitive and strong on volatile datasets, but it is not uniformly best across the twelve benchmarks. It trails or roughly ties the best methods on Weather, BeijingAir, Sunspots, ETTm1, and ETTm2. The paper's narrative sometimes reads as if DPRNet establishes a general new frontier, while the actual results are more nuanced: the adapter is most useful in certain non-stationary or volatile regimes and less decisive on regular periodic datasets.

There are no error bars or significance tests. The checklist states that error bars are not reported, and this is a real weakness here because many gains are small. For example, several adapter improvements in Table 3 are on the order of 0.001-0.005 MSE/MAE. Without multiple seeds, it is hard to know whether the 61/70 improvement count is stable.

The backbone-agnostic study does not cover all twelve datasets, despite the broader dataset claim. Table 3 reports 70 backbone-dataset pairs, which corresponds to seven backbones on ten datasets, not twelve. The paper should explain which datasets are omitted and why. Since the adapter's value is claimed to depend on non-stationarity, selectively missing datasets can affect the conclusion.

Quality: 3: good
Clarity: 2: not good
Significance: 2: not good
Originality: 2: not good
Questions:
Can the authors add local FiLM, SE-style, or gated residual adapter baselines under the same parameter budget?
Why are only 70 backbone-dataset pairs reported in the adapter table if the benchmark suite has twelve datasets?
How stable are the reported gains over 3-5 random seeds, especially for the small datasets and for the tiny improvements on ETT-style benchmarks?
Does the adapter gain correlate quantitatively with volatility-of-volatility or spectral entropy?
Limitations:
yes

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
No.

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes

# Official Review of Submission6876 by Reviewer 8uUP
Summary:
The paper introduces Dynamic Pattern Recalibration (DPR), a backbone-agnostic mechanism, to resolves the so-called static pattern response issue. Specifically, DPR senses dynamics through multi-scale convolutions on temporal patches, computes a soft-routing distribution over a learned feature basis, and applies token-level pattern recalibration via a residual Hadamard product.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths:
The paper is well-structured and clearly identifies the static pattern response issue—the limitation of globally shared, static feature transformations under volatile local dynamics. It is generally easy to follow.
Weaknesses:
Major Concerns
In Figure 2, the "DPR Adapter" block is positioned above the "Backbone" block. However, based on the description in Section 3 and Equations (2) and (3), the adapter is applied after the static base mapping (the backbone's encoder) to modify its output features.

It remains debatable whether the assumed "static pattern response" is indeed a major bottleneck in time series forecasting. Existing forecasters which ignore this issue, , e.g., OLinear, could deliver better performance (with competitive computational efficiency) than the proposed DPRNet.

The proposed recalibration mechanism (especially with the introduction of Orthogonal Regularization) increases the training difficulty. Despite this overhead, DPRNet does not consistently outperform current state-of-the-art models. For instance, on several critical datasets such as ETTh1, ETTh2, ETTm1, ETTm2, and Exchange, DPRNet falls behind more modern SOTA baselines such as OLinear and TimeMixer++. The baselines used in the comparison are somewhat old. The authors should evaluate their approach against more modern and representative methods.

The authors should provide a clear comparative table displaying the training and inference efficiency (e.g., training time per epoch, inference latency, memory footprint) of DPRNet compared with standard baselines. As the patch-based forecaster, DPRNet could suffer from heavy computational complexity.

In DPR, learning the feature basis directly over the feature dimension d can cause a sharp, undesirable increase in the number of learnable parameters. Is this approach optimal? Following the methodology of recent minimalist forecasting works (such as TimeBase), learning orthogonal temporal patches (or basis components in the time domain) might be a more intuitive, structured, and computationally efficient design choice. The authors should discuss or empirically justify why learning a feature-dimension basis is superior.

Minor Concerns
Several mathematical notations are introduced without prior definition, e.g., Eq. (1), which hinders readability.
In Section 3.1 (Page 3, Line 112 block), it should be Reversible Instance Normalization.
The PDF's hyperlink functionality seems slightly misconfigured. Clicking on citation numbers or equation markers only jumps to the corresponding page itself, rather than navigating directly to the specific reference entry or equation line.
Quality: 3: good
Clarity: 3: good
Significance: 2: not good
Originality: 2: not good
Questions:
See weaknesses.

Limitations:
yes

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
None

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes