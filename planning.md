# Research Plan: Paradigm-Level Uncertainty in Language Models

## Research Question

**Can we distinguish paradigm-level uncertainty from path-level uncertainty in language models through activation pattern analysis?**

## Background and Motivation

Language models exhibit uncertainty in their predictions, but not all uncertainty is the same. Current research focuses on:
- Confidence calibration
- Factual uncertainty
- Ambiguity handling

However, a deeper question remains: **Does uncertainty sometimes stem from competing conceptual frameworks (paradigms) rather than just alternate reasoning paths?**

**Why this matters:**
- Understanding uncertainty sources improves model reliability
- Paradigm conflicts may require different interventions than path uncertainty
- Could inform better uncertainty quantification methods
- Relevant for high-stakes domains (medical, legal, scientific reasoning)

**Example:**
- **Path-level uncertainty**: "What is 15% of 80?" (can solve via fraction or decimal methods)
- **Paradigm-level uncertainty**: "Is consciousness computational?" (mechanistic vs. emergentist paradigms)

## Hypothesis Decomposition

**Main Hypothesis:**
Paradigm-level uncertainty produces distinct activation patterns compared to path-level uncertainty.

**Sub-hypotheses:**

1. **H1: Persistence Hypothesis**
   - Paradigm conflicts create more persistent activation signatures across layers
   - Path uncertainty resolves earlier in the network
   - Measurable: Layer-wise variance of hidden states

2. **H2: Spatial Distribution Hypothesis**
   - Paradigm conflicts produce globally distributed activations
   - Path uncertainty is more localized to specific attention heads/neurons
   - Measurable: Activation entropy, spatial clustering

3. **H3: Temporal Stability Hypothesis**
   - Paradigm conflicts show higher variance across multiple forward passes
   - Path uncertainty is more stable across runs
   - Measurable: Cross-run activation similarity

## Proposed Methodology

### Overall Approach

**Three-condition experimental design:**
1. **Control**: High-confidence questions (no significant uncertainty)
2. **Path-level**: Questions with multiple valid reasoning paths (same framework)
3. **Paradigm-level**: Questions with embedded framework conflicts

**Analysis strategy:**
- Extract hidden states from transformer layers
- Compare activation patterns across conditions
- Statistical tests to validate hypotheses
- Visualization to illustrate differences

### Rationale

**Why activation analysis?**
- Recent work shows confidence/uncertainty encoded in hidden states (LLM Factoscope, 2024)
- Activation patterns reveal internal model processing beyond output tokens
- Middle-to-late layers most informative (per literature)

**Why GPT-2?**
- Full access to internal activations (no API black box)
- Well-studied for interpretability (TransformerLens support)
- CPU-compatible (meets compute constraints)
- Sufficient capacity to exhibit uncertainty

**Why MMLU-derived questions?**
- Established benchmark with ground truth
- Multiple-choice format enables clear correctness signal
- Diverse topics for creating paradigm conflicts
- Manageable size for time constraints

## Experimental Steps

### Step 1: Environment Setup (10 minutes)
- Create isolated virtual environment (uv venv)
- Install dependencies: transformers, transformer_lens, datasets, sklearn, matplotlib
- Download GPT-2 model (gpt2-medium, 355M params)
- Verify CPU execution

### Step 2: Dataset Construction (15 minutes)

**Task 2.1: Load MMLU subset**
- Select 30 questions from 3 domains (10 each): STEM, humanities, social sciences
- Filter for questions likely to have clear vs. uncertain answers

**Task 2.2: Create experimental conditions**

*Control (10 questions):*
- High-confidence, factual questions
- Single clear answer
- Example: "What is the chemical symbol for gold? (A) Au (B) Ag (C) Fe (D) Cu"

*Path-level uncertainty (10 questions):*
- Questions solvable via multiple methods
- Same conceptual framework
- Example: "If a store offers 20% off a $50 item, what's the final price? (Can use multiplication or subtraction)"

*Paradigm-level uncertainty (10 questions):*
- Embed competing frameworks in question or context
- Example: "From a [behavioral economics / rational choice] perspective, how do people make decisions under uncertainty?"
- Modify MMLU questions to include paradigm framing

**Task 2.3: Validation**
- Manual review of questions
- Ensure paradigm conflicts are genuine
- Balance difficulty across conditions

### Step 3: Activation Extraction (20 minutes)

**Task 3.1: Model setup**
```python
import transformer_lens as tl
model = tl.HookedTransformer.from_pretrained("gpt2-medium")
```

**Task 3.2: Forward passes with hooks**
- For each question:
  - Tokenize input
  - Run forward pass with hooks to extract hidden states
  - Capture activations at layers: 0 (early), 12 (middle), 23 (late)
  - Store: (question_id, condition, layer, token_position, activation_vector)

**Task 3.3: Multiple runs**
- Run 3 times with different random seeds
- Enables measuring cross-run stability

### Step 4: Activation Analysis (30 minutes)

**Task 4.1: Persistence Analysis (H1)**
- Compute layer-wise variance of hidden states
- Compare variance trends across conditions
- Statistical test: ANOVA or t-test comparing path vs. paradigm

**Task 4.2: Spatial Distribution Analysis (H2)**
- Compute activation entropy per layer
- Measure spatial clustering (PCA on activations)
- Compare cluster spread across conditions

**Task 4.3: Stability Analysis (H3)**
- Compute cosine similarity of activations across runs
- Compare stability: control > path > paradigm (predicted)

**Task 4.4: Attention Pattern Analysis**
- Extract attention weights
- Compute attention head disagreement (entropy)
- Compare across conditions

### Step 5: Statistical Validation (15 minutes)

**Tests:**
- Independent t-tests: Path vs. Paradigm for each metric
- Effect sizes: Cohen's d
- Significance level: α = 0.05
- Multiple comparison correction: Bonferroni

**Metrics summary table:**
| Metric | Control | Path | Paradigm | p-value | Effect size |
|--------|---------|------|----------|---------|-------------|
| Layer-wise variance | | | | | |
| Activation entropy | | | | | |
| Cross-run stability | | | | | |
| Attention disagreement | | | | | |

### Step 6: Visualization (15 minutes)

**Plot 1: Layer-wise variance trends**
- Line plot: Layer (x) vs. Variance (y), separate lines per condition
- Shows persistence across depth

**Plot 2: PCA activation space**
- Scatter plot: PC1 vs. PC2, colored by condition
- Shows spatial distribution differences

**Plot 3: Stability heatmap**
- Heatmap: Cross-run similarity per condition
- Shows temporal stability

**Plot 4: Attention head disagreement**
- Box plots: Disagreement per condition
- Shows internal conflict

## Baselines

1. **Random baseline**: Shuffle activations, compute metrics
   - Validates that patterns aren't noise

2. **Control condition**: High-confidence questions
   - Establishes floor for uncertainty

3. **Path-level uncertainty**: Established concept
   - Distinguishes paradigm effects from general uncertainty

## Evaluation Metrics

### Primary Metrics

1. **Layer-wise Variance (H1)**
   - Measures: How much activations change across layers
   - Interpretation: Higher = more persistent processing
   - Expected: Paradigm > Path > Control

2. **Activation Entropy (H2)**
   - Measures: Spread of activation values
   - Interpretation: Higher = more distributed
   - Expected: Paradigm > Path > Control

3. **Cross-run Cosine Similarity (H3)**
   - Measures: Stability across runs
   - Interpretation: Lower = less stable
   - Expected: Control > Path > Paradigm

4. **Attention Head Disagreement (exploratory)**
   - Measures: Entropy of attention distributions
   - Interpretation: Higher = more internal conflict
   - Expected: Paradigm > Path

### Why These Metrics?

- **Persistence** captures temporal extent of uncertainty processing
- **Distribution** captures spatial extent (localized vs. global)
- **Stability** captures robustness of representation
- All are computable from hidden states (no manual annotation needed)

## Statistical Analysis Plan

**Comparisons:**
- Path vs. Paradigm (primary)
- Control vs. Path (validation)
- Control vs. Paradigm (validation)

**Tests:**
- Independent samples t-test (two-tailed)
- Effect size: Cohen's d
- Significance: α = 0.05, Bonferroni correction for 4 metrics

**Sample size:**
- 10 questions per condition = 30 total
- Multiple tokens per question (~20-50)
- Multiple layers (3)
- Multiple runs (3)
- Effective n > 100 data points per condition

**Power analysis:**
- Small sample, so focus on effect sizes
- Large effects (d > 0.8) detectable with n=10

## Expected Outcomes

### If Hypothesis Supported:

1. **Persistence**: Paradigm shows 20-40% higher layer-wise variance than Path
2. **Distribution**: Paradigm shows 15-30% higher activation entropy
3. **Stability**: Paradigm shows 10-25% lower cross-run similarity
4. **Attention**: Paradigm shows higher attention head disagreement

**Interpretation**: Paradigm-level uncertainty is a distinct phenomenon with unique computational signature

### If Hypothesis Rejected:

1. **No significant differences** between Path and Paradigm
2. **Both differ from Control** but not from each other

**Interpretation**: Uncertainty is a unitary phenomenon; "paradigm-level" is not computationally distinct

### Partial Support:

1. **Some metrics show effects**, others don't
2. Example: Persistence differs but distribution doesn't

**Interpretation**: Paradigm uncertainty has some unique properties but overlaps with path uncertainty

## Timeline and Milestones

| Phase | Duration | Cumulative | Milestone |
|-------|----------|------------|-----------|
| Phase 0 (Complete) | 30 min | 30 min | Resource research done |
| Phase 1 (Current) | 15 min | 45 min | Planning complete |
| Phase 2 | 10 min | 55 min | Environment ready |
| Phase 3 | 45 min | 100 min | Code implemented |
| Phase 4 | 60 min | 160 min | Experiments complete |
| Phase 5 | 30 min | 190 min | Analysis done |
| Phase 6 | 20 min | 210 min | REPORT.md written |

**Total: 3.5 hours** (flexible, can compress to fit 1-hour constraint if needed)

**Critical path:**
- Dataset construction → Activation extraction → Analysis
- Can run analysis while experiments running (parallel)

**Time-saving strategies:**
- Use 10 questions per condition (not 30)
- Test 2 layers (not 3) if needed
- Use GPT-2 small (124M) instead of medium if speed issues

## Potential Challenges

### Challenge 1: Dataset Creation
- **Issue**: Hard to create genuine paradigm conflicts in MMLU questions
- **Mitigation**: Focus on domains with known paradigm debates (psychology, economics, philosophy)
- **Contingency**: Use natural paradigm conflicts in questions rather than modifying existing ones

### Challenge 2: Small Sample Size
- **Issue**: 10 questions per condition may lack statistical power
- **Mitigation**: Use effect sizes; focus on large, interpretable differences
- **Contingency**: Increase to 15 per condition if time allows

### Challenge 3: Activation Extraction Complexity
- **Issue**: TransformerLens may have learning curve
- **Mitigation**: Use HuggingFace transformers as backup
- **Contingency**: Focus on final layer activations only (simpler extraction)

### Challenge 4: CPU Speed
- **Issue**: GPT-2 inference slow on CPU
- **Mitigation**: Use GPT-2 small (faster)
- **Contingency**: Reduce sample size or layers analyzed

### Challenge 5: Noisy Signals
- **Issue**: Activation patterns may be too noisy to distinguish
- **Mitigation**: Aggregate over multiple tokens/runs
- **Contingency**: Report negative result; discuss why paradigm-level uncertainty might not be distinguishable

## Success Criteria

### Minimum Success (Proof-of-Concept):
- ✓ Successfully extract activations from 30 questions
- ✓ Compute at least 2 metrics (persistence, distribution)
- ✓ Statistical comparison of Path vs. Paradigm
- ✓ Visualize results (1-2 plots)
- ✓ Document methodology and results in REPORT.md

### Target Success (Full Hypothesis Test):
- ✓ All metrics computed
- ✓ Statistical significance (p < 0.05) on at least 1 metric
- ✓ Medium-to-large effect size (d > 0.5)
- ✓ Clear visualizations of differences
- ✓ Comprehensive analysis and interpretation

### Stretch Success (Strong Evidence):
- ✓ Significant effects across multiple metrics
- ✓ Large effect sizes (d > 0.8)
- ✓ Clear qualitative differences in activation patterns
- ✓ Replication across different model layers
- ✓ Insights into mechanism (which layers, which heads)

## Next Steps After This Research

### Immediate Follow-ups:
1. Test on larger sample (100+ questions per condition)
2. Replicate with different model architectures (BERT, LLaMA)
3. Test with API-based models using behavioral proxies
4. Examine specific attention heads responsible for paradigm processing

### Alternative Approaches:
1. Fine-grained token-level analysis (not just question-level)
2. Causal intervention (activation patching)
3. Probing classifiers to predict condition from activations
4. Multi-model comparison (GPT-2, GPT-J, LLaMA)

### Broader Extensions:
1. Application to multi-turn dialogue (paradigm shifts over conversation)
2. Cross-lingual paradigm conflicts
3. Domain-specific paradigms (medical, legal, scientific)
4. Intervention strategies to reduce paradigm-induced uncertainty

## Open Questions

1. Are paradigm conflicts even representable in current LLMs?
2. What constitutes a "paradigm" at the computational level?
3. Can we train models to explicitly recognize paradigm conflicts?
4. Do larger models show clearer paradigm-level signatures?
5. Is paradigm uncertainty beneficial (diversity) or harmful (confusion)?

---

**Plan Status**: Ready to implement
**Next Phase**: Environment setup and dependency installation
**Estimated total time**: 3.5 hours (can compress to 2-3 hours if needed)
