# Phase 0: Resource Research & Decision Documentation

## Research Question
Can we distinguish paradigm-level uncertainty from path-level uncertainty in language models through activation pattern analysis?

## Research Conducted (30 minutes)

### 1. Literature Search Results

**Key Papers Found (2024-2025):**

1. **"Language Models Are Capable of Metacognitive Monitoring" (May 2025)**
   - LLMs can monitor internal activations
   - Confidence signals encoded in activation patterns
   - Relevant for understanding uncertainty representation

2. **"Interpreting and Mitigating Unwanted Uncertainty in LLMs" (2024)**
   - Documents flip-flop effect: LLMs change correct answers 46% of time when challenged
   - 17% accuracy drop in multi-turn scenarios
   - Suggests multiple uncertainty types exist

3. **"Benchmarking LLMs via Uncertainty Quantification" (NeurIPS 2024)**
   - GitHub: smartyfh/LLM-Uncertainty-Bench
   - Uses MMLU dataset with 10,000 instances
   - Key finding: Larger LLMs may show greater uncertainty

4. **"LLM Factoscope" (2024)**
   - Uses Integrated Gradients for activation analysis
   - Most influential features in middle-to-late layers
   - Analyzes contribution of activation maps

5. **"Patchscopes Framework" (Google Research 2024)**
   - Unified framework for inspecting hidden representations
   - Natural language explanations from internal states

### 2. Datasets Identified

**Primary Choice: MMLU (Massive Multitask Language Understanding)**
- ✓ Well-established benchmark with ground truth
- ✓ Multiple-choice format enables clear correctness signal
- ✓ Diverse topics allow testing across domains
- ✓ Available via HuggingFace datasets library
- ✓ Used in recent uncertainty research (LLM-Uncertainty-Bench)

**Alternative: TruthfulQA**
- Tests factual accuracy and truthfulness
- Designed to expose common misconceptions
- Relevant for uncertainty in high-trust scenarios

**Decision: Use MMLU subset**
- More suitable for activation analysis (multiple layers of reasoning)
- Can create paradigm conflict by embedding contradictory frameworks
- Manageable size for 1-hour experiment constraint

### 3. Methods & Baselines

**Activation Analysis Approach:**
- Extract hidden states from transformer layers
- Compare activation patterns between:
  1. **Path-level uncertainty**: Multiple valid reasoning paths, same paradigm
  2. **Paradigm-level uncertainty**: Conflicting conceptual frameworks

**Tools Identified:**
- **TransformerLens**: Python library for hooking into transformers (mechanistic interpretability)
- **HuggingFace Transformers**: Access to model internals
- **Integrated Gradients**: Attribution method for activation analysis

**Proposed Methods:**
1. **Baseline**: Standard activation analysis on uncertain vs. certain questions
2. **Path-level test**: Questions with multiple solving methods (e.g., math problems)
3. **Paradigm-level test**: Questions with embedded framework conflicts (e.g., frequentist vs. Bayesian framing)

### 4. Model Selection

**Available APIs:**
- ✓ OpenAI (GPT-4.1, GPT-5)
- ✓ Claude (Sonnet 4.5)
- ✓ OpenRouter (various models)

**Decision: Use GPT-2 or Small Open Model for Activation Analysis**
- Rationale: Need direct access to internal activations
- API-based models (GPT-4, Claude) don't expose internal states
- GPT-2 (124M-1.5B params) is:
  - Small enough to run on CPU
  - Well-documented for interpretability
  - Supported by TransformerLens
  - Can download from HuggingFace

**For comparison: API calls to larger models**
- Use GPT-4.1 or Claude for behavioral validation
- Compare uncertainty patterns at behavioral level

### 5. Evaluation Metrics

**Primary Metrics:**
1. **Activation Pattern Persistence**: How long uncertainty signal persists across layers
2. **Spatial Distribution**: Localized vs. globally distributed activations
3. **Layer-wise Variance**: Variability in hidden states across transformer layers
4. **Attention Head Divergence**: Disagreement between attention heads

**Statistical Tests:**
- t-tests for comparing path vs. paradigm activation patterns
- Effect sizes (Cohen's d)
- Clustering analysis (PCA/t-SNE visualization)

### 6. Resource Constraints

**Time: 1 hour total**
- Phase 0 (Research): 30 min ✓
- Phase 1 (Planning): 15 min
- Phase 2 (Setup): 10 min
- Phase 3 (Implementation): 45 min
- Phase 4 (Experiments): 60 min
- Phase 5 (Analysis): 30 min
- Phase 6 (Documentation): 20 min
- Total: ~3.5 hours (scaled down to fit 1h by using smaller dataset)

**Compute: CPU only**
- ✓ GPT-2 small runs on CPU
- ✓ MMLU subset fits in memory
- ✓ No GPU required

**Budget: $100**
- Minimal costs (small model inference is free)
- Possible API calls for validation: ~$5-10

## Gaps & Proposed Solutions

**Gap 1: No existing "paradigm-level uncertainty" dataset**
- **Solution**: Create synthetic test cases by:
  - Modifying MMLU questions to embed paradigm conflicts
  - Example: Present medical question with both Western and Traditional medicine frameworks
  - Control: Same question with path-level uncertainty only

**Gap 2: No established baseline for this specific phenomenon**
- **Solution**: Create own baselines:
  - Random activation patterns
  - Standard uncertain vs. certain questions
  - Path-level uncertainty (established concept)

**Gap 3: Limited time for extensive experiments**
- **Solution**: Focus on proof-of-concept:
  - Use 50-100 questions (not full MMLU)
  - Test 2-3 layers (early, middle, late)
  - Measure 2-3 key metrics (persistence, distribution)

## Final Resource Plan

**Dataset:**
- MMLU subset (100 questions, 5 domains)
- Augmented with paradigm conflicts (manual creation)
- Control set: path-level uncertainty

**Models:**
- Primary: GPT-2 (124M or 355M) via TransformerLens
- Validation: GPT-4.1 API calls (behavioral only)

**Libraries:**
- transformers, datasets (HuggingFace)
- transformer_lens (interpretability)
- scikit-learn (analysis)
- matplotlib, seaborn (visualization)
- numpy, scipy (statistics)

**Experiments:**
1. Baseline: Certain vs. uncertain questions
2. Path-level: Multiple solving methods
3. Paradigm-level: Framework conflicts
4. Analysis: Compare activation patterns

## Justification

This approach balances:
- ✓ Scientific rigor (established datasets, statistical tests)
- ✓ Feasibility (CPU-compatible, 1-hour constraint)
- ✓ Novelty (tests new hypothesis about uncertainty types)
- ✓ Reproducibility (open models, documented methods)

The use of GPT-2 for activation analysis is standard practice in mechanistic interpretability research, and MMLU provides a solid foundation for creating test cases.
