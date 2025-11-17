# Paradigm-Level Uncertainty in Language Models

> Research investigating whether paradigm-level uncertainty (competing conceptual frameworks) produces distinct activation patterns compared to path-level uncertainty in LLMs.

## Quick Summary

**Research Question**: Can we distinguish paradigm-level from path-level uncertainty through activation pattern analysis in GPT-2?

**Key Findings**:
- **No significant distinction found** between paradigm-level and path-level uncertainty (p=0.674)
- Both uncertainty types showed higher activation entropy than control questions (medium effect)
- Surprisingly, control questions showed **highest** layer-wise variance (opposite of hypothesis)
- Results suggest GPT-2 may not represent paradigm conflicts as computationally distinct

**Main Conclusion**: In GPT-2, paradigm-level uncertainty cannot be distinguished from path-level uncertainty through activation pattern analysis. This challenges the hypothesis that paradigms create unique computational signatures, though larger models may show different results.

---

## Project Structure

```
paradigm-uncertainty-nlp-20e9/
├── README.md                          # This file
├── REPORT.md                          # Comprehensive research report
├── planning.md                        # Detailed research plan
├── resources.md                       # Literature review and resource decisions
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
│
├── notebooks/
│   └── 2025-11-16-22-37_ParadigmUncertaintyExperiment.ipynb  # Main experiment
│
└── results/
    ├── activation_analysis.png        # Main visualization
    ├── pca_activations.png            # PCA cluster analysis
    └── results_summary.json           # Raw metrics and statistics
```

---

## Key Results Summary

### Statistical Findings

| Comparison | Metric | p-value | Cohen's d | Result |
|------------|--------|---------|-----------|--------|
| **Path vs Paradigm** | **Activation Entropy** | **0.674 ns** | **-0.06** | **No difference** |
| Path vs Paradigm | Layer-wise Variance | <0.001*** | 1.13 | Path > Paradigm |
| Control vs Path | Activation Entropy | <0.001*** | -0.51 | Path > Control |
| Control vs Paradigm | Activation Entropy | <0.001*** | -0.64 | Paradigm > Control |

### Main Patterns

1. **No Paradigm/Path Distinction**: Both uncertainty types showed similar activation patterns
2. **Control Questions Most Variable**: Highest layer-wise variance (opposite of prediction)
3. **Uncertain Questions Higher Entropy**: Both path and paradigm > control (but path ≈ paradigm)
4. **Perfect Stability**: Deterministic model behavior (all cross-run similarity = 1.00)

---

## How to Reproduce

### 1. Environment Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiment

Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/2025-11-16-22-37_ParadigmUncertaintyExperiment.ipynb
```

Or run all cells programmatically:
```bash
jupyter nbconvert --to notebook --execute notebooks/2025-11-16-22-37_ParadigmUncertaintyExperiment.ipynb
```

### 3. View Results

- **Comprehensive report**: See `REPORT.md`
- **Visualizations**: Check `results/activation_analysis.png` and `results/pca_activations.png`
- **Raw data**: `results/results_summary.json`

---

## Experimental Design

### Three Conditions

1. **Control** (n=10): High-confidence factual questions
   - Example: "The chemical symbol for gold is"

2. **Path-level** (n=10): Multiple solving methods, same framework
   - Example: "To convert Celsius to Fahrenheit, one can either multiply by 1.8 and add 32, or"

3. **Paradigm-level** (n=10): Competing conceptual frameworks
   - Example: "From a behavioral economics perspective versus rational choice theory, decision-making under uncertainty is explained by"

### Methodology

- **Model**: GPT-2 small (124M parameters)
- **Layers analyzed**: 2 (early), 6 (middle), 11 (late)
- **Runs per question**: 3
- **Total forward passes**: 90
- **Metrics**: Layer-wise variance, activation entropy, cross-run similarity, token variance

---

## Main Visualizations

### Figure 1: Activation Pattern Comparison
![Activation Analysis](results/activation_analysis.png)

Shows layer-wise variance (highest for control), activation entropy (similar for path/paradigm), and statistical comparisons.

### Figure 2: PCA Clustering
![PCA Analysis](results/pca_activations.png)

Three conditions cluster separately in PCA space, with paradigm forming tightest cluster.

---

## Implications

### Theoretical
- Paradigm conflicts may not be represented distinctly in GPT-2
- Uncertainty may be processed uniformly regardless of source
- Larger models may show different patterns (requires future testing)

### Practical
- Generic uncertainty quantification methods may suffice for current LLMs
- Paradigm-specific interventions may not be needed (or detectable)
- Future architectures could explicitly model competing frameworks

---

## Limitations

1. **Small sample**: 10 questions per condition (limited statistical power for small effects)
2. **Single model**: GPT-2 only; larger models may behave differently
3. **Operationalization**: Questions may not genuinely induce paradigm conflicts
4. **Aggregation**: Mean-pooling may obscure fine-grained patterns
5. **Deterministic model**: Cannot test stability hypothesis (all runs identical)

---

## Future Work

### Immediate Next Steps
1. Test larger models (GPT-J, LLaMA-2, GPT-4)
2. Refine paradigm operationalization (implicit conflicts, domain-specific)
3. Fine-grained analysis (token-level, attention heads)
4. Probing classifiers to quantify information in activations

### Broader Extensions
1. Multi-turn dialogue paradigm shifts
2. Causal intervention experiments (activation patching)
3. Cross-lingual paradigm analysis
4. Paradigm-aware architectures

---

## Citation

If you use this work, please cite:

```
Paradigm-Level Uncertainty in Language Models
Research Project: Conceptual Crossroads
November 2025
```

---

## Contact & Feedback

For questions or feedback about this research:
- Open an issue in this repository
- See full methodology in `REPORT.md`
- Review experimental plan in `planning.md`

---

## License

This research is provided for academic and educational purposes. Code and data are available under standard open research practices.

---

**Last Updated**: November 16, 2025
**Experiment Duration**: ~3.5 hours
**Status**: Complete ✅
