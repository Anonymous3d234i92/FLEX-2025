# FLEX: Interleaved Learning and Exploration: A Self-Adaptive Fuzz Testing Framework for MLIR


## Overview

MLIR (Multi-Level Intermediate Representation) has rapidly become a foundational technology for modern compiler frameworks, enabling extensibility across diverse domains. However, existing fuzzing approaches—based on manually crafted templates or rule-based mutations—struggle to generate sufficiently diverse and semantically valid test cases, making it difficult to expose subtle or deep-seated bugs within MLIR’s complex and evolving code space.

**FLEX** is a novel self-adaptive fuzzing framework for MLIR. FLEX leverages neural networks for program generation, a perturbed sampling strategy to encourage diversity, and a feedback-driven augmentation loop that iteratively improves its model using both crashing and non-crashing test cases. Starting from a limited seed corpus, FLEX progressively learns valid syntax and semantics and autonomously produces high-quality test inputs.

We evaluate FLEX on the upstream MLIR compiler against four state-of-the-art fuzzers. In a 30-day campaign, FLEX discovers 80 previously unknown bugs—including multiple new root causes and parser bugs. In 24-hour fixed-revision comparisons, FLEX detects 53 bugs (over 3.5× as many as the best baseline) and achieves 28.2% code coverage, outperforming the next-best tool by 42%. Ablation studies further confirm the critical role of both perturbed generation and diversity augmentation in FLEX’s effectiveness.

## Directory Structure

```
.
├── code
│   ├── exp_srcipt/    # MLIR compiler running scripts
│   └── model/         # Test program generation and execution code (entry point for running the project)
└── data
    ├── Discussion/
    ├── RQ2/
    └── RQ3/
```

* **code/**: Contains all code for FLEX.

  * **exp\_srcipt/**: Scripts to run and manage MLIR compilation and testing.
  * **model/**: Main code for test program generation and running experiments.
    *To run FLEX, start from this directory.*
* **data/**: Contains all experiment results and bug data.

  * Each subdirectory (e.g., `Discussion/`, `RQ2/`, `RQ3/`) contains a `bugs.json` file, which is a dictionary mapping each discovered bug to its corresponding triggering stack trace.

## How to Use

1. **Clone this repository**

   ```bash
   git clone https://github.com/Anonymous3d234i92/FLEX-2025.git
   cd FLEX
   ```

2. **Generating Test Programs and Running Experiments**

   * Navigate to `code/model/` and follow the instructions in the README or scripts to start generating test programs and running fuzzing experiments.
