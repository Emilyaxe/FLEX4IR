# An IR-Centric Compiler Testing Framework via Self-Adaptive Fuzzing


## Overview

Intermediate representation (IR) based optimizations are fundamental to modern compilers, yet they remain among the most error-prone components. Existing IR-level fuzzing approaches rely heavily on handcrafted rules, which struggle to cover the vast and heterogeneous program space, inherently limiting test input diversity. This lack of diversity becomes a major obstacle to uncovering subtle or deep-seated compiler bugs. In this paper, we present FLEX4IR, a novel self-adaptive fuzzing framework that addresses these limitations through neural program generation and iterative learning. FLEX4IR starts with a small seed corpus and employs five key steps: diversity-guided sampling to select highly diverse training data, neural model training to learn valid IR syntax and semantics, perturbed generation to introduce meaningful variations, compiler execution to detect crashes, and diversity augmentation to expand the corpus with semantically equivalent variants. We implement and evaluate FLEX4IR on two representative compilers: MLIR and LLVM. On MLIR, a 5-day campaign uncovers 19 previously unknown bugs, with all confirmed or fixed. On LLVM, a 5-day campaign discovers 14 previously unknown bugs, with 9 confirmed or fixed. In 24-hour comparisons against five state-of-the-art MLIR fuzzers, FLEX4IR detects 26 bugs (1.73× more than the best baseline) and achieves 23.27% line coverage (17.08% higher than the best baseline). Ablation studies confirm that diversity-guided sampling, perturbed generation, and diversity augmentation are all essential to FLEX4IR’s effectiveness,demonstrating the promise of self-adaptive, learning-based fuzzing for improving compiler reliability.


## Directory Structure

```
.
├── code
│   ├── exp_srcipt/    # Running scripts
│   └── model/         # Test program generation and execution code (entry point for running the project)
└── data
    ├── Discussion/
    ├── RQ2/
    └── RQ3/
```

* **code/**: Contains all code for FLEX4IR.

  * **exp\_srcipt/**: Scripts to run and manage MLIR compilation and testing.
  * **model/**: Main code for test program generation and running experiments.
    *To run FLEX, start from this directory.*
* **data/**: Contains the bug data.

  * Each subdirectory (e.g., `Discussion/`, `RQ2/`, `RQ3/`) contains a `bugs.json` file, which is a dictionary mapping each discovered bug to its corresponding triggering stack trace.

## How to Use

1. **Clone this repository**

   ```bash
   git clone 
   cd FLEX4IR
   ```

2. **Generating Test Programs and Running Experiments**

   * Navigate to `code/model/` and follow the instructions in the README or scripts to start generating test programs and running fuzzing experiments.
