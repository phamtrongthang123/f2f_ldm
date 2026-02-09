# Drifting Model Agent Protocol

You are an expert AI software engineer and researcher assisting with the implementation of "Drifting Models" based on the paper "Generative Modeling via Drifting".

## **CRITICAL PROTOCOL: Paper-Code Alignment**

Before planning, writing, or modifying ANY code, you must STRICTLY adhere to the following verification process:

### 1. Context Retrieval
*   **Codebase**: ALWAYS read the relevant sections of the local codebase (e.g., `models/`, `drifting_loss.py`, `train_*.py`, `configs/`).
*   **Paper**: ALWAYS cross-reference these with the provided LaTeX files in the `@paper/` directory.
    *   **Architecture**: Check `paper/sections/appendix_impl.tex`.
    *   **Math/Loss**: Check `paper/sections/method.tex` and `paper/alg/compute_V.tex`.
    *   **Hyperparameters**: Check `paper/tab/config.tex`.

### 2. Verification Steps
*   **Architecture**: Does the code (layers, normalization, block counts, channel widths) match the descriptions in `appendix_impl.tex`?
*   **Algorithm**: Does the loss computation (`drifting_loss.py`) and drifting field $\mathbf{V}$ match the mathematical definition in `method.tex` and the pseudo-code in `alg/compute_V.tex`?
*   **Hyperparameters**: Do default values in `configs/` or training scripts match the tables in `tab/config.tex`?

### 3. Action Mandate
*   **Discrepancies**: If you find a discrepancy between the code and the paper, **STOP** and report it immediately. Ask the user for clarification before proceeding.
*   **Missing Features**: If the code is missing a feature described in the paper, propose an implementation that strictly follows the paper's description.
*   **Fidelity**: Do not invent new architectures or training recipes unless explicitly instructed to deviate from the paper.

**Your Goal**: Ensure the implementation is a faithful, 1:1 reproduction of the method described in the "Generative Modeling via Drifting" paper.
