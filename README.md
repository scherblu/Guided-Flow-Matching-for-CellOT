# Guided Flow Matching for CellOT

## 🛠 Prerequisites
* **Python 3.12+**
* **Conda**
* **CUDA-enabled GPU**

## ⚙️ Installation

## 🚀 Dependency management with pixi

### Step 1: Install pixi

*Skip this if you already have pixi installed.*

```bash
# macOS / Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Or with Homebrew (macOS)
brew install pixi
```

Restart your terminal after installation. See [pixi installation docs](https://pixi.sh/latest/#installation) for Windows and other options.

### Step 2: Set up the environment

```bash
pixi install                   # create environment from pixi.toml
pixi run install-kernel        # register Jupyter kernel
pixi shell                     # activate session in pixi env
```

## Working with single-cell data
*Careful: we're working with spatially resolved protein data & morphology features. Classical single-cell data is transcriptomics (gene activity) without spatial information*

- Best practices guide: https://www.sc-best-practices.org/preamble.html
- Your best tool: https://scanpy.readthedocs.io/en/stable/index.html
- The software universe: https://scverse.org/
- Before you spend too much time, please contact a bio person :) me for example
