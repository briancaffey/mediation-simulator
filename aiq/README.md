# NVIDIA Agent Intelligence Toolkit

## Overview

The NVIDIA Agent Intelligence Toolkit (AIQ) is a collection of tools and resources designed to help developers build, test, and deploy AI agents.

# Mediation Simulator

## Overview

This project uses the NVIDIA Agent Intelligence Toolkit (AIQ) to build a mediation simulator. It's a local development project that leverages AIQ's capabilities for building and testing AI agents.

## Setup

This project uses Python 3.13+ and can be set up using `uv`:

```
cd aiq
```

```bash
# Create a virtual environment in the .venv directory
uv venv --seed .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv sync
```

or

```
uv pip install -e .
```

The `--seed` option creates the virtual environment in a specific directory (`.venv`) rather than a random location. This makes the environment location consistent and predictable, following common Python conventions.

## Dependencies

- Python 3.13+
- aiqtoolkit
- aiqtoolkit[llama-index]
- aiqtoolkit[langchain]


## Run the project with AIQ

Using a simple calculator tool example:

```
aiq run --config_file configs/case_generation.yml --input ""
```
