# Lab Tools

A collection of Python tools for analyzing experimental laboratory data.

## Available Tools

| Tool | Description | Docs |
|------|-------------|------|
| **[CV Analyzer](CV_Analysis/)** | Cyclic Voltammetry analysis -- peak detection, onset, Ep/2, reversibility classification | [Usage](CV_Analysis/docs/usage.md) |

## Installation

### All tools at once

```bash
pip install -e .
```

This installs every tool in the repository and makes their CLI commands
available (e.g. `cv-analyzer`).

### Individual tool

```bash
pip install -e ./CV_Analysis
```

### Dependencies

Python >= 3.9 and the following packages (installed automatically):

* numpy
* scipy
* matplotlib
* openpyxl

## Repository Structure

```
lab_tools/
├── pyproject.toml          # top-level meta-package (installs everything)
├── README.md               # this file
├── CV_Analysis/            # Cyclic Voltammetry Analyzer
│   ├── pyproject.toml      # standalone package definition
│   ├── cv_analyzer/        # Python package
│   ├── example/            # example input data
│   └── docs/               # tool-specific documentation
└── ...                     # future tools
```

## Adding a New Tool

1. Create a new directory at the top level (e.g. `UV_Vis_Analysis/`).
2. Add a Python package inside it with its own `pyproject.toml`.
3. In the **top-level** `pyproject.toml`:
   - Append the directory name to `[tool.setuptools.packages.find] where`.
   - Add CLI entry points under `[project.scripts]`.
   - Add any new dependencies to `[project.dependencies]`.
4. Update this README table.

## License

Internal use -- Glorius Group.
