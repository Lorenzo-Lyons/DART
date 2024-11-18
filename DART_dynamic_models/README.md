# Dart dynamic models

## Installation

Navigate to package root folder (This folder). Then install the package by running:

```
pip install dist/DART_dynamic_models-0.1.0-py3-none-any.whl
```

## Modifying the package contents
To update the contents of this package the user must copy-paste the new versions of *dart_dynamic_models* and/or *SVGP_saved_parameters* into the python package forder (i.e. *DART_dynamics_models/DART_dynamic_models*) and overwrite the current version.

The package must then be rebuilt:

```
python setup.py sdist bdist_wheel
```
and reinstalled (by adding the force reinstall tag):
```
pip install dist/DART_dynamic_models-0.1.0-py3-none-any.whl --force-reinstall
```

