# Quadratic Tetrahedral Mass Properties Calculator

A Python implementation for accurately computing mass properties (volume and centroid) of 10-node quadratic tetrahedral finite elements using Gaussian quadrature.

## Overview

This code demonstrates how to precisely calculate the volume and center of mass of second-order (quadratic) tetrahedral elements commonly used in finite element analysis. Rather than approximating these elements as linear tetrahedra, this implementation performs exact integration using 4-point Gaussian quadrature to achieve higher accuracy.

## Features

- Calculation of 10-node quadratic tetrahedral shape functions and their derivatives
- Exact volume computation using Gaussian quadrature
- Precise centroid (center of mass) calculation
- Element orientation verification
- Mass calculation with customizable density
- Example file parser for Nastran BDF format

## Mathematical Background

The implementation uses:

- Quadratic tetrahedral shape functions with natural coordinates (ξ, η, ζ)
- Barycentric coordinate transformations
- 4-point symmetric Gaussian quadrature (degree 2 exact)
- Jacobian-based volume computation
- Linear sub-element check for proper orientation

## Requirements

- Python ≥ 3.13
- NumPy ≥ 2.2.6

## Usage

The code demonstrates reading tetrahedral elements from a Nastran BDF file and calculating their properties:

```python
# Read a mesh file and compute mass properties
import numpy as np
from tetra_mass_props import analyze_quadratic_tetra

# Example: create a 10-node tetrahedral element
coords = np.array([
    [0, 0, 0],    # Node 1 (corner)
    [1, 0, 0],    # Node 2 (corner)
    [0, 1, 0],    # Node 3 (corner)
    [0, 0, 1],    # Node 4 (corner)
    [0.5, 0, 0],  # Node 5 (edge)
    [0.5, 0.5, 0],# Node 6 (edge)
    [0, 0.5, 0],  # Node 7 (edge)
    [0, 0, 0.5],  # Node 8 (edge)
    [0.5, 0, 0.5],# Node 9 (edge)
    [0, 0.5, 0.5] # Node 10 (edge)
])

# Analyze the element
result = analyze_quadratic_tetra(coords, density=1.0)
print(f"Volume: {result['volume']}")
print(f"Centroid: {result['centroid']}")
print(f"Mass: {result['mass']}")
```

## Theory

The volume of a curved quadratic tetrahedron is computed as:

$V = \int_{\Omega} d\Omega$

This is transformed to natural coordinates:

$V = \int_0^1 \int_0^{1-\xi} \int_0^{1-\xi-\eta} |J(\xi,\eta,\zeta)| d\zeta d\eta d\xi$

Where $|J|$ is the determinant of the Jacobian matrix relating natural to physical coordinates.

For quadratic tetrahedra, this integration is performed using Gaussian quadrature for accuracy.

## License

MIT
