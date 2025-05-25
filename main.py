"""
Python module for analysis of 10-node quadratic tetrahedral finite elements.
Provides shape functions, their derivatives, Jacobian computation,
exact volume and centroid calculation via 4-point Gaussian quadrature,
and orientation check with mass calculation.
"""

from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

DTYPE_FLOAT: Final = np.float64
FloatArray: TypeAlias = NDArray[np.float64]


# --------------------------------------
# Shape functions
# --------------------------------------
def _compute_tetrahedral_shape_functions(
    xi: float, eta: float, zeta: float
) -> FloatArray:
    """
    Compute the 10-node quadratic tetrahedral shape functions at given natural coordinates.

    Args:
        xi:    Natural coordinate ξ (>=0).
        eta:   Natural coordinate η (>=0).
        zeta:  Natural coordinate ζ (>=0), with ξ+η+ζ <= 1.

    Returns:
        N:  (10,) array of shape function values [N1...N10].
    """
    L1: Final = 1 - xi - eta - zeta
    L2: Final = xi
    L3: Final = eta
    L4: Final = zeta
    N = np.zeros(10, dtype=DTYPE_FLOAT)
    N[0] = L1 * (2 * L1 - 1)
    N[1] = L2 * (2 * L2 - 1)
    N[2] = L3 * (2 * L3 - 1)
    N[3] = L4 * (2 * L4 - 1)
    N[4] = 4 * L1 * L2
    N[5] = 4 * L2 * L3
    N[6] = 4 * L1 * L3
    N[7] = 4 * L1 * L4
    N[8] = 4 * L2 * L4
    N[9] = 4 * L3 * L4
    return N


def _compute_tetrahedral_shape_derivatives(
    xi: float, eta: float, zeta: float
) -> FloatArray:
    """
    Compute derivatives of the 10-node quadratic tetrahedral shape functions w.r.t. natural coords.

    Args:
        xi:    Natural coordinate ξ.
        eta:   Natural coordinate η.
        zeta:  Natural coordinate ζ.

    Returns:
        dN:  (10,3) array where columns are dN/dξ, dN/dη, dN/dζ.
    """
    L1: Final = 1 - xi - eta - zeta
    L2: Final = xi
    L3: Final = eta
    L4: Final = zeta
    dN = np.zeros((10, 3), dtype=DTYPE_FLOAT)
    # d/dxi, d/deta, d/dzeta
    # N_i = L_i(2L_i-1) -> dN_i/dL_j = delta_ij * (4L_i - 1)
    # dL1/dxi = -1, dL1/deta = -1, dL1/dzeta = -1
    dN[0, :] = -1.0 * (4 * L1 - 1)  # dL1/dxi = -1, dL1/deta = -1, dL1/dzeta = -1
    # N2 = L2 (2 L2 - 1)
    dN[1, 0] = 4 * L2 - 1  # dL2/dxi = 1
    # N3
    dN[2, 1] = 4 * L3 - 1  # dL3/deta = 1
    # N4
    dN[3, 2] = 4 * L4 - 1  # dL4/dzeta = 1

    # N5 = 4 L1 L2
    dN[4, 0] = 4 * (L1 - L2)  # d/dxi (L1*1 + L2*(-1))
    dN[4, 1] = 4 * (-L2)  # d/deta (L1*0 + L2*(-1))
    dN[4, 2] = 4 * (-L2)  # d/dzeta (L1*0 + L2*(-1))

    # N6 = 4 L2 L3
    # dN6/dxi = 4 * ( (dL2/dxi)*L3 + (dL3/dxi)*L2 ) = 4 * ( 1*L3 + 0*L2 ) = 4*L3
    dN[5, 0] = 4 * L3
    # dN6/deta = 4 * ( (dL2/deta)*L3 + (dL3/deta)*L2 ) = 4 * ( 0*L3 + 1*L2 ) = 4*L2
    dN[5, 1] = 4 * L2
    # dN6/dzeta = 4 * ( (dL2/dzeta)*L3 + L2*(dL3/dzeta) ) = 4 * ( 0*L3 + L2*0 ) = 0 (already zero)

    # N7 = 4 L1 L3
    dN[6, 0] = 4 * (-L3)  # d/dxi (L1*0 + L3*(-1))
    dN[6, 1] = 4 * (L1 - L3)  # d/deta (L1*1 + L3*(-1))
    dN[6, 2] = 4 * (-L3)  # d/dzeta (L1*0 + L3*(-1))

    # N8 = 4 L1 L4
    dN[7, 0] = 4 * (-L4)  # d/dxi (L1*0 + L4*(-1))
    dN[7, 1] = 4 * (-L4)  # d/deta (L1*0 + L4*(-1))
    dN[7, 2] = 4 * (L1 - L4)  # d/dzeta (L1*1 + L4*(-1))

    # N9 = 4 L2 L4
    dN[8, 0] = 4 * L4  # dN9/dxi = 4 * L4
    # dN9/deta = 0 (already zero)
    dN[8, 2] = 4 * L2  # dN9/dzeta = 4 * L2

    # N10 = 4 L3 L4
    # dN10/dxi = 0 (already zero)
    dN[9, 1] = 4 * L4  # dN10/deta = 4 * L4
    dN[9, 2] = 4 * L3  # dN10/dzeta = 4 * L3
    return dN


# --------------------------------------
# Constants
# --------------------------------------

# Quadrature scheme constants for exact integration of quadratic geometry
# 4-point symmetric rule (degree 2 exact)
# Barycentric coordinates for quadrature points: L1, L2, L3, L4
_QUAD_A: Final[float] = (5 + 3 * (5**0.5)) / 20
_QUAD_B: Final[float] = (5 - (5**0.5)) / 20
_QUAD_BARY_COORDS: Final[FloatArray] = np.array(
    [
        [_QUAD_A, _QUAD_B, _QUAD_B, _QUAD_B],
        [_QUAD_B, _QUAD_A, _QUAD_B, _QUAD_B],
        [_QUAD_B, _QUAD_B, _QUAD_A, _QUAD_B],
        [_QUAD_B, _QUAD_B, _QUAD_B, _QUAD_A],
    ],
    dtype=DTYPE_FLOAT,
)

# Natural coordinates (xi, eta, zeta) for quadrature points, derived from barycentric (L2, L3, L4)
TETRA_QUADRATURE_POINTS: Final[FloatArray] = _QUAD_BARY_COORDS[:, 1:4]
TETRA_QUADRATURE_WEIGHTS: Final[FloatArray] = np.array(
    [0.25, 0.25, 0.25, 0.25], dtype=DTYPE_FLOAT
)

# Precompute shape functions and their derivatives at quadrature points
# to avoid recomputing them for every element.
_PRECOMPUTED_SHAPE_FUNCTIONS_AT_QUAD_POINTS_LIST: list[FloatArray] = [
    _compute_tetrahedral_shape_functions(xi, eta, zeta)
    for xi, eta, zeta in TETRA_QUADRATURE_POINTS
]
_PRECOMPUTED_SHAPE_FUNCTIONS_AT_QUAD_POINTS: Final[FloatArray] = np.array(
    _PRECOMPUTED_SHAPE_FUNCTIONS_AT_QUAD_POINTS_LIST, dtype=DTYPE_FLOAT
)

_PRECOMPUTED_DSHAPE_DXI_AT_QUAD_POINTS_LIST: list[FloatArray] = [
    _compute_tetrahedral_shape_derivatives(xi, eta, zeta)
    for xi, eta, zeta in TETRA_QUADRATURE_POINTS
]
_PRECOMPUTED_DSHAPE_DXI_AT_QUAD_POINTS: Final[FloatArray] = np.array(
    _PRECOMPUTED_DSHAPE_DXI_AT_QUAD_POINTS_LIST, dtype=DTYPE_FLOAT
)


# --------------------------------------
# Jacobian
# --------------------------------------
def jacobian_at_point(coords: FloatArray, dN_at_point: FloatArray) -> FloatArray:
    """
    Compute the (3,3) Jacobian matrix mapping natural to global coordinates.

    This function is kept for potential standalone use, but for volume/centroid
    computation, Jacobian calculation is integrated into the combined function.

    Args:
        coords:  (10,3) array of global node coordinates.
        dN_at_point: (10,3) array of shape function derivatives at the point.

    Returns:
        J:  (3,3) Jacobian matrix.
    """
    J = dN_at_point.T @ coords
    return J


# --------------------------------------
# Volume & Centroid
# --------------------------------------
def compute_quadratic_tetra_volume_and_centroid(
    coords: FloatArray,
) -> tuple[float, FloatArray]:
    """
    Compute exact volume and centroid of a 10-node quadratic tetrahedron
    via Gaussian quadrature in a single loop, using precomputed shape functions
    and their derivatives at quadrature points.

    Args:
        coords: (10,3) global node coordinates.

    Returns:
        vol: Exact volume.
        centroid: (3,) coordinates of centroid.
    """
    w = TETRA_QUADRATURE_WEIGHTS
    vol = 0.0
    centroid_numerator = np.zeros(3, dtype=DTYPE_FLOAT)

    for i, wi in enumerate(w):
        N_gp = _PRECOMPUTED_SHAPE_FUNCTIONS_AT_QUAD_POINTS[i]  # (10,)
        dN_gp = _PRECOMPUTED_DSHAPE_DXI_AT_QUAD_POINTS[i]  # (10, 3)

        J = dN_gp.T @ coords  # Jacobian matrix (3,10) @ (10,3) -> (3,3)
        detJ = np.linalg.det(J)
        if not detJ > 0:
            raise ValueError(
                f"Jacobian determinant is not positive ({detJ:.6e}) at quadrature point {i}. "
                "Element may be distorted or incorrectly defined."
            )
        vol += wi * detJ

        x_gp = N_gp @ coords  # Global coordinates of the current quadrature point
        centroid_numerator += wi * x_gp * detJ

    if abs(vol) < 1e-12:  # Check for effectively zero volume
        # Return 0 volume and a zero vector for centroid if volume is negligible.
        return 0.0, np.zeros(3, dtype=DTYPE_FLOAT)

    centroid = centroid_numerator / vol
    return vol, centroid


# --------------------------------------
# Linear volume & orientation
# --------------------------------------
def compute_linear_tetra_volume(corner_coords) -> float:
    """
    Compute linear tetrahedron volume and its sign using first 4 corner nodes.

    Args:
        corner_coords: (4,3) array of first four node coords.

    Returns:
        V_lin: Linear tetra volume (signed/6).
        sign: +1 or -1 indicating orientation.
    """
    M = corner_coords[1:, :] - corner_coords[0, :]
    det = np.linalg.det(M)
    V_lin = det / 6.0
    return V_lin


def analyze_quadratic_tetra(coords, density=1.0):
    """
    Analyze a 10-node quadratic tetrahedron element.

    Performs an orientation check using the linear sub-element,
    then computes the exact quadratic volume, centroid, and mass.

    Args:
        coords: (10,3) global node coordinates of the quadratic tetrahedron.
        density: Material density (default is 1.0).

    Returns:
        A dictionary containing:
            "volume_linear": Volume of the linear tet (corners only), for orientation.
            "volume": Exact volume of the quadratic tetrahedron.
            "centroid": (3,) coordinates of the centroid.
            "mass": Mass of the element.
    """

    # Orientation check
    V_lin = compute_linear_tetra_volume(coords[:4])
    if V_lin <= 0:
        raise ValueError(f"Invalid element orientation: linear volume {V_lin:.6e} <= 0")

    # Exact quadratic volume and centroid
    V, C = compute_quadratic_tetra_volume_and_centroid(coords)
    # Mass
    mass = density * V
    return {"volume_linear": V_lin, "volume": V, "centroid": C, "mass": mass}


def main():
    # rho = 1.0
    height = 10
    radius = 50
    volume = height * np.pi * (radius**2)

    sfields = [slice(num * 8, (num + 1) * 8) for num in range(10)]
    ifp = "disc.bdf"
    grid: dict[int, FloatArray] = {}
    ctetra: dict[int, NDArray[np.integer]] = {}
    with open(ifp) as f:
        iterator = iter(f)
        try:
            while True:
                line = next(iterator)
                values = [line[sli] for sli in sfields]
                card = values[0].rstrip()
                match card:
                    case "GRID":
                        values = [line[sli] for sli in sfields]
                        nid = int(values[1])
                        pos = [float(xyz) for xyz in values[3:6]]
                        grid[nid] = np.array(pos)
                    case "CTETRA":
                        line2 = next(iterator)
                        values2 = [line2[sli] for sli in sfields]
                        eid = int(values[1])
                        verts = [int(nid) for nid in values[3:9] + values2[1:5]]
                        ctetra[eid] = np.array(verts)
        except StopIteration:
            print("EOF")
            pass
    # xyz_idx = np.array([nid for nid in grid.keys()])
    nid2idx = {nid: idx for idx, nid in enumerate(grid.keys())}
    xyz_mat = np.array([xyz for xyz in grid.values()])

    vol_1st_tmp = 0.0
    vol_2nd_tmp = 0.0
    cog_1st_tmp = np.zeros(3, dtype=DTYPE_FLOAT)
    for eid, verts in ctetra.items():
        indices = [nid2idx[nid] for nid in verts]
        coords = xyz_mat[indices[:4], :]
        vectors = coords[1:, :] - coords[0, :]
        det_tmp = np.linalg.det(vectors)
        assert det_tmp > 0
        cog_1st_tmp += det_tmp * np.mean(coords, axis=0)
        vol_1st_tmp += det_tmp
        ret = analyze_quadratic_tetra(xyz_mat[indices, :])
        vol_2nd_tmp += ret["volume"]
    vol_1st = vol_1st_tmp / 6.0
    cog_1st = cog_1st_tmp / vol_1st_tmp
    print(cog_1st)
    vol_2nd = vol_2nd_tmp / 6.0
    print(volume, vol_1st, (vol_1st / volume - 1) * 100)
    print(volume, vol_2nd, (vol_2nd / volume - 1) * 100)


if __name__ == "__main__":
    main()
