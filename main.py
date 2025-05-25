import numpy as np
from numpy.typing import NDArray


def main():
    # rho = 1.0
    height = 10
    radius = 50
    volume = height * np.pi * (radius**2)

    sfields = [slice(num * 8, (num + 1) * 8) for num in range(10)]
    ifp = "disc.bdf"
    grid: dict[int, NDArray[np.floating]] = {}
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
    cog_1st_tmp = np.zeros(3)
    for eid, verts in ctetra.items():
        indices = [nid2idx[nid] for nid in verts[:4]]
        coords = xyz_mat[indices, :]
        vectors = coords[1:, :] - coords[0, :]
        det_tmp = np.linalg.det(vectors)
        assert det_tmp > 0
        cog_1st_tmp += det_tmp * np.mean(coords, axis=0)
        vol_1st_tmp += det_tmp
    vol_1st = vol_1st_tmp / 6.0
    cog_1st = cog_1st_tmp / vol_1st_tmp
    print(volume, vol_1st, (vol_1st / volume - 1) * 100)
    print(cog_1st)


if __name__ == "__main__":
    main()
