"""Geometry shared by the touchdown experiments (numpy only; pxr is imported inside the functions that read USD).

    TerrainGridSurface   exact height and normal of a lander-scene terrain tier (regular grid, builder's triangulation)
    plane_slope_deg      slope of the plane fitted around a point
    convex_hull_2d, support_margin_m    static stability of the centre of mass over the contact points
    footpad_geometry     footpad dish sample points and pad centres of the Apollo LM, in the lander frame
"""
import numpy as np


class TerrainGridSurface:
    """Height and normal of a terrain tier built on a regular grid (apollo_lander_modeling build_apollo17_scene.py)."""

    def __init__(self, grid, heights):
        """grid: the tier's customData "grid" dict (x0, y0, dx, dy, nx, ny); heights: (ny, nx) vertex heights."""
        self.x0, self.y0, self.dx, self.dy, self.nx, self.ny = grid["x0"], grid["y0"], grid["dx"], grid["dy"], grid["nx"], grid["ny"]
        self.heights = np.asarray(heights, dtype=np.float64)
        self.half_size = -self.x0

    @classmethod
    def from_mesh_prim(cls, prim):
        """Read grid definition and vertex heights from a tier mesh prim (pxr)."""
        from pxr import UsdGeom

        grid = prim.GetCustomDataByKey("grid")
        points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), np.float64).reshape(grid["ny"], grid["nx"], 3)
        return cls(grid, points[:, :, 2])

    def _cell(self, x, y):
        """Cell indices and fractional position of (x, y)."""
        i = int(np.clip((x - self.x0) // self.dx, 0, self.nx - 2))
        j = int(np.clip((y - self.y0) // self.dy, 0, self.ny - 2))
        return i, j, (x - self.x0 - i * self.dx) / self.dx, (y - self.y0 - j * self.dy) / self.dy

    def height(self, x, y):
        """Height of the triangle surface at (x, y), same triangulation as the builder."""
        i, j, u, v = self._cell(x, y)
        za, zb, zc, zd = self.heights[j, i], self.heights[j, i + 1], self.heights[j + 1, i], self.heights[j + 1, i + 1]
        return za + u * (zb - za) + v * (zd - zb) if u >= v else za + u * (zd - zc) + v * (zc - za)

    def normal(self, x, y):
        """Upward unit normal of the triangle under (x, y)."""
        i, j, u, v = self._cell(x, y)
        za, zb, zc, zd = self.heights[j, i], self.heights[j, i + 1], self.heights[j + 1, i], self.heights[j + 1, i + 1]
        if u >= v:
            slope_x, slope_y = (zb - za) / self.dx, (zd - zb) / self.dy
        else:
            slope_x, slope_y = (zd - zc) / self.dx, (zc - za) / self.dy
        n = np.array([-slope_x, -slope_y, 1.0])
        return n / np.linalg.norm(n)


def plane_slope_deg(surface, x, y, radius=6.0):
    """Slope (deg) of the plane fitted to the terrain within `radius` m of (x, y)."""
    points, heights = [], []
    for r in (0.0, radius / 3, 2 * radius / 3, radius):
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            points.append((x + r * np.cos(a), y + r * np.sin(a)))
            heights.append(surface.height(*points[-1]))
    design = np.c_[np.array(points) - [x, y], np.ones(len(points))]
    coefficients, *_ = np.linalg.lstsq(design, np.array(heights), rcond=None)
    return float(np.degrees(np.arctan(np.hypot(coefficients[0], coefficients[1]))))


def convex_hull_2d(points):
    """Counter-clockwise convex hull of 2-D points (monotone chain)."""
    points = sorted(map(tuple, points))

    def half_hull(sequence):
        """One half of the monotone-chain hull."""
        hull = []
        for p in sequence:
            while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= 0:
                hull.pop()
            hull.append(p)
        return hull
    lower, upper = half_hull(points), half_hull(points[::-1])
    return lower[:-1] + upper[:-1]


def support_margin_m(contact_points_xy, center_of_mass_xy):
    """Distance from the COM's vertical projection to the nearest support-polygon edge (+ inside); NaN with < 3 supports."""
    if len(contact_points_xy) < 3:
        return float("nan")
    hull = convex_hull_2d(contact_points_xy)
    if len(hull) < 3:
        return float("nan")
    margin = 1e9
    for i in range(len(hull)):
        a, b = np.array(hull[i]), np.array(hull[(i + 1) % len(hull)])
        edge = b - a
        normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        margin = min(margin, float(np.dot(np.asarray(center_of_mass_xy) - a, normal)))
    return margin


def footpad_geometry(lander_prim, subsample=6):
    """Footpad dish sample points {pad: (k, 3)} and pad bottom centres {pad: (3,)} in the lander frame (pxr).

    The dish points are the render-mesh vertices in the lowest 0.45 m that lie more than 3 m from the lander axis (the four
    footpads); pads are named by their side: +X, -X, +Y, -Y."""
    from pxr import Usd, UsdGeom

    points = np.vstack([np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(), np.float64) for p in Usd.PrimRange(lander_prim)
                        if p.IsA(UsdGeom.Mesh) and "/Geometry/" in str(p.GetPath())])
    dish = points[points[:, 2] < points[:, 2].min() + 0.45]
    dish = dish[np.hypot(dish[:, 0], dish[:, 1]) > 3.0]
    pads = {}
    for point in dish:
        key = ("+X" if point[0] > 0 else "-X") if abs(point[0]) > abs(point[1]) else ("+Y" if point[1] > 0 else "-Y")
        pads.setdefault(key, []).append(point)
    pad_points = {k: np.array(v)[::subsample] for k, v in pads.items()}
    pad_centers = {k: np.r_[v[:, :2].mean(0), v[:, 2].min()] for k, v in pad_points.items()}
    return pad_points, pad_centers
