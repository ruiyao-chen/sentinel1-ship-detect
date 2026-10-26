"""The Scene: one SAR acquisition over an AOI, whatever backend produced it.

Every backend (synthetic, Earth Engine, Planetary Computer) returns this, and
the detector, clustering and validation code only ever see this.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .geo import Grid


@dataclass
class TruthTarget:
    """An injected target (synthetic backend only)."""
    id: int
    row: float
    col: float
    length_m: float
    width_m: float
    heading_deg: float          # clockwise from north
    scr_db: float               # peak target power / local clutter mean
    kind: str = "ship"          # "point", "ship" or "structure"
    ais: bool = True            # does it broadcast AIS?
    sog_kn: float = 0.0
    cog_deg: float = 0.0


@dataclass
class Scene:
    vv: np.ndarray                         # linear intensity (sigma0 or gamma0)
    water: np.ndarray                      # bool, water before any land buffer
    grid: Grid
    vh: np.ndarray | None = None
    time_utc: str | None = None            # ISO 8601, acquisition (scene centre) time
    source: str = "synthetic"
    scene_id: str = ""
    incidence_deg: float | None = None     # mean over the AOI
    look_azimuth_deg: float | None = None  # ground-range direction, clockwise from north
    orbit_pass: str | None = None
    truth: list[TruthTarget] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def incidence_at(self, x, y):
        """Incidence angle (deg) at UTM x, y: the annotation plane if known, else the scene mean."""
        coef = self.meta.get("incidence_plane")
        if not coef:
            return np.full(np.shape(x), self.incidence_deg, dtype=float)
        a, b, c = coef
        return a + b * (np.asarray(x) - self.grid.x0) / 1e3 + c * (np.asarray(y) - self.grid.y0) / 1e3

    def water_buffered(self, buffer_m: float) -> np.ndarray:
        """Water with land grown by ``buffer_m`` (removes coast, piers, breakwaters)."""
        n = int(round(buffer_m / self.grid.pixel_m))
        if n <= 0:
            return self.water.copy()
        land = ~self.water
        disk = _disk(n)
        return ~ndimage.binary_dilation(land, structure=disk) & np.isfinite(self.vv)

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        header = {k: getattr(self, k) for k in ("time_utc", "source", "scene_id", "incidence_deg",
                                                "look_azimuth_deg", "orbit_pass", "meta")}
        header["grid"] = self.grid.as_dict()
        header["truth"] = [asdict(t) for t in self.truth]
        arrays = {"vv": self.vv.astype(np.float32), "water": self.water}
        if self.vh is not None:
            arrays["vh"] = self.vh.astype(np.float32)
        np.savez_compressed(path, header=json.dumps(header), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        z = np.load(path, allow_pickle=False)
        h = json.loads(str(z["header"]))
        g = h.pop("grid")
        grid = Grid(g["epsg"], g["x0"], g["y0"], g["pixel_m"], tuple(g["shape"]))
        truth = [TruthTarget(**t) for t in h.pop("truth")]
        return cls(vv=z["vv"].astype(float), water=z["water"].astype(bool), grid=grid,
                   vh=z["vh"].astype(float) if "vh" in z else None, truth=truth, **h)


def _disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r
