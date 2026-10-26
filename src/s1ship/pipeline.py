"""One call from a Scene to a target list: masks, clutter fit, CFAR, clustering, VH check."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from . import cfar, config, targets
from .scene import Scene


@dataclass
class DetectionRun:
    result: cfar.CfarResult
    fit: cfar.ClutterFit | None
    targets: pd.DataFrame
    water: np.ndarray                      # mask actually searched (land buffer, structures removed)
    pixel_m: float = config.S1_PIXEL_M
    vh_result: cfar.CfarResult | None = None
    vh_fit: cfar.ClutterFit | None = None

    @property
    def water_km2(self) -> float:
        """Area actually tested (pixels with a CFAR decision)."""
        return float(self.result.valid.sum()) * self.pixel_m ** 2 / 1e6


def detect_scene(scene: Scene, pol: str = "vv", method: str = "ca", clutter_model: str = "k",
                 pfa: float = config.CFAR.pfa, land_buffer_m: float = config.LAND_BUFFER_M,
                 guard: int = config.CFAR.guard_px, background: int = config.CFAR.background_px,
                 censor: bool = config.CFAR.censor, t_model: str = "gaussian",
                 exclude: np.ndarray | None = None, vh_confirm: bool = True,
                 vh_pfa: float = config.VH_CONFIRM_PFA) -> DetectionRun:
    """Detect and measure targets in one polarisation of ``scene``.

    clutter_model: "k" fits the K shape to the scene (``cfar.fit_clutter``);
    "gamma" uses the nominal IW GRDH looks (4.4) with no texture.
    ``exclude``: pixels never searched (e.g. the fixed-structure mask).
    With ``vh_confirm`` and a VH channel, VH is run through its own CA-CFAR at
    ``vh_pfa`` with a K model fitted to the VH sea (whatever ``clutter_model``
    says for VV: the confirming channel must hold its own rate), and each VV
    target gets ``confirmed`` (a VH detection within VH_CONFIRM_DILATE_PX of its
    pixels) and ``confirm_peak_db``.
    """
    img = scene.vv if pol == "vv" else scene.vh
    if img is None:
        raise ValueError(f"scene has no {pol} channel")
    water = scene.water_buffered(land_buffer_m) & np.isfinite(img)
    if exclude is not None:
        water &= ~ndimage.binary_dilation(exclude, iterations=config.STRUCTURE_DILATE_PX)
    fit, L, nu = _clutter(img, water, clutter_model, guard, background)
    res = cfar.detect(img, water, method, pfa=pfa, L=L, nu=nu, guard=guard, background=background,
                      censor=censor, t_model=t_model)

    confirm = ratio = vh_res = vh_fit = None
    other = scene.vh if pol == "vv" else scene.vv
    if vh_confirm and other is not None:
        ow = water & np.isfinite(other)
        vh_fit, Lv, nuv = _clutter(other, ow, "k", guard, background)
        vh_res = cfar.detect(other, ow, "ca", pfa=vh_pfa, L=Lv, nu=nuv, guard=guard, background=background,
                             censor=censor)
        confirm = ndimage.binary_dilation(vh_res.detections, iterations=config.VH_CONFIRM_DILATE_PX)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(vh_res.background > 0, other / vh_res.background, np.nan)

    tg = targets.measure(res.detections, img, res.background, scene.grid, res.threshold,
                         confirm=confirm, confirm_ratio=ratio)
    return DetectionRun(res, fit, tg, water, scene.grid.pixel_m, vh_res, vh_fit)


def _clutter(img, water, model, guard, background):
    if model == "k":
        fit = cfar.fit_clutter(img, water, guard=guard, background=background)
        return fit, fit.L, fit.nu
    if model == "gamma":
        return None, config.S1_ENL_IW_GRDH, np.inf
    raise ValueError(f"unknown clutter model {model!r}")
