"""s1ship: validation-first ship detection in Sentinel-1 SAR imagery.

A CFAR (constant false-alarm rate) detector for ships in Sentinel-1 IW GRD
intensity, with gamma or K-distributed sea clutter, land masking, clustering of
detected pixels into targets with length estimates, and validation against AIS.
The detection science is independent of where the pixels come from: a
synthetic backend with injected targets of known position, size and
signal-to-clutter ratio runs everything offline, and two real-data backends
(Google Earth Engine, Microsoft Planetary Computer) fetch the same scene.

    config       areas of interest, sensor constants, detector defaults, sources
    clutter      gamma and K intensity models, ENL / K-shape estimators, CFAR thresholds
    cfar         cell-averaging and two-parameter CFAR with guard/background windows
    targets      clustering detections into targets; centroid, length, heading
    synthetic    multi-look speckle scenes with injected point and extended targets
    geo          UTM projection, the scene pixel grid, azimuth shift and ambiguity geometry
    scene        the Scene contract every backend returns
    data         synthetic, Earth Engine and Planetary Computer backends; fixed-structure mask
    pipeline     masks + clutter fit + CFAR + clustering + VH confirmation, in one call
    ais          MarineCadastre AIS download, bounding-box filter, positions at SAR time
    validation   matching to AIS; Pd by length, unexplained detections per km^2, dark candidates
    plotting     report figures
"""

__version__ = "0.1.0"
