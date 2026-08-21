"""Configuration for the bulk-microphysics scheme and precipitation diagnostics.

Every microphysical process can be switched off individually; disabling a
process both removes its tendency AND makes the diagnostics report the matching
reason code (e.g. ``NO_RIMING_MODEL``) so the confidence model stays honest
about what was actually computed.  This is how "disable microphysics restores
thermodynamic-only diagnostics" (validation test 17) is realised.

The reporting thresholds (0.50 for rain/snow/graupel, 0.75 for hail) are kept
exactly as in the existing application layer -- they are *reporting safeguards*,
not substitutes for the process modelling.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProcessSwitches:
    """Master on/off flags for each family of microphysical processes."""
    condensation: bool = True       # saturation adjustment (cloud water)
    autoconversion: bool = True     # q_c -> q_r (Kessler)
    accretion: bool = True          # rain collects cloud water
    rain_evaporation: bool = True   # q_r -> q_v in subsaturated air
    ice_nucleation: bool = True     # q_v/q_c -> q_i (kernel + Fletcher/Bigg)
    deposition: bool = True         # vapour deposition/sublimation on ice/snow
    aggregation: bool = True        # q_i -> q_s
    riming: bool = True             # supercooled q_c onto snow/graupel
    graupel_conversion: bool = True # rimed snow -> graupel
    graupel_melting: bool = True    # q_g -> q_r above 0 degC
    hail_growth: bool = True        # embryo + wet/dry growth
    hail_melting: bool = True       # q_h -> q_r below the freezing level
    sedimentation: bool = True      # terminal fall of q_r,q_s,q_g,q_h

    def all_off(self) -> "ProcessSwitches":
        return ProcessSwitches(**{f: False for f in self.__dataclass_fields__})


@dataclass
class MicrophysicsConfig:
    scheme: str = "single_moment"           # single_moment | (double_moment: future)
    processes: ProcessSwitches = field(default_factory=ProcessSwitches)

    # nucleation-source coupling
    stochastic_nucleation: bool = False     # Poisson embryo count vs mean-field
    embryo_radius_liquid: float = 1.0e-6    # m, initial droplet radius
    embryo_radius_ice: float = 5.0e-6       # m, initial ice-crystal radius
    vapour_limited: bool = True             # cap nucleation source at available q_v
    # partition of the nucleation source to avoid double counting between
    # aerosol activation and the Eq.39 shifted-equilibrium heterogeneous model.
    activation_pathway: str = "eq39"        # eq39 | ccn | homogeneous

    # reporting thresholds (safeguards; NOT process substitutes)
    threshold_rain: float = 0.50
    threshold_snow: float = 0.50
    threshold_graupel: float = 0.50
    threshold_hail: float = 0.75

    # numerical / validity envelope
    T_valid: tuple = (233.0, 320.0)         # K, correlation validity for diagnostics
    conservation_tol: float = 1.0e-9        # rel. total-water error tolerated
    seed: int = 20260821                    # RNG seed (reproducibility)

    def threshold(self, category: str) -> float:
        return {
            "rain": self.threshold_rain, "snow": self.threshold_snow,
            "graupel": self.threshold_graupel, "hail": self.threshold_hail,
        }[category]


def from_dict(d: dict) -> MicrophysicsConfig:
    """Build a MicrophysicsConfig from a plain dict (e.g. parsed YAML).

    Recognised keys mirror the dataclass fields; a ``processes`` sub-mapping sets
    individual switches.  Unknown keys are ignored so configs stay forward
    compatible.  The reporting thresholds may be overridden but default to the
    fixed 0.50 / 0.75 safeguards.
    """
    d = d or {}
    proc = ProcessSwitches()
    for k, v in (d.get("processes") or {}).items():
        if hasattr(proc, k):
            setattr(proc, k, bool(v))
    kw = dict(processes=proc)
    for key in ("scheme", "stochastic_nucleation", "embryo_radius_liquid",
                "embryo_radius_ice", "vapour_limited", "activation_pathway",
                "threshold_rain", "threshold_snow", "threshold_graupel",
                "threshold_hail", "conservation_tol", "seed"):
        if key in d:
            kw[key] = d[key]
    if "T_valid" in d:
        kw["T_valid"] = tuple(d["T_valid"])
    return MicrophysicsConfig(**kw)


def from_yaml(path: str) -> MicrophysicsConfig:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return from_dict(raw if isinstance(raw, dict) else {})


__all__ = ["MicrophysicsConfig", "ProcessSwitches", "from_dict", "from_yaml"]
