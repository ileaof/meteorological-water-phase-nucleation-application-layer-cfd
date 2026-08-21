"""Physical constants and single-moment bulk-microphysics category parameters.

Every value carries its source.  Thermodynamic constants match the repo
convention (see ``meteorological_flow.thermodynamics``) so the microphysics stays
consistent with the validated nucleation kernel and the 3D flow solver.

Category (hydrometeor-class) parameters follow the classic single-moment bulk
schemes:

* Lin, Farley & Orville (1983), *J. Climate Appl. Meteor.* **22**, 1065.
* Rutledge & Hobbs (1983, 1984), *J. Atmos. Sci.* **40**/**41**.
* Hong & Lim (2006, WSM6), *J. Korean Meteor. Soc.* **42**, 129.

Coefficients that are empirical fits (fall-speed a,b; distribution intercepts
N0; autoconversion thresholds) are flagged ``EMPIRICAL`` inline so a reader can
audit exactly which numbers are tuned rather than derived.

All SI: kg, m, s, K, Pa, J.
"""
from __future__ import annotations

import math

# --- thermodynamics (match meteorological_flow.thermodynamics) --------------
R_d = 287.058          # J kg^-1 K^-1  dry-air gas constant
R_v = 461.5            # J kg^-1 K^-1  water-vapour gas constant
cp_d = 1005.0          # J kg^-1 K^-1  dry-air c_p
cp_v = 1846.0          # J kg^-1 K^-1  water-vapour c_p
cw = 4186.0            # J kg^-1 K^-1  liquid-water specific heat
ci = 2106.0            # J kg^-1 K^-1  ice specific heat
EPS = 0.62197          # M_w / M_d
g0 = 9.81              # m s^-2
P0_REF = 100000.0      # Pa, potential-temperature reference pressure
T0 = 273.15            # K, melting point
Tt = 273.16            # K, triple point

# Latent heats [J kg^-1] at 0 degC (constant first-order values, as in the
# flow solver; a T-dependent form is a documented future upgrade).
Lv = 2.501e6           # vaporisation / condensation
Ls = 2.836e6           # sublimation / deposition
Lf = Ls - Lv           # fusion (freezing / melting) ~ 3.34e5

# --- bulk densities [kg m^-3] ----------------------------------------------
rho_w = 1000.0         # liquid water
rho_i = 917.0          # cloud ice (hexagonal)
rho_s = 100.0          # snow (aggregate, low density)      Lin83
rho_g = 400.0          # graupel (rimed)                    Lin83
rho_h = 900.0          # hail (near solid ice)              Lin83

# --- Marshall-Palmer / exponential intercepts N0 [m^-4] --------------------
# N(D) = N0 exp(-lambda D).  EMPIRICAL fits.
N0_r = 8.0e6           # rain          Marshall & Palmer (1948)
N0_s = 3.0e6           # snow          Gunn & Marshall (1958) / Lin83
N0_g = 4.0e6           # graupel       Lin83
N0_h = 4.0e4           # hail          Lin83 (fewer, larger particles)

# --- fall-speed power laws  V(D) = a * D**b   (V in m/s, D in m) ------------
# EMPIRICAL fits (mass-weighted closed form applied in size_distributions.py).
# rain: Liu & Orville (1969) / Lin83;  snow: Locatelli & Hobbs (1974);
# graupel & hail: drag-balance sqrt law V = a D^0.5 with C_d ~ 0.6.
VT_A = {"rain": 841.99667, "snow": 11.72, "graupel": 124.0, "hail": 140.0}
VT_B = {"rain": 0.8,       "snow": 0.41,  "graupel": 0.66,  "hail": 0.5}

# reference air density for the (rho0/rho)^0.4 fall-speed correction (Foote &
# du Toit 1969); density-corrected V grows in thinner air aloft.
RHO0_VT = 1.225        # kg m^-3

# --- warm-rain (Kessler 1969) ----------------------------------------------
# EMPIRICAL.  Autoconversion q_c -> q_r above a threshold; accretion of cloud
# by rain; both widely used defaults.
KESSLER_K1 = 1.0e-3    # s^-1     autoconversion rate
KESSLER_QC_CRIT = 1.0e-3  # kg/kg cloud-water autoconversion threshold
KESSLER_K2 = 2.2       # accretion coefficient (q_r^0.875 form)

# --- ice processes ----------------------------------------------------------
# Heterogeneous IN concentration: Fletcher (1962) N_IN = N0f exp(beta (T0-T)).
FLETCHER_N0 = 1.0e-2   # m^-3     EMPIRICAL
FLETCHER_BETA = 0.6    # K^-1     EMPIRICAL
# Immersion freezing of cloud droplets: Bigg (1953) volume law.
BIGG_A = 0.66          # K^-1     EMPIRICAL
BIGG_B = 100.0         # m^-3 s^-1 EMPIRICAL
# ice -> snow autoconversion (aggregation) threshold, Lin83.
QI_CRIT_SNOW = 1.0e-3  # kg/kg    EMPIRICAL
QI_AUTO_RATE = 1.0e-3  # s^-1     EMPIRICAL

# --- riming / graupel conversion -------------------------------------------
# snow -> graupel when rimed enough (rime mass fraction), Lin83-style threshold.
QS_CRIT_GRAUPEL = 6.0e-4   # kg/kg EMPIRICAL
RIME_TO_GRAUPEL_RATE = 1.0e-3  # s^-1 EMPIRICAL
E_COLLECT = 1.0        # riming/collection efficiency (default; documented)

# --- hail growth ------------------------------------------------------------
# Schumann-Ludlam wet-growth limit and dry/wet regime handling are computed in
# processes.hail from a surface latent-heat balance; these are gates only.
QG_CRIT_HAIL = 1.0e-3          # kg/kg graupel embryo source threshold  EMPIRICAL
HAIL_LWC_CRIT = 1.0e-3         # kg/m^3 supercooled LWC for wet growth   EMPIRICAL
HAIL_UPDRAFT_CRIT = 10.0       # m/s minimum updraft to loft hail embryos EMPIRICAL
HAIL_RESIDENCE_CRIT = 60.0     # s minimum residence in growth zone      EMPIRICAL

# --- ventilation (rain evaporation / sublimation) --------------------------
# f = a_vent + b_vent * Sc^(1/3) * Re^(1/2)  (Rutledge & Hobbs 1983).
VENT_A = 0.78
VENT_B = 0.31
DIFF_VAPOR = 2.26e-5   # m^2/s  water-vapour diffusivity in air (~0 degC)
K_THERM = 2.43e-2      # W m^-1 K^-1 thermal conductivity of air
NU_AIR = 1.5e-5        # m^2/s  kinematic viscosity of air
SC = NU_AIR / DIFF_VAPOR   # Schmidt number

# numerical
TINY = 1.0e-20
QSMALL = 1.0e-12       # kg/kg below which a category is treated as empty

# gamma-function shortcut for the mass-weighted V_t closed form
GAMMA4 = math.gamma(4.0)   # = 6

__all__ = [name for name in dir() if not name.startswith("_")]
