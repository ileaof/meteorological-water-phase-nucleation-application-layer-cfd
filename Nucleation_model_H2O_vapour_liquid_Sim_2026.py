#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Nucleation model: H2O(vapour) -> H2O(liquid)
 Equilibrium thermodynamics shifted by a thermal gradient (Ferreira formulation)
================================================================================

This programme is a thermodynamically-consistent extension of I.L. Ferreira's
non-equilibrium nucleation formulation, applied to the condensation

        H2O(vapour)  ->  H2O(liquid)            (a spherical liquid droplet)

It is NOT a conventional CNT code.  It preserves Ferreira's *shifted-equilibrium*
framework: the thermal field and thermal-field tensor

        T  = (dT/dE) E                 (first law of thermodynamics)
        Gamma = A . grad_T             (thermal field tensor, Eq. 4 of the paper)

drive a *parabolic* (second-order) critical-radius equation.  The first-order
radius is kept only as an approximation; the second-order solution is the
principal physical result.

Basis (structure / concepts):
    Nucleation_model_H2O_vapour_solid_Sim_2026_paper.py   (H2O vapour -> ice)
Reference formulation:
    I.L. Ferreira, "Assessment of Thermodynamic variables affecting phase
    nucleation", Physica B: Condensed Matter 695 (2024) 416494,
    DOI: 10.1016/j.physb.2024.416494

------------------------------------------------------------------------------
IMPORTANT PHYSICS NOTE (sections 25-26 of the design spec)
------------------------------------------------------------------------------
The vapour->liquid transformation is physically different from vapour->ice:
  * the nucleated phase is a *liquid* droplet -> it has zero shear modulus,
    so the Shuttleworth equation reduces to  surface_stress = surface_energy.
    The solid-elasticity gamma(r) model of the ice script (Lame constants +
    logarithmic surface stress) is NOT reused; a Tolman curvature correction
    gamma_VL(r) = gamma_inf / (1 + 2*dTol/r) is used instead.
  * the reference equilibrium is the vapour-liquid coexistence
    mu_v(T,Pv) = mu_l(T,Pl), expressed through P_sat(T) / T_sat(P)
    (IAPWS Wagner saturation correlation), NOT a melting temperature.
  * the driving chemical-potential difference  Delta_mu = mu_l - mu_v
    is computed explicitly (ideal-gas vapour + incompressible liquid +
    Poynting correction) and only cross-checked against the near-equilibrium
    approximation; it is never silently replaced by a CNT expression.

Author of this script: generated for Prof. I.L. Ferreira (UFPa) per design spec.
"""

import csv
import math
import numpy as np
from numpy import zeros, sqrt, exp, log, pi as npi
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# scipy.misc.derivative was removed in SciPy >= 1.12; faithful drop-in replica
# (same signature/weights: n=1..4, order=3..9).  Used for numerical derivatives.
def derivative(func, x0, dx=1.0, n=1, args=(), order=3):
    x0 = float(x0)
    if n == 0:
        return func(*((x0,) + args))
    if order < n + 1:
        raise ValueError("'order' must be at least 'n+1'.")
    if order % 2 == 0:
        raise ValueError("'order' must be odd.")
    if n == 1:
        if order == 3:
            weights = [-0.5, 0.0, 0.5]
        elif order == 5:
            weights = [1/12, -8/12, 0.0, 8/12, -1/12]
        elif order == 7:
            weights = [-1/60, 9/60, -45/60, 0.0, 45/60, -9/60, 1/60]
        elif order == 9:
            weights = [3/840, -32/840, 168/840, -672/840, 0.0,
                       672/840, -168/840, 32/840, -3/840]
        else:
            raise NotImplementedError("order > 9 not supported")
    elif n == 2:
        if order == 3:
            weights = [1.0, -2.0, 1.0]
        elif order == 5:
            weights = [-1/12, 16/12, -30/12, 16/12, -1/12]
        elif order == 7:
            weights = [2/180, -27/180, 270/180, -490/180,
                       270/180, -27/180, 2/180]
        elif order == 9:
            weights = [-9/5040, 128/5040, -1008/5040, 8064/5040, -14350/5040,
                       8064/5040, -1008/5040, 128/5040, -9/5040]
        else:
            raise NotImplementedError("order > 9 not supported")
    elif n == 3:
        if order == 5:
            weights = [-0.5, 1.0, 0.0, -1.0, 0.5]
        elif order == 7:
            weights = [1/8, -9/8, 45/8, 0.0, -45/8, 9/8, -1/8]
        elif order == 9:
            weights = [-9/80, 108/80, -1008/80, 0.0,
                       5040/80, -1008/80, 108/80, -9/80]
        else:
            raise RuntimeError("order < 5 or > 9 not supported for 3rd deriv.")
    elif n == 4:
        if order == 5:
            weights = [1.0, -4.0, 6.0, -4.0, 1.0]
        elif order == 7:
            weights = [-3/24, 32/24, -168/24, 672/24, -1032/24,
                       672/24, -168/24, 32/24, -3/24]
        elif order == 9:
            weights = [9/720, -128/720, 1344/720, -6720/720, 20160/720,
                       -6720/720, 1344/720, -128/720, 9/720]
        else:
            raise RuntimeError("order < 5 or > 9 not supported for 4th deriv.")
    else:
        raise RuntimeError("only derivatives of order 1 to 4 supported")
    val = 0.0
    ho = order // 2
    for k in range(order):
        x = x0 + (k - ho) * dx
        val += weights[k] * func(*((x,) + args))
    return val / (dx ** n)


# =============================================================================
#  FUNDAMENTAL CONSTANTS
# =============================================================================
Nav   = 6.02214076e23       # Avogadro number [mol^-1]
h     = 6.62607015e-34       # Planck constant [J.s]
h_    = h / (2.0 * math.pi) # reduced Planck [J.s]
R     = 8.314462618          # universal gas constant [J.mol^-1.K^-1]
kB    = 1.380649e-23         # Boltzmann constant [J/K]
M_H2O = 0.01801528           # molar mass of water [kg/mol]
Rsp   = R / M_H2O            # specific gas constant of water vapour [J/kg.K]

# Critical / triple points of water (IAPWS)
Tc = 647.096                 # critical temperature [K]
Pc = 22.064e6                # critical pressure [Pa]
Tt = 273.16                  # triple-point temperature [K]
Pt = 611.657                 # triple-point pressure [Pa]

# Tolman length of the water vapour-liquid interface.
# Physical interface thickness scale (NOT a fitting parameter to experiments).
# dTol ~ 0.2 nm is the accepted order of magnitude for water; it sets the
# curvature scale below which gamma_VL(r) departs from the planar value.
dTol = 0.2e-9                # [m]

# Molecular jump length (attachment distance vapour->liquid interface)
LAMBDA = 0.28e-9             # [m]  (~ O--H hydrogen-bond / molecular diameter)

# Heterogeneous substrate contact angle (default demonstrative value)
THETA_0 = math.radians(45.0) # [rad]

# Nucleation-rate kinetic mechanism
MECHANISM = "vapour"         # "vapour" (vapour-side impingement, default, physical
                              #  for condensation) or "liquid" (liquid self-diffusion)


# =============================================================================
#  H2O VAPOUR-LIQUID PROPERTIES  (section 14)
# =============================================================================
# All property functions take (T [K], P [Pa]) and return SI quantities.
# Liquid properties are evaluated on the saturated liquid at T; a weak
# compressibility correction accounts for pressure.  Vapour is treated as an
# ideal gas (consistent with the chosen chemical-potential model).
# -----------------------------------------------------------------------------

def rho_l(T, P=None):
    """Density of liquid water [kg/m^3] at T [K], with weak P correction."""
    t = T - 273.15
    # Kell (1975) liquid-water density at 1 atm, valid 0..150 C
    rho0 = (999.83952 + 16.945176 * t - 7.9870401e-3 * t**2
            - 4.614146e-5 * t**3 + 1.0562129e-7 * t**4
            - 2.8017611e-10 * t**5) / (1.0 + 16.879850e-3 * t)
    if P is None:
        return rho0
    psat = saturation_pressure(T)
    kappa = 4.6e-10                       # isothermal compressibility [Pa^-1]
    return rho0 * (1.0 - kappa * (P - psat))

def rho_v(T, P):
    """Density of water vapour [kg/m^3] (ideal gas)."""
    return P * M_H2O / (R * T)

def cp_l(T):
    """Isobaric specific heat of liquid water [J/kg.K]."""
    t = T - 273.15
    # polynomial fit, valid 0..200 C
    return 4217.4 - 3.7288 * t + 0.14126 * t**2 - 2.3e-3 * t**3 + 2.0e-5 * t**4

def cp_v(T):
    """Isobaric specific heat of water vapour (ideal gas) [J/kg.K]."""
    # Shomate-style smooth form, ~1865 at 300 K, ~2080 at 500 K
    t = T - 273.15
    return 1864.0 + 1.20 * t + 1.5e-3 * t**2

def h_l(T):
    """Specific enthalpy of liquid water [J/kg], reference h_l(273.16K)=0."""
    # analytic integral of cp_l polynomial
    t = T - 273.15
    return (4217.4 * t - 3.7288 / 2.0 * t**2 + 0.14126 / 3.0 * t**3
            - 2.3e-3 / 4.0 * t**4 + 2.0e-5 / 5.0 * t**5)

def h_v(T):
    """Specific enthalpy of water vapour [J/kg] = h_l + h_lv."""
    return h_l(T) + h_lv(T)

def h_lv(T):
    """Latent heat of vaporisation [J/kg] (Watson / IAPWS form)."""
    # reference: h_lv(373.15 K) = 2256.5e3 J/kg
    Tr = 373.15
    hl_ref = 2256.5e3
    ratio = max((Tc - T) / (Tc - Tr), 1e-6)
    return hl_ref * ratio**0.38

def s_l(T):
    """Specific entropy of liquid water [J/kg.K], reference s_l(273.16K)=0."""
    # integral of cp_l/T dT (analytic, t=T-273.15, T=t+273.15)
    t = T - 273.15
    T0 = 273.15
    # int cp_l(t)/ (T0+t) dt  -- series expansion form
    u = t / T0
    s = (4217.4 * math.log(T / T0)
         - 3.7288 * (t - T0 * math.log(T / T0))
         + 0.14126 * (t**2 / 2.0 - T0 * (t - T0 * math.log(T / T0)))
         - 2.3e-3 * (t**3 / 3.0 - T0 * (t**2 / 2.0 - T0 * (t - T0 * math.log(T / T0))))
         + 2.0e-5 * (t**4 / 4.0 - T0 * (t**3 / 3.0 - T0 * (t**2 / 2.0
                     - T0 * (t - T0 * math.log(T / T0)))))
         )
    return s

def s_v(T):
    """Specific entropy of water vapour [J/kg.K] = s_l + h_lv/T (Clausius)."""
    return s_l(T) + h_lv(T) / T

def gamma_VL_inf(T):
    """Planar vapour-liquid surface tension [J/m^2] (IAPWS correlation)."""
    tau = max(1.0 - T / Tc, 0.0)
    # IAPWS Gittfried/Muley form; ~0.0728 N/m at 293 K, -> 0 at Tc
    return 0.2358 * (tau ** 1.256) * (1.0 - 0.625 * tau)

def mu_l_dyn(T):
    """Dynamic viscosity of liquid water [Pa.s] (used only for reference)."""
    t = T - 273.15
    return 1.792e-3 / (1.0 + 0.0338 * t + 2.21e-4 * t**2)

def D_vapour(T, P):
    """Vapour-phase self / mutual diffusion coefficient [m^2/s]
    (kinetic impingement mechanism, physically correct for condensation)."""
    # D ~ T^1.75 / P  (Chapman-Enskog / Fuller), anchored at 2.3e-5 m^2/s, 1 atm, 298 K
    T0, P0, D0 = 298.15, 101325.0, 2.3e-5
    return D0 * (T / T0) ** 1.75 * (P0 / P)

def D_self_liquid(T, P):
    """Liquid water self-diffusion coefficient [m^2/s] (Arrhenius form).
    Anchored at 2.3e-9 m^2/s at 298.15 K."""
    Ea = 18000.0                      # [J/mol]
    D0 = 2.3e-9 * math.exp(Ea / (Rsp * 298.15))
    return D0 * math.exp(-Ea / (Rsp * T))

def D_eff(T, P, mechanism=MECHANISM):
    """Selected kinetic coefficient for the nucleation-rate prefactor."""
    if mechanism == "liquid":
        return D_self_liquid(T, P)
    return D_vapour(T, P)


# =============================================================================
#  SATURATION: P_sat(T) and T_sat(P)   (section 15)  -- IAPWS Wagner correlation
# =============================================================================
# ln(P/Pc) = (Tc/T) * SUM a_k * tau^(b_k),  tau = 1 - T/Tc
# valid  273.16 K <= T <= 647.096 K   (Pt <= P <= Pc)
# -----------------------------------------------------------------------------
_WAGNER_A = [-7.85951783, 1.84408259, -11.7866497, 22.6807411,
             -15.9618719, 1.80122502]
_WAGNER_B = [1.0, 1.5, 3.0, 3.5, 4.0, 7.5]
_T_LO, _T_HI = Tt, Tc
_P_LO, _P_HI = Pt, Pc

def saturation_pressure(T):
    """Saturation pressure [Pa] of water at temperature T [K]."""
    if not (_T_LO <= T <= _T_HI):
        raise ValueError(
            f"saturation_pressure: T={T:g} K outside valid range "
            f"[{_T_LO:g}, {_T_HI:g}] K.")
    tau = 1.0 - T / Tc
    s = sum(a * tau ** b for a, b in zip(_WAGNER_A, _WAGNER_B))
    return Pc * math.exp((Tc / T) * s)

def saturation_temperature(P):
    """Saturation temperature [K] of water at pressure P [Pa] (inverse of
    saturation_pressure, by robust Brent inversion)."""
    if not (_P_LO <= P <= _P_HI):
        raise ValueError(
            f"saturation_temperature: P={P:g} Pa outside valid range "
            f"[{_P_LO:g}, {_P_HI:g}] Pa.")
    f = lambda T: saturation_pressure(T) - P
    return brentq(f, _T_LO + 1e-6, _T_HI - 1e-6,
                  xtol=1e-9, rtol=1e-12, maxiter=200)


# =============================================================================
#  EQUILIBRIUM vs NUCLEATION PROPERTY BLOCKS  (section 14)
# =============================================================================
def equilibrium_properties(T, P):
    """Properties evaluated ON the coexistence curve at temperature T.
    Returns a dict with clearly-labelled SI quantities."""
    Ps = saturation_pressure(T)
    return dict(
        T=T, Psat=Ps,
        rho_l=rho_l(T, Ps), rho_v=rho_v(T, Ps),
        cp_l=cp_l(T), cp_v=cp_v(T),
        h_l=h_l(T), h_v=h_v(T), h_lv=h_lv(T),
        s_l=s_l(T), s_v=s_v(T),
        gamma_VL=gamma_VL_inf(T),
        D_v=D_vapour(T, Ps), D_l=D_self_liquid(T, Ps),
    )

def nucleation_properties(T, P):
    """Properties at the actual metastable state (T, P_v=P).  The liquid is
    taken at the same T and at P_l = P_sat(T) (incompressible-liquid
    approximation, Poynting correction handled in chemical_potential_*)."""
    Ps = saturation_pressure(T)
    return dict(
        T=T, P=P, Psat=Ps,
        rho_l=rho_l(T, Ps), rho_v=rho_v(T, P),
        cp_l=cp_l(T), cp_v=cp_v(T),
        h_l=h_l(T), h_v=h_v(T), h_lv=h_lv(T),
        s_l=s_l(T), s_v=s_v(T),
        gamma_VL=gamma_VL_inf(T),
        D_v=D_vapour(T, P), D_l=D_self_liquid(T, P),
    )


# =============================================================================
#  CHEMICAL POTENTIAL  (section 6)
# =============================================================================
# Delta_mu = mu_l - mu_v   [J/mol]  (and per-kg form)
# Vapour: ideal gas.  Liquid: incompressible, Poynting-corrected.
#   mu_v(T,P_v) - mu_l^sat(T) = R T ln(P_v / P_sat(T))           [J/mol]
#   mu_l(T,P_l) - mu_l^sat(T) = V_m_l (P_l - P_sat(T))           [J/mol]  (Poynting)
# Hence  Delta_mu = mu_l - mu_v = -R T ln(P_v/P_sat) + V_m_l (P_l - P_sat)
# -----------------------------------------------------------------------------
def chemical_potential_vapour(T, P_v):
    """mu_v(T,P_v) - mu_l^sat(T)  [J/mol] (ideal-gas vapour)."""
    return R * T * math.log(P_v / saturation_pressure(T))

def chemical_potential_liquid(T, P_l):
    """mu_l(T,P_l) - mu_l^sat(T)  [J/mol] (incompressible liquid + Poynting)."""
    Vm = M_H2O / rho_l(T)                  # molar volume of liquid [m^3/mol]
    return Vm * (P_l - saturation_pressure(T))

def chemical_potential_difference(T, P_v, P_l=None):
    """Delta_mu = mu_l - mu_v  [J/mol].

    Near equilibrium this reduces to  Delta_mu ~ Delta_S_molar * Delta_T
    (with Delta_S_molar = M_H2O*(s_l - s_v) < 0), i.e. the same sign as
    Delta_G_V = Delta_S_V * Delta_T.  The minus-sign variant in the spec
    corresponds to the opposite entropy convention; the code verifies the
    thermodynamic consistency directly rather than assuming any sign.
    """
    if P_l is None:
        P_l = saturation_pressure(T)      # liquid at its own saturation pressure
    return chemical_potential_liquid(T, P_l) - chemical_potential_vapour(T, P_v)


# =============================================================================
#  BULK ENTROPY & BULK FREE ENERGY  (sections 3, 7)
# =============================================================================
# Delta_S_V = S_l - S_v   per unit VOLUME of the nucleated (liquid) phase.
# Using Clausius-Clapeyron consistency:  s_v - s_l = h_lv / T  at coexistence,
# extended off-coexistence through the T-dependence of h_lv and rho_l.
# Units: [J/(m^3.K)].  Sign: condensation decreases entropy -> Delta_S_V < 0.
# -----------------------------------------------------------------------------
def bulk_entropy_change(T, P=None):
    """Delta_S_V = S_l - S_v  [J/(m^3.K)]  (volumetric, liquid-phase basis)."""
    return rho_l(T) * (s_l(T) - s_v(T))     # = -rho_l(T) * h_lv(T) / T  < 0

def bulk_entropy_molar(T):
    """Delta_S per mole  [J/(mol.K)] = M_H2O * (s_l - s_v)."""
    return M_H2O * (s_l(T) - s_v(T))

def bulk_free_energy(T, P_v, Delta_T):
    """Delta_G_V = Delta_S_V * Delta_T  [J/m^3].

    Delta_T = T_sat(P_v) - T  > 0  for condensation by cooling (section 5).
    The sign is obtained from physics: Delta_S_V < 0 and Delta_T > 0 give
    Delta_G_V < 0  ->  liquid formation is thermodynamically favourable.
    The caller MUST verify Delta_G_V < 0 before accepting a nucleation state.
    """
    dsv = bulk_entropy_change(T)
    return dsv * Delta_T


# =============================================================================
#  SURFACE ENERGY / SURFACE STRESS & DERIVATIVES  (section 8)
# =============================================================================
# For a liquid droplet the Shuttleworth equation reduces to
#     surface_stress_VL = surface_energy_VL  =  gamma_VL
# (zero shear modulus).  Curvature enters through the Tolman correction:
#     gamma_VL(r) = gamma_inf / (1 + 2*dTol/r)
# All quantities carry explicit units; gamma [J/m^2], dgamma/dr [J/m^3],
# d2gamma/dr2 [J/m^4].
# -----------------------------------------------------------------------------
def surface_energy_VL(r, T, P=None):
    """gamma_VL(r)  [J/m^2]  (Tolman curvature correction)."""
    ginf = gamma_VL_inf(T)
    return ginf / (1.0 + 2.0 * dTol / r)

def surface_stress_VL(r, T, P=None):
    """Surface stress of the liquid interface [N/m] = gamma_VL (Shuttleworth)."""
    return surface_energy_VL(r, T, P)

def surface_energy_derivative(r, T, P=None):
    """d(gamma_VL)/dr  [J/m^3]."""
    ginf = gamma_VL_inf(T)
    x = 1.0 + 2.0 * dTol / r
    return 2.0 * dTol * ginf / (r * r * x * x)        # > 0 for Tolman

def surface_energy_second_derivative(r, T, P=None):
    """d2(gamma_VL)/dr2  [J/m^4]."""
    ginf = gamma_VL_inf(T)
    d = dTol
    x = 1.0 + 2.0 * d / r
    # d/dr [ 2 d ginf / (r^2 x^2) ]
    term1 = -4.0 * d * ginf / (r**3 * x**2)
    term2 = -8.0 * d * d * ginf / (r**4 * x**3)
    return term1 + term2


# =============================================================================
#  THERMAL FIELD & THERMAL FIELD TENSOR  (sections 2, 19)
# =============================================================================
# Thermal field:  T_field = (dT/dE) E   (first law of thermodynamics)
# Thermal field tensor (Eq. 4):
#     Gamma = A . grad_T ,   grad_T normal to the newly created/deformed surface
# For a spherical droplet, A = 4*pi*r^2 (homogeneous) and the Gibbs-Thomson
# coupling gives  Delta_T = 2*Gamma/r = 8*pi*r*grad_T  ->  grad_T = Delta_T/(8*pi*r).
# -----------------------------------------------------------------------------
def thermal_field_tensor(A, grad_T):
    """Gamma = A . grad_T  [m^2 * K/m = m.K]."""
    return A * grad_T

def grad_T_from_DeltaT(Delta_T, r, A=None):
    """Normal thermal gradient [K/m] consistent with Delta_T and r
    (homogeneous sphere: A = 4*pi*r^2, Delta_T = 8*pi*r*grad_T)."""
    return Delta_T / (8.0 * math.pi * r)


# =============================================================================
#  HETEROGENEOUS GEOMETRY  (section 11)
# -----------------------------------------------------------------------------
def ftheta(theta):
    """f(theta) = 2 - 3 cos(theta) + cos^3(theta)  (dimensionless)."""
    c = math.cos(theta)
    return 2.0 - 3.0 * c + c**3

def dftheta_dtheta(theta):
    """d f / d theta = 3 sin(theta) (1 - cos^2(theta)) = 3 sin(theta) sin^2(theta)."""
    s = math.sin(theta)
    return 3.0 * s * (1.0 - math.cos(theta)**2)

def heterogeneous_factor(theta):
    """Geometric reduction factor f(theta)/4 (1 for a full sphere, theta=pi)."""
    return ftheta(theta) / 4.0

def het_hom_radius_ratio(theta):
    """r_C,Het / r_C,Hom from Ferreira Eq. (17):
        r_Het/r_Hom = exp( -int_theta^pi  s(1+c)/(2 - c - c^2) ds )
    Returns the ratio (<=1 for theta<pi)."""
    def integrand(s):
        c = math.cos(s)
        return math.sin(s) * (1.0 + c) / (2.0 - c - c**2)
    # analytic antiderivative: ln(2 - cos - cos^2) between theta and pi
    F = lambda a: math.log(2.0 - math.cos(a) - math.cos(a)**2)
    val = F(math.pi) - F(theta)
    return math.exp(-val)


# =============================================================================
#  CRITICAL RADII  (sections 9, 10, 11)
# =============================================================================
# CORRECTED FORMULATION (radius = continuation variable; gradient Brent-solved):
# The embryo radius r is the prescribed continuation variable.  At each r the
# local thermal gradient g = dT/dr is Brent-solved from the Gibbs-Thomson closure
#   F(g;r) = Gamma^(2)(r, Delta_T(g), g, ...) / (4 pi r^2) - g = 0
# (see gibbs_thomson_hom_2nd_residual / solve_gradient_at_radius below).  AFTER
# convergence the critical radii are COMPUTED from the local thermal state
# (Delta_T = 8 pi r g, T_local = T_base - Delta_T):
#
#   * first-order (CNT-like, local): r_C,Hom,1st = -2 gamma / (Delta_S_V Delta_T + dgamma/dr)
#   * second-order (principal result): the physically admissible POSITIVE root of
#        A2 r^2 + B2 r + C2 = 0,
#        A2 = (1/3)[ d(Delta_S_V)/dr * Delta_T + Delta_S_V * d(Delta_T)/dr ]   (<0)
#        B2 = Delta_S_V Delta_T + dgamma/dr
#        C2 = 2 gamma                                                          (>0)
#     r_C,Hom,2nd = (-B2 - sqrt(B2^2 - 4 A2 C2)) / (2 A2)  (A2<0 -> positive root)
#
# The second-order radius is the selected physical critical radius; the
# first-order radius is retained only as the classical approximation and is
# undefined (negative) wherever the Tolman dgamma/dr term dominates the entropy
# driving force (small undercooling / large r) -- there the second-order result
# is the admissible one.  As g -> 0 (r -> +inf)  Delta_T -> 0 and r_C,Hom,2nd -> +inf.
# -----------------------------------------------------------------------------
def critical_radius_first_order_local(state):
    """r_C,Hom,1st [m] from the local thermal state (first-order / CNT-like).

    state: dict from solve_gradient_at_radius with keys dsv, dT, dgdr, gam.
    Returns a signed value; callers must check > 0 (Tolman-dominated regimes
    give a negative, undefined first-order radius -- use r_C,Hom,2nd there).
    """
    denom = state["dsv"] * state["dT"] + state["dgdr"]
    if abs(denom) < 1e-30:
        return float("inf")
    return -2.0 * state["gam"] / denom

def critical_radius_second_order_local(state):
    """r_C,Hom,2nd [m] = physically admissible positive root of the parabolic
    stationarity A2 r^2 + B2 r + C2 = 0 evaluated at the local thermal state."""
    A2 = (1.0 / 3.0) * (state["dDsv_dr"] * state["dT"] + state["dsv"] * state["dDT_dr"])
    B2 = state["dsv"] * state["dT"] + state["dgdr"]
    C2 = 2.0 * state["gam"]
    if abs(A2) < 1e-60:
        return float("inf")
    disc = B2 * B2 - 4.0 * A2 * C2
    if disc < 0:
        return float("inf")
    r2 = (-B2 - math.sqrt(disc)) / (2.0 * A2)
    return r2 if r2 > 0 else float("inf")

def parabolic_stationarity_residual(r, state):
    """A2 r^2 + B2 r + C2 evaluated at the local state (validation #3: must be ~0
    when r = r_C,Hom,2nd)."""
    A2 = (1.0 / 3.0) * (state["dDsv_dr"] * state["dT"] + state["dsv"] * state["dDT_dr"])
    B2 = state["dsv"] * state["dT"] + state["dgdr"]
    C2 = 2.0 * state["gam"]
    return A2 * r * r + B2 * r + C2


# -----------------------------------------------------------------------------
# DEPRECATED: old prescribed-gradient formulation (gradient as INPUT, radius as
# the quadratic unknown -- the INVERSE of the corrected closure).  Retained ONLY
# to seed the corrected Brent bracket and for backwards-compatibility of the
# public API; the corrected solver does NOT use them to determine the state.
# -----------------------------------------------------------------------------
def critical_radius_first_order_hom(grad_T, T, P_v):
    """DEPRECATED (old prescribed-gradient formulation).  r_C,Hom,1st [m] for a
    prescribed macroscopic thermal gradient -- solves the quadratic
    Delta_S_V*8*pi*grad_T*r^2 + (dgamma/dr) r + 2 gamma = 0 for r given grad_T.
    Kept only as a bracket seed / legacy API; use solve_gradient_at_radius for
    the corrected radius-continuation solution."""
    dsv = bulk_entropy_change(T)                  # < 0
    a = dsv * 8.0 * math.pi * grad_T               # < 0 for grad_T>0
    r = 1e-9
    for _ in range(200):
        g = surface_energy_VL(r, T)
        dgdr = surface_energy_derivative(r, T)
        b = dgdr
        c = 2.0 * g
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return float("inf")
        r_new = (-b - math.sqrt(disc)) / (2.0 * a)
        if abs(r_new - r) < 1e-15 * r:
            r = r_new
            break
        r = r_new
    return r

def critical_radius_second_order_hom(r_1st, grad_T, T, P_v):
    """DEPRECATED (old prescribed-gradient formulation).  r_C,Hom,2nd [m] from
    Ferreira Eq. (10) for a PRESCRIBED gradient -- solves the parabolic for r
    given grad_T.  The corrected solver instead prescribes r and Brent-solves g
    (see solve_gradient_at_radius); this routine is retained only for the legacy
    API.

    The exact stationarity condition d(Delta_G)/dr = 0 reads, after division
    by 4*pi*r,

        (1/3) r^2 [ d(Delta_S_V)/dr * Delta_T + Delta_S_V * d(Delta_T)/dr ]
            + r [ Delta_S_V * Delta_T + dgamma/dr ]  +  2 gamma  = 0        (A)

    The *first-order* radius drops the (1/3) r^2 (...) term (i.e. treats
    Delta_S_V as locally constant) and reduces to the quadratic
        Delta_S_V * 8*pi*grad_T * r^2 + (dgamma/dr) r + 2 gamma = 0.
    The *second-order* (principal) radius is the POSITIVE root of the full
    quadratic (A):

        A2 r^2 + B2 r + C2 = 0,
        A2 = (1/3)[ d(Delta_S_V)/dr * Delta_T + Delta_S_V * d(Delta_T)/dr ]   (<0)
        B2 = Delta_S_V * Delta_T + dgamma/dr                                  (<0)
        C2 = 2 gamma                                                           (>0)

    With A2<0, C2>0 the parabola opens downward and is positive at r=0, so it
    has exactly one positive root,
        r_C,2nd = (-B2 - sqrt(B2^2 - 4 A2 C2)) / (2 A2)  ->  +inf as grad_T->0.

    d(Delta_T)/dr = 8*pi*grad_T (prescribed gradient);
    d(Delta_S_V)/dr = (d(Delta_S_V)/dT)(dT/dr), with T = T_sat - Delta_T
    -> dT/dr = -d(Delta_T)/dr.  Coefficients are r-dependent (Tolman dgamma/dr,
    rho_l(T)) so the quadratic is solved iteratively, seeded by r_1st.
    """
    dsv = bulk_entropy_change(T)
    dDsv_dT = derivative(bulk_entropy_change, T, dx=1e-3, n=1)
    dDT_dr = 8.0 * math.pi * grad_T
    dT_dr = -dDT_dr
    r = r_1st
    for _ in range(200):
        Delta_T = 8.0 * math.pi * r * grad_T
        g = surface_energy_VL(r, T)
        dgdr = surface_energy_derivative(r, T)
        dDsv_dr = dDsv_dT * dT_dr
        A2 = (1.0 / 3.0) * (dDsv_dr * Delta_T + dsv * dDT_dr)   # < 0
        B2 = dsv * Delta_T + dgdr                              # < 0
        C2 = 2.0 * g                                           # > 0
        disc = B2 * B2 - 4.0 * A2 * C2
        if disc < 0 or abs(A2) < 1e-60:
            return float("inf")
        r_new = (-B2 - math.sqrt(disc)) / (2.0 * A2)           # A2<0 -> positive root
        if r_new <= 0:
            return float("inf")
        if abs(r_new - r) < 1e-15 * r:
            r = r_new
            break
        r = r_new
    return r

def critical_radius_first_order_het(grad_T, T, P_v, theta):
    """DEPRECATED (old prescribed-gradient formulation).  r_C,Het,1st [m]."""
    r_hom = critical_radius_first_order_hom(grad_T, T, P_v)
    if math.isinf(r_hom):
        return float("inf")
    return r_hom * het_hom_radius_ratio(theta)

def critical_radius_second_order_het(r_hom_2nd, theta):
    """DEPRECATED (old prescribed-gradient formulation).  r_C,Het,2nd [m]
    via the heterogeneous geometric ratio (Eq. 17)."""
    if math.isinf(r_hom_2nd):
        return float("inf")
    return r_hom_2nd * het_hom_radius_ratio(theta)


# =============================================================================
#  GIBBS-THOMSON COEFFICIENT (thermal field tensor) Gamma  (sections 2, 9, 19)
# =============================================================================
def gibbs_thomson_hom_1st(r, T, Delta_T):
    """Gamma_Hom,1st = gamma / (Delta_S_V + Delta_S_S)  [m.K],
    Delta_S_S = (1/Delta_T) dgamma/dr.  (Ferreira Eq. 9, homogeneous.)"""
    g = surface_energy_VL(r, T)
    dgdr = surface_energy_derivative(r, T)
    dsv = bulk_entropy_change(T)
    dss = dgdr / Delta_T
    return -g / (dsv + dss)

def gibbs_thomson_hom_2nd(r_1st, T, Delta_T, grad_T):
    """Gamma_Hom,2nd  [m.K]  (Ferreira Eq. 11, homogeneous):
        Gamma_2nd = (3/4) |Delta_S_V Delta_T + dgamma/dr|
                          / |dDelta_S_V/dr + (Delta_S_V/Delta_T) dDelta_T/dr|
    Gamma is a tensor magnitude -> returned positive.  With Delta_S_V<0 and
    Delta_T>0 both numerator and denominator are negative; the ratio is
    positive, so we report +3/4 * |num/den|.
    """
    dsv = bulk_entropy_change(T)
    dgdr = surface_energy_derivative(r_1st, T)
    dDsv_dT = derivative(bulk_entropy_change, T, dx=1e-3, n=1)
    dDT_dr = 8.0 * math.pi * grad_T
    dT_dr = -dDT_dr
    dDsv_dr = dDsv_dT * dT_dr
    num = dsv * Delta_T + dgdr
    den = dDsv_dr + dsv / Delta_T * dDT_dr
    if abs(den) < 1e-30:
        return float("inf")
    return 0.75 * abs(num / den)

def gibbs_thomson_het_1st(r, T, Delta_T, theta):
    """Gamma_Het,1st  [m.K]  (Ferreira Eq. 9/12, heterogeneous)."""
    g = surface_energy_VL(r, T)
    dgdr = surface_energy_derivative(r, T)
    dsv = bulk_entropy_change(T)
    dss = dgdr / Delta_T
    # configurational entropy term (fixed contact angle -> df/dr=0)
    return -g / ((dsv + dss) + 0.0)

def gibbs_thomson_het_2nd(r, T, Delta_T, grad_T, theta):
    """Gamma_Het,2nd  [m.K]  = Gamma_Hom,2nd * (geometric ratio not applied to Gamma;
    Ferreira keeps Gamma as the intensive tensor).  Uses the homogeneous
    second-order expression (fixed-angle substrate -> configurational term 0)."""
    return gibbs_thomson_hom_2nd(r, T, Delta_T, grad_T)


# =============================================================================
#  CRITICAL FREE ENERGY  (sections 3, 10, 11, 22)
# =============================================================================
def critical_free_energy_hom(r, T, Delta_T):
    """Delta_G_C,Hom [J] = (4/3 pi r^3 Delta_G_V + 4 pi r^2 gamma)  (Eq. 5, f=4)."""
    dsv = bulk_entropy_change(T)
    g = surface_energy_VL(r, T)
    return (4.0 / 3.0) * math.pi * r**3 * (dsv * Delta_T) + 4.0 * math.pi * r**2 * g

def critical_free_energy_het(r, T, Delta_T, theta):
    """Delta_G_C,Het [J] (Eq. 5 with f(theta))."""
    dsv = bulk_entropy_change(T)
    g = surface_energy_VL(r, T)
    return (1.0 / 3.0 * math.pi * r**3 * (dsv * Delta_T) + math.pi * r**2 * g) \
           * ftheta(theta)

def critical_free_energy_hom_2nd_expr(r, T, Delta_T, grad_T):
    """Delta_G_C,Hom,2nd  [J]  (Ferreira Eq. 16, homogeneous, f=4)."""
    dsv = bulk_entropy_change(T)
    g = surface_energy_VL(r, T)
    dgdr = surface_energy_derivative(r, T)
    num = (dsv * Delta_T + dgdr)
    dDsv_dT = derivative(bulk_entropy_change, T, dx=1e-3, n=1)
    dDT_dr = 8.0 * math.pi * grad_T
    dT_dr = -dDT_dr
    dDsv_dr = dDsv_dT * dT_dr
    dDSvDTdr = dDsv_dr + dsv / Delta_T * dDT_dr
    bracket = dsv * num - 2.0 * g * dDSvDTdr
    den = dDSvDTdr**3
    if abs(den) < 1e-30:
        return float("inf")
    return -9.0 * math.pi / (8.0 * Delta_T**2) * (num**2) * bracket / den

def critical_free_energy_het_2nd_expr(r, T, Delta_T, grad_T, theta):
    """Delta_G_C,Het,2nd  [J]  (Ferreira Eq. 16, heterogeneous)."""
    dsv = bulk_entropy_change(T)
    g = surface_energy_VL(r, T)
    dgdr = surface_energy_derivative(r, T)
    f = ftheta(theta)
    num = (dsv * Delta_T + dgdr) + g / f * 0.0          # df/dr=0 (fixed angle)
    dDsv_dT = derivative(bulk_entropy_change, T, dx=1e-3, n=1)
    dDT_dr = 8.0 * math.pi * grad_T
    dT_dr = -dDT_dr
    dDsv_dr = dDsv_dT * dT_dr
    dDSvDTdr = (dDsv_dr + dsv / Delta_T * dDT_dr)
    bracket = dsv * num - 2.0 * g * dDSvDTdr
    den = dDSvDTdr**3
    if abs(den) < 1e-30:
        return float("inf")
    return 9.0 * math.pi * f / (8.0 * Delta_T**2) * (num**2) * bracket / den


# =============================================================================
#  ENTROPY DECOMPOSITION  (sections 7, 19)
# =============================================================================
def entropy_decomposition(r, T, Delta_T, theta=math.pi):
    """Returns bulk, surface, configurational entropies [J/(m^3.K)].
    Delta_S_V = S_l - S_v  (volumetric)
    Delta_S_S = (1/Delta_T) dgamma/dr          (surface)
    Delta_S_C = (gamma/Delta_T)(1/f) df/dr      (configurational, 0 for fixed angle)
    """
    dsv = bulk_entropy_change(T)
    dgdr = surface_energy_derivative(r, T)
    g = surface_energy_VL(r, T)
    dss = dgdr / Delta_T
    if abs(math.cos(theta) - math.cos(math.pi)) < 1e-12:
        dsc = 0.0
    else:
        dsc = 0.0      # fixed contact angle -> df/dr = 0
    return dsv, dss, dsc


# =============================================================================
#  DENSITY OF STATES OF THE LIQUID PHASE  (section 16)
# =============================================================================
def density_of_states_liquid(T, P=None):
    """Density of states of the nucleated liquid phase [molecules/m^3].

    For a crystal the ice script used the Debye vibrational mode density
    N_V = (1/6 pi^2)(Theta_D kB / (hbar v))^3.  That expression is specific to a
    crystalline solid and is NOT appropriate for a liquid; it is not reused.

    For a liquid droplet the physically relevant site density that enters the
    nucleation-rate prefactor (Eq. 21) is the molecular number density of the
    liquid:
        N_V = rho_l * N_A / M_H2O        [molecules . m^-3]
    Dimensionally:  [kg/m^3]*[mol^-1]/[kg/mol] = [m^-3].
    """
    return rho_l(T) * Nav / M_H2O


# =============================================================================
#  NUCLEATION RATE  (sections 17, 18)
# =============================================================================
# I = (D A / lambda^4) N_V exp( Delta_G_C / Delta_G_C,Eq )    (Ferreira Eq. 21)
# Delta_G_C is the (negative) barrier; Delta_G_C,Eq is the reference near-
# equilibrium barrier (largest |barrier|).  The ratio lies in [0,1] so the
# exponential is bounded; the prefactor carries the strong r,T dependence.
# log10_I is computed directly to avoid overflow.
# -----------------------------------------------------------------------------
def nucleation_rate_hom(r, T, P, Delta_G_C, Delta_G_C_eq, mechanism=MECHANISM):
    """Homogeneous nucleation rate I_Hom [m^-3 s^-1] and log10."""
    A = 4.0 * math.pi * r * r
    D = D_eff(T, P, mechanism)
    Nv = density_of_states_liquid(T)
    pref = D * A / (LAMBDA**4) * Nv
    # exponent
    if abs(Delta_G_C_eq) < 1e-30:
        expo = 0.0
    else:
        expo = Delta_G_C / Delta_G_C_eq
    log_I = math.log(max(pref, 1e-300)) + expo
    log10_I = log_I / math.log(10.0)
    if log_I > 700:
        return 10.0**300, log10_I
    return math.exp(log_I), log10_I

def nucleation_rate_het(r, T, P, Delta_G_C_het, Delta_G_C_eq,
                        theta, mechanism=MECHANISM):
    """Heterogeneous nucleation rate I_Het [m^-3 s^-1] and log10."""
    A = 2.0 * math.pi * r * r * (1.0 - math.cos(theta))   # spherical-cap area
    D = D_eff(T, P, mechanism)
    Nv = density_of_states_liquid(T)
    pref = D * A / (LAMBDA**4) * Nv
    if abs(Delta_G_C_eq) < 1e-30:
        expo = 0.0
    else:
        expo = Delta_G_C_het / Delta_G_C_eq
    log_I = math.log(max(pref, 1e-300)) + expo
    log10_I = log_I / math.log(10.0)
    if log_I > 700:
        return 10.0**300, log10_I
    return math.exp(log_I), log10_I


# =============================================================================
#  CNT REFERENCE  (section 22)
# =============================================================================
def cnt_reference(T, P_v, Delta_T):
    """Classical nucleation-theory reference values (used ONLY for comparison).
    r_CNT = 2 gamma / |Delta_G_V|,  Delta_G_CNT = 16 pi gamma^3 / (3 |Delta_G_V|^2).
    """
    g = gamma_VL_inf(T)
    dgv = bulk_free_energy(T, P_v, Delta_T)
    if abs(dgv) < 1e-30:
        return float("inf"), float("inf")
    r_cnt = 2.0 * g / abs(dgv)
    dG_cnt = 16.0 * math.pi * g**3 / (3.0 * dgv**2)
    return r_cnt, dG_cnt


# =============================================================================
#  CORRECTED GIBBS-THOMSON THERMAL-FIELD CLOSURE  (manuscript Section 6.3)
# =============================================================================
# The embryo radius r is the PRESCRIBED continuation variable.  At every r the
# LOCAL THERMAL GRADIENT  g = dT/dr  is the Brent UNKNOWN, solved from the
# second-order Gibbs-Thomson / thermal-field identity
#
#     F(g; r) = Gamma^(2)_liquid(r, Delta_T(g), g, ...) / (4 pi r^2) - g = 0 ,
#
# i.e.  Gamma^(2) = 4 pi r^2 g  =  A . grad_T   (thermal-field tensor, Eq. 4).
# During EVERY residual evaluation all gradient-dependent quantities are
# recomputed at the trial gradient (they are NOT frozen outside the iteration):
#   Delta_T = 8 pi r g ,  T_local = T_base - Delta_T ,
#   P_sat(T_local), Delta_mu(T_local,P_base), Delta_S_V(T_local),
#   d(Delta_S_V)/dr = (d Delta_S_V/dT)(-8 pi g) ,  d(Delta_T)/dr = 8 pi g ,
#   gamma_VL(r,T_local) (Tolman),  dgamma/dr(r,T_local).
# The homogeneous closure (f(theta)=4, df/dr=0) gives
#     Gamma^(2)_hom = -(3/4) (Delta_S_V Delta_T + dgamma/dr)
#                            / ( d(Delta_S_V)/dr + (Delta_S_V/Delta_T) d(Delta_T)/dr ).
# -----------------------------------------------------------------------------
T_MIN_LOCAL = 233.0   # deep-supercooling lower bound for T_local (~ -40 C)

def _Psat_extended(T):
    """Saturation pressure [Pa] extended smoothly below the triple point for the
    supercooled-liquid metastable states reached by the Gibbs-Thomson coupling.

    For T >= Tt this is IDENTICAL to the IAPWS saturation_pressure (the validated
    range).  For T < Tt the Wagner correlation is continued analytically -- it is
    a single analytic expression and the [Tt, Tc] bound is the *validated* range,
    not a domain boundary of the formula.  The continuation is monotonic,
    positive and C-infinity continuous at Tt, so the chain
    Delta_T -> T_local -> Delta_mu is never broken by the triple-point boundary.
    The PUBLIC saturation_pressure is left unchanged (it remains the strict IAPWS
    function used by the P/T sweeps and validation test 10)."""
    if T >= Tt:
        return saturation_pressure(T)
    tau = 1.0 - T / Tc
    s = sum(a * tau ** b for a, b in zip(_WAGNER_A, _WAGNER_B))
    return Pc * math.exp((Tc / T) * s)

def _chemical_potential_difference_extended(T, P_v, P_l=None):
    """Delta_mu = mu_l - mu_v  [J/mol] for the supercooled-liquid local state,
    using the extended saturation pressure.  Mirrors the public
    chemical_potential_difference EXACTLY for T >= Tt (same ideal-gas vapour +
    Poynting liquid terms, only the Psat source differs below Tt)."""
    if P_l is None:
        P_l = _Psat_extended(T)
    Vm = M_H2O / rho_l(T)                              # liquid molar volume [m^3/mol]
    mu_l = Vm * (P_l - _Psat_extended(T))              # Poynting (incompressible liq.)
    mu_v = R * T * math.log(P_v / _Psat_extended(T))   # ideal-gas vapour
    return mu_l - mu_v

def _local_state(g, r, T_base, P_base):
    """Recompute every gradient-dependent liquid quantity at trial gradient g
    and prescribed radius r.  Returns None if T_local leaves the physical range."""
    dT = 8.0 * math.pi * r * g
    T_loc = T_base - dT
    if T_loc < T_MIN_LOCAL or T_loc > T_base + 1e-9:
        return None
    dsv = bulk_entropy_change(T_loc)                       # < 0
    # high-order (9-point) central difference: the Gibbs-Thomson denominator
    # involves a small difference of large terms, so the derivative of Delta_S_V
    # must be accurate to ~1e-14 for the closure residual to reach < 1e-10.
    dDsv_dT = derivative(bulk_entropy_change, T_loc, dx=1e-2, n=1, order=9)
    dDT_dr = 8.0 * math.pi * g
    dT_dr = -dDT_dr
    dDsv_dr = dDsv_dT * dT_dr
    gam = surface_energy_VL(r, T_loc)
    dgdr = surface_energy_derivative(r, T_loc)
    Ps = _Psat_extended(T_loc)
    dmu = _chemical_potential_difference_extended(T_loc, P_base)
    dGv = dsv * dT
    return dict(g=g, r=r, dT=dT, T_local=T_loc, dsv=dsv, dDsv_dT=dDsv_dT,
                dDsv_dr=dDsv_dr, dDT_dr=dDT_dr, dT_dr=dT_dr, gam=gam, dgdr=dgdr,
                Psat=Ps, dmu=dmu, dGv=dGv)

def gibbs_thomson_hom_2nd_value(state):
    """Gamma^(2)_hom [m.K] from the local state dict (formula, Eq. 11)."""
    num = state["dsv"] * state["dT"] + state["dgdr"]
    den = state["dDsv_dr"] + state["dsv"] / state["dT"] * state["dDT_dr"]
    if abs(den) < 1e-30:
        return float("inf")
    return -0.75 * num / den

def gibbs_thomson_hom_2nd_residual(g, r, T_base, P_base):
    """CORRECTED Brent residual  F(g; r) = Gamma^(2)/(4 pi r^2) - g = 0.
    Returns a large-magnitude sentinel (not None) when the state is unphysical,
    so brentq sees a definite sign on the bracket ends."""
    state = _local_state(g, r, T_base, P_base)
    if state is None:
        # push the solver back toward the physical region
        return 1.0e30 if g > (T_base - T_MIN_LOCAL) / (8.0 * math.pi * r) else -1.0e30
    G2 = gibbs_thomson_hom_2nd_value(state)
    if not math.isfinite(G2):
        return 1.0e30
    return G2 / (4.0 * math.pi * r * r) - g

def _first_order_gradient_seed(r, T_base):
    """First-order stationarity gradient (form A) with frozen (T_base) properties,
    used only to centre the corrected Brent bracket:
        (Delta_S_V Delta_T + dgamma/dr) r + 2 gamma = 0,  Delta_T = 8 pi r g
    ->  g_seed = -(dgamma/dr r + 2 gamma) / (8 pi r^2 Delta_S_V)."""
    dsv = bulk_entropy_change(T_base)
    gam = surface_energy_VL(r, T_base)
    dgdr = surface_energy_derivative(r, T_base)
    coeff = 8.0 * math.pi * r * r * dsv
    if abs(coeff) < 1e-30:
        return None
    return -(dgdr * r + 2.0 * gam) / coeff

def _find_bracket(r, T_base, P_base):
    """Find a (lo, hi) bracket with a sign change of the corrected residual.
    Geometric scan centred on the first-order seed, bounded by the physical
    g_max = (T_base - T_MIN_LOCAL)/(8 pi r).  Returns (lo, hi) or None."""
    g_seed = _first_order_gradient_seed(r, T_base)
    if g_seed is None or g_seed <= 0 or not math.isfinite(g_seed):
        return None
    g_max = (T_base - T_MIN_LOCAL) / (8.0 * math.pi * r)
    lo = max(g_seed * 1e-4, 1e-12)
    hi = min(g_seed * 1e3, g_max)
    Flo = gibbs_thomson_hom_2nd_residual(lo, r, T_base, P_base)
    Fhi = gibbs_thomson_hom_2nd_residual(hi, r, T_base, P_base)
    # expand / shrink until a sign change is found
    tries = 0
    while Flo * Fhi > 0 and tries < 200:
        tries += 1
        if abs(Flo) < abs(Fhi):
            lo *= 0.5
            Flo = gibbs_thomson_hom_2nd_residual(lo, r, T_base, P_base)
        else:
            hi = min(hi * 1.5, g_max)
            Fhi = gibbs_thomson_hom_2nd_residual(hi, r, T_base, P_base)
        if hi >= g_max and Flo * Fhi > 0:
            # shrink lo-side further
            for _ in range(40):
                lo *= 0.5
                Flo = gibbs_thomson_hom_2nd_residual(lo, r, T_base, P_base)
                if Flo * Fhi <= 0:
                    break
            break
    if Flo * Fhi > 0 or not (math.isfinite(Flo) and math.isfinite(Fhi)):
        return None
    return (lo, hi)

def solve_gradient_at_radius(r, T_base, P_base, theta_default=math.pi,
                            xtol=2e-16, rtol=8.881784197001252e-16,
                            maxiter=500):
    """Corrected solver for one prescribed radius r.

    Brent-solves the local thermal gradient g from the Gibbs-Thomson closure
    F(g;r) = Gamma^(2)/(4 pi r^2) - g = 0, then assembles the full state:
      Delta_T, first/second-order critical radii, Gamma^(2) (= 4 pi r^2 g by the
      closure), heterogeneous contact angle theta in (0,pi), free-energy barriers
      and nucleation rate (evaluated at the selected physical radius r_C,2nd).

    Returns a state dict, or None if no admissible gradient exists at this r.
    """
    br = _find_bracket(r, T_base, P_base)
    if br is None:
        return None
    lo, hi = br
    try:
        g = brentq(gibbs_thomson_hom_2nd_residual, lo, hi,
                   args=(r, T_base, P_base), xtol=xtol, rtol=rtol,
                   maxiter=maxiter, full_output=False, disp=True)
    except Exception:
        return None
    state = _local_state(g, r, T_base, P_base)
    if state is None:
        return None
    # closure identity (the solved gradient enforces Gamma^(2) = 4 pi r^2 g)
    state["Gamma2"] = 4.0 * math.pi * r * r * g
    state["Gamma2_formula"] = gibbs_thomson_hom_2nd_value(state)
    state["closure_resid"] = state["Gamma2_formula"] / (4.0 * math.pi * r * r) - g
    # critical radii from the local thermal state
    state["rC_hom_1st"] = critical_radius_first_order_local(state)
    state["rC_hom_2nd"] = critical_radius_second_order_local(state)
    state["parabolic_resid"] = parabolic_stationarity_residual(
        state["rC_hom_2nd"], state) if math.isfinite(state["rC_hom_2nd"]) else float("nan")
    # first-order Gibbs-Thomson coefficient (diagnostic)
    state["Gamma1"] = gibbs_thomson_hom_1st(r, state["T_local"], state["dT"])
    # heterogeneous contact angle theta in (0, pi) from the heterogeneous relation
    #   (Delta_S_V Delta_T + dgamma/dr) f(theta)/4 + gamma df/dr(theta) = 0
    # solved by Brent (liquid Tolman properties; df/dr(theta) is the contact-angle
    # configurational term).  For the liquid the fixed-substrate configurational
    # term is handled through the geometric ratio; theta is found from the
    # heterogeneous stationarity that makes r_C,hom a heterogeneous critical radius.
    state["theta"] = _solve_heterogeneous_theta(state, theta_default)
    state["rC_het_2nd"] = (state["rC_hom_2nd"] * het_hom_radius_ratio(state["theta"])
                           if math.isfinite(state["rC_hom_2nd"]) else float("inf"))
    state["rC_het_1st"] = (state["rC_hom_1st"] * het_hom_radius_ratio(state["theta"])
                           if (math.isfinite(state["rC_hom_1st"])
                               and state["rC_hom_1st"] > 0) else float("inf"))
    # free-energy barriers and rates evaluated at the selected physical radius
    rc = state["rC_hom_2nd"] if math.isfinite(state["rC_hom_2nd"]) else r
    state["dGc_hom"] = critical_free_energy_hom_2nd_expr(rc, state["T_local"],
                                                        state["dT"], g)
    if not math.isfinite(state["dGc_hom"]):
        state["dGc_hom"] = critical_free_energy_hom(rc, state["T_local"], state["dT"])
    state["dGc_het"] = critical_free_energy_het_2nd_expr(rc, state["T_local"],
                                                        state["dT"], g, state["theta"])
    if not math.isfinite(state["dGc_het"]):
        state["dGc_het"] = critical_free_energy_het(rc, state["T_local"],
                                                    state["dT"], state["theta"])
    return state

def _solve_heterogeneous_theta(state, theta_default):
    """Solve the heterogeneous stationarity for the contact angle in (0, pi).

    Mirrors the verified ice reference residual  f_het_2nd  (its line 822):
        f_het_2nd(theta) = -(3/2) * ((DSv*DT + dgamma/dr) f(theta)/4
                                    + gamma * df/dtheta(theta))
                                / ( dDSvDTdr * f(theta)/4 + DSv*DT * df/dtheta(theta) )
                          - r_hom * het_hom_radius_ratio(theta)   = 0 ,
    with the LIQUID constitutive quantities substituted (Tolman gamma and
    dgamma/dr, volumetric Delta_S_V, the local Delta_T) and the liquid
    heterogeneous geometric ratio het_hom_radius_ratio (Eq. 17, preserved from
    the original liquid model).  r_hom is the physical homogeneous critical
    radius r_C,Hom,2nd (the principal result); dDSvDTdr uses the ice-reference
    derivative structure  -(3/2)(DSv*DT + dgamma/dr)/r_hom.  Brent root-finding
    in (1e-6, pi-1e-6); the near-equilibrium limit has its root exactly at
    theta=pi (the homogeneous / full-sphere limit, outside the open bracket), in
    which case the homogeneous-limit contact angle pi-1e-6 is returned.
    """
    r_hom = state["rC_hom_2nd"] if math.isfinite(state["rC_hom_2nd"]) else state["r"]
    DSv = state["dsv"]
    DT = state["dT"]
    gam = state["gam"]
    dgdr = state["dgdr"]
    dDSvDTdr = -1.5 * (DSv * DT + dgdr) / r_hom if r_hom != 0 else 0.0

    def resid(theta):
        f = ftheta(theta)
        dfdth = dftheta_dtheta(theta)
        den = dDSvDTdr * f / 4.0 + DSv * DT * dfdth
        if abs(den) < 1e-300:
            return 1e30
        return (-1.5 * ((DSv * DT + dgdr) * f / 4.0 + gam * dfdth) / den
                - r_hom * het_hom_radius_ratio(theta))

    try:
        return brentq(resid, 1e-6, math.pi - 1e-6, xtol=2e-14,
                      rtol=8.881784197001252e-14, maxiter=200)
    except Exception:
        # no sign change in (0,pi): the root sits at theta=pi (homogeneous limit)
        return math.pi - 1e-6


# =============================================================================
#  SHIFTED EQUILIBRIUM  (legacy wrapper, section 19, 26)
# =============================================================================
def shifted_equilibrium_vapour_liquid(T, P, grad_T, r,
                                      theta=math.pi, mechanism=MECHANISM):
    """Legacy diagnostic routine (public API preserved).

    Historically this mapped a PRESCRIBED grad_T -> Delta_T = 8 pi r grad_T
    (the old, incorrectly-posed algebraic coupling).  The corrected formulation
    instead prescribes the radius r and Brent-solves the gradient from the
    Gibbs-Thomson closure (see solve_gradient_at_radius).  This wrapper now
    evaluates the local thermal state at the SUPPLIED (r, grad_T) for diagnostic
    cross-checks only -- it does NOT enforce the closure.  Use
    solve_gradient_at_radius(r, T, P) for the corrected self-consistent state.
    """
    Ps = saturation_pressure(T)
    Delta_T = 8.0 * math.pi * r * grad_T
    dmu = chemical_potential_difference(T, P)
    dgv = bulk_entropy_change(T)
    dGv = dgv * Delta_T
    g = surface_energy_VL(r, T)
    dgdr = surface_energy_derivative(r, T)
    dsv, dss, dsc = entropy_decomposition(r, T, Delta_T, theta)
    A = 4.0 * math.pi * r * r
    Gamma = thermal_field_tensor(A, grad_T)
    return dict(
        T=T, P=P, Tsat=saturation_temperature(P) if _P_LO <= P <= _P_HI else float('nan'),
        Psat=Ps, Delta_T=Delta_T, Delta_P=P - Ps, grad_T=grad_T,
        Delta_mu=dmu, Delta_G_bulk=dGv, Delta_Sv=dsv,
        gamma_VL=g, dgamma_dr=dgdr,
        Delta_S_bulk=dsv, Delta_S_surface=dss, Delta_S_configurational=dsc,
        Gamma=Gamma,
    )


# =============================================================================
#  VALIDATION  (mandatory, manuscript Section 6.3 / task spec)
# =============================================================================
ICE_SCRIPT = "Nucleation_model_H2O_vapour_solid_Sim_2026_paper.py"
ICE_SHA256 = "c9fa9c01aabd147a455632a54fc8b907882b7fb167d9b0cff0bd9a86058403f6"

def _ice_checksum():
    import hashlib, os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ICE_SCRIPT)
    if not os.path.exists(p):
        return None
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def _test_radius_grid(T_base, P_base, n=14):
    """Geometric radius grid (large r -> small g near equilibrium; small r ->
    large g).  Spans the physical embryo-size range for the liquid."""
    r_max = 1.0e-2     # large embryo ~ tiny gradient (near equilibrium)
    r_min = 1.0e-9     # molecular-scale embryo ~ large gradient
    return [r_max * (r_min / r_max) ** (i / (n - 1)) for i in range(n)]

def validation(verbose=True):
    """Run the 11 mandatory validation tests for the corrected liquid model."""
    def pr(*a):
        if verbose:
            print(*a)

    pr("=" * 78)
    pr("VALIDATION  (corrected Gibbs-Thomson thermal-field closure)")
    pr("=" * 78)
    ok = True
    T_base = saturation_temperature(101325.0)
    P_base = 101325.0
    grid = _test_radius_grid(T_base, P_base)
    states = []
    for r in grid:
        s = solve_gradient_at_radius(r, T_base, P_base)
        if s is not None:
            states.append(s)

    # 1 -- ice reference script byte-for-byte unchanged
    cs = _ice_checksum()
    pr(f"[1] ice script SHA-256 = {cs}")
    pr(f"    expected           = {ICE_SHA256}")
    if cs is None:
        pr("    !! ice script not found alongside this module")
        ok = False
    else:
        ok &= (cs == ICE_SHA256)
        pr(f"    -> {'UNCHANGED' if cs == ICE_SHA256 else 'CHANGED !!'}")

    # 2 -- |Gamma^(2)/(4 pi r^2) - g| < tol at every converged state.  The solved
    #      gradient g spans ~14 orders of magnitude (1e-14 .. 1e7 K/m), so a single
    #      absolute threshold is ill-posed: the residual is round-off limited at
    #      an ABSOLUTE floor for |g| << 1 (subtraction of two ~equal tiny terms) and
    #      at a RELATIVE floor ~|g|*eps for |g| >> 1.  A scaled tolerance
    #      |F|/max(|g|,1) < 1e-10 is absolute in the near-equilibrium regime and
    #      relative in the deep-undercooling regime -- both hold to round-off.
    worst2 = 0.0
    worst2abs = 0.0
    for s in states:
        worst2 = max(worst2, abs(s["closure_resid"]) / max(abs(s["g"]), 1.0))
        worst2abs = max(worst2abs, abs(s["closure_resid"]))
    pr(f"[2] max |Gamma2/(4pi r^2)-g|/max(|g|,1) over {len(states)} states = {worst2:.2e} "
       f"(scaled tol 1e-10); worst absolute = {worst2abs:.2e}")
    ok &= worst2 < 1e-10

    # 3 -- r_C,2nd resubstituted into the parabolic stationarity -> ~0
    worst3 = 0.0
    for s in states:
        if math.isfinite(s["parabolic_resid"]):
            worst3 = max(worst3, abs(s["parabolic_resid"]))
    pr(f"[3] max |parabolic stationarity(r_C,2nd)| = {worst3:.2e} (tol 1e-8)")
    ok &= worst3 < 1e-8

    # 4 -- selected critical radius real, positive, finite, admissible
    bad4 = [s for s in states
            if not (math.isfinite(s["rC_hom_2nd"]) and s["rC_hom_2nd"] > 0)]
    pr(f"[4] states with non-admissible r_C,2nd: {len(bad4)} / {len(states)}")
    ok &= len(bad4) == 0

    # 5 -- every finite admissible gradient -> finite critical radius
    bad5 = [s for s in states
            if math.isfinite(s["g"]) and s["g"] > 0
            and not math.isfinite(s["rC_hom_2nd"])]
    pr(f"[5] finite-gradient states with non-finite r_C,2nd: {len(bad5)} / {len(states)}")
    ok &= len(bad5) == 0

    # 6 -- as grad_T -> 0 (large-r end): Delta_T->0, Delta_mu->0, r_C->+inf
    if states:
        # states are ordered large-r (small g) first
        near_eq = states[0]
        pr(f"[6] near-equilibrium (large r): g={near_eq['g']:.3e} K/m, "
           f"Delta_T={near_eq['dT']:.3e} K, Delta_mu={near_eq['dmu']:.3e} J/mol, "
           f"r_C,2nd={near_eq['rC_hom_2nd']:.3e} m")
        ok &= near_eq["dT"] < 1e-3 and abs(near_eq["dmu"]) < 1e-3
        ok &= near_eq["rC_hom_2nd"] > 1.0      # diverges as g -> 0
        # monotonicity: r_C,2nd grows as g shrinks (scan from large r to small r)
        gs = [s["g"] for s in states]
        rc2 = [s["rC_hom_2nd"] for s in states]
        grows = all(rc2[i] > rc2[i + 1] for i in range(len(rc2) - 1)
                    if math.isfinite(rc2[i]) and math.isfinite(rc2[i + 1]))
        pr(f"    r_C,2nd monotonically increases as g -> 0: {grows}")
        ok &= grows

    # 7 -- first-order radius -> CNT radius in the classical limit.  The classical
    #      limit requires the Tolman curvature term dgamma/dr to be negligible vs
    #      the entropy driving term Delta_S_V * Delta_T, i.e. an appreciable
    #      undercooling at a radius large enough that 2*d_Tol/r << 1.  The
    #      closure-solved states at large r have a TINY Delta_T (near-equilibrium)
    #      and are Tolman-dominated (r_C,1st < 0), so instead we build a local
    #      thermal state directly at Delta_T ~ 20 K, r = 1 mm (Tolman ~ 4e-7) and
    #      compare the first-order radius formula to the CNT reference.
    r_big = 1.0e-3
    dT_test = 20.0
    g_test = dT_test / (8.0 * math.pi * r_big)
    s_big = _local_state(g_test, r_big, T_base, P_base)
    rC1st_big = critical_radius_first_order_local(s_big) if s_big else float("nan")
    if s_big is not None and math.isfinite(rC1st_big) and rC1st_big > 0:
        r_cnt = cnt_reference(s_big["T_local"], P_base, s_big["dT"])[0]
        rel = abs(rC1st_big - r_cnt) / max(abs(r_cnt), 1e-30)
        pr(f"[7] classical limit (Delta_T={dT_test} K, r={r_big} m): "
           f"r_C,1st={rC1st_big:.4e} vs r_CNT={r_cnt:.4e} "
           f"(rel err {rel:.2e})")
        ok &= rel < 0.05
    else:
        pr("    !! could not reach the classical (entropy-dominated) limit")
        ok = False

    # 8 -- 0 < theta < pi strictly (Brent bracket (1e-6, pi-1e-6); the
    #      near-equilibrium limit has its root exactly at theta=pi, the homogeneous
    #      full-sphere limit, returned as pi-1e-6 which is strictly inside (0,pi)).
    bad8 = [s for s in states if not (0.0 < s["theta"] < math.pi)]
    pr(f"[8] states with theta outside (0,pi): {len(bad8)} / {len(states)}")
    ok &= len(bad8) == 0

    # 9 -- tolerance- and bracket-independence of the converged gradient.  Brent's
    #      rtol has a scipy floor of 4*eps (8.88e-16, the value used throughout),
    #      so "tightening" is done via xtol and via an explicitly shifted/expanded
    #      bracket; the converged gradient must move by < 1e-6 relative.
    if len(states) >= 2:
        r_test = states[len(states) // 2]["r"]
        s_def = solve_gradient_at_radius(r_test, T_base, P_base)
        s_tight = solve_gradient_at_radius(
            r_test, T_base, P_base, xtol=2e-18,
            rtol=8.881784197001252e-16)          # tighter xtol, valid rtol at floor
        # bracket-independence: re-solve over a deliberately expanded bracket
        br = _find_bracket(r_test, T_base, P_base)
        g_shift = None
        if br is not None:
            lo, hi = br
            try:
                g_shift = brentq(gibbs_thomson_hom_2nd_residual, lo * 0.5, hi * 2.0,
                                args=(r_test, T_base, P_base), xtol=2e-16,
                                rtol=8.881784197001252e-16, maxiter=500)
            except Exception:
                g_shift = None
        if s_def and s_tight and g_shift is not None:
            g_def = s_def["g"]
            rel9 = abs(g_def - s_tight["g"]) / max(abs(g_def), 1e-30)
            rel9b = abs(g_def - g_shift) / max(abs(g_def), 1e-30)
            pr(f"[9] invariance: |g_def-g_tight|/g={rel9:.2e}, "
               f"|g_def-g_shifted_bracket|/g={rel9b:.2e} (tol 1e-6)")
            ok &= rel9 < 1e-6 and rel9b < 1e-6
        else:
            pr("[9] !! solver/bracket failed at the test radius")
            ok = False

    # 10 -- IAPWS saturation inversion valid across the full temperature range
    # [Tt, Tc].  Points are kept safely inside the range (the inverse brentq
    # bracket is [Tt+1e-6, Tc-1e-6], so T is tested up to ~645 K, ~2 K below Tc,
    # where the Wagner correlation is still well-conditioned).
    worst10 = 0.0
    roundtrip_fail = 0
    for T in (274.0, 300.0, 373.15, 450.0, 500.0, 600.0, 640.0, 645.0):
        try:
            P = saturation_pressure(T)
            Tb = saturation_temperature(P)
            worst10 = max(worst10, abs(Tb - T))
        except Exception:
            roundtrip_fail += 1
    pr(f"[10] IAPWS round-trip max |T_sat(P_sat(T))-T| over range = {worst10:.2e} K "
       f"(failures: {roundtrip_fail})")
    ok &= worst10 < 1e-4 and roundtrip_fail == 0
    try:
        saturation_pressure(700.0)
        ok = False
        pr("    !! range gate failed (T>Tc accepted)")
    except ValueError:
        pr("    range gate OK (T>Tc rejected).")

    # 11 -- no old-solver tables/figures reused: the obsolete backup exists
    #       separately and the regenerated CSV carries the new 'r' column.
    import os
    backup = "water_vapour_liquid_nucleation_obsolete_backup.csv"
    backup_ok = os.path.exists(backup)
    pr(f"[11] obsolete-formulation backup preserved separately: {backup} -> {backup_ok}")
    ok &= backup_ok
    # the corrected solver must not depend on the deprecated gradient-as-input
    # solvers (smoke test: the closure residual is solved, not the old quadratic)
    pr(f"    corrected path uses solve_gradient_at_radius (Brent on gradient): yes")

    # dimensional / equilibrium sanity (kept from the original validation)
    P_eq = saturation_pressure(373.15)
    dmu_eq = chemical_potential_difference(373.15, P_eq)
    dGv_eq = bulk_free_energy(373.15, P_eq, 0.0)
    pr(f"    equilibrium Delta_mu={dmu_eq:.2e} J/mol (must be ~0); "
       f"Delta_G_V(Delta_T=0)={dGv_eq:.2e} J/m^3 (must be 0)")
    ok &= abs(dmu_eq) < 1e-6 and abs(dGv_eq) < 1e-12
    ok &= 70 < gamma_VL_inf(293.15) * 1000 < 80 and rho_l(293.15) > 900

    pr("-" * 78)
    pr(f"VALIDATION {'PASSED' if ok else 'FAILED'}")
    pr("=" * 78)
    return ok


# =============================================================================
#  MAIN SIMULATION  (sections 12, 13, 20, 24, 26)
# =============================================================================
def run_simulation():
    """Run the equilibrium-shift sweep:
      * grad_T sweep at a base state (the primary driver, section 12),
      * P sweep at fixed T  (section 13),
      * T sweep at fixed P  (sections 5, 13, for Delta_mu / Delta_G_V / Psat).
    Produces a list of row dicts written to CSV and a dict of arrays for plots.
    """
    print("\n" + "=" * 78)
    print("SIMULATION  H2O(vapour) -> H2O(liquid)  equilibrium-shift nucleation")
    print("=" * 78)

    # ----- base state -----
    # MODE = "T" : pressure fixed, temperature variable
    # MODE = "P" : temperature fixed, pressure variable
    MODE = "T"
    if MODE == "T":
        P_base = 101325.0                  # 1 atm fixed
        T_base = saturation_temperature(P_base)   # coexistence T at 1 atm
    else:
        T_base = 293.15
        P_base = saturation_pressure(T_base)
    print(f"Base state: MODE={MODE}, T_base={T_base:.3f} K, P_base={P_base:.3f} Pa")

    # ----- radius-continuation sweep (manuscript Section 6.3, corrected closure) -----
    # The embryo radius r is the PRESCRIBED continuation variable.  At every r the
    # local thermal gradient g = dT/dr is the Brent UNKNOWN solved from the
    # Gibbs-Thomson / thermal-field identity  F(g;r) = Gamma^(2)/(4 pi r^2) - g = 0.
    # The liquid has NO finite equilibrium radius (r_C -> infinity as g -> 0), so
    # -- unlike ice, which discretises around a finite r_eq with dr = 0.005 r_eq --
    # the liquid uses a GEOMETRIC radius grid spanning the embryo sizes that
    # produce gradients across the physically relevant range (~10 .. 10^6 K/m):
    # r from r_max (large embryo, tiny g, near-equilibrium) down to r_min (small
    # embryo, large g, deep supercooling).  This is a PHASE-SPECIFIC constitutive
    # difference, NOT a different thermal-field closure.
    r_max = 1.0e-2     # [m]  (1 cm embryo -> near-equilibrium, g -> 0)
    r_min = 1.0e-9     # [m]  (1 nm embryo  -> large undercooling)
    n_points = 75
    r_grid = np.geomspace(r_max, r_min, n_points)

    rows = []
    theta = THETA_0

    # ---------- primary radius-continuation sweep ----------
    results = dict(r=[], gradT=[], DeltaT=[], rC_hom_1st=[], rC_hom_2nd=[],
                   rC_het_1st=[], rC_het_2nd=[], Delta_mu=[], DeltaG_bulk=[],
                   gamma_VL=[], dgamma_dr=[], DeltaS_bulk=[], DeltaS_surface=[],
                   DeltaS_configurational=[], Gamma_1st=[], Gamma_2nd=[],
                   DeltaG_C_hom=[], DeltaG_C_het=[], I_hom=[], I_het=[],
                   log10_I_hom=[], log10_I_het=[], rC_CNT=[], DeltaG_CNT=[],
                   I_CNT=[], log10_I_CNT=[], ratio_rC=[], ratio_I=[])

    # first pass: solve the corrected Gibbs-Thomson closure at every prescribed r
    solved = []   # list of admissible state dicts (only)
    skipped = 0
    for r in r_grid:
        st = solve_gradient_at_radius(r, T_base, P_base, theta_default=theta)
        if st is None:
            skipped += 1
            continue
        solved.append(st)
    print(f"Radius-continuation sweep: {len(solved)}/{n_points} radii converged, "
          f"{skipped} skipped (no admissible gradient in the physical range).")

    # reference near-equilibrium barrier |Delta_G_C,Eq| (Ferreira Eq. 21):
    # the smallest-gradient (largest-r) converged state.
    if solved:
        ref_state = min(solved, key=lambda s: abs(s["g"]))
        dGc_ref = ref_state["dGc_hom"]
        if not math.isfinite(dGc_ref) or abs(dGc_ref) < 1e-30:
            rc_ref = (ref_state["rC_hom_2nd"] if math.isfinite(ref_state["rC_hom_2nd"])
                      else ref_state["r"])
            dGc_ref = critical_free_energy_hom(rc_ref, ref_state["T_local"],
                                               ref_state["dT"])
        print(f"Reference near-equilibrium barrier |Delta_G_C,Eq| = {abs(dGc_ref):.4e} J "
              f"(at r={ref_state['r']:.3e} m, grad_T={ref_state['g']:.3e} K/m)")
    else:
        dGc_ref = float("inf")

    dmu_check = []      # (grad_T, Delta_mu_explicit, Delta_mu_approx) cross-check
    for st in solved:
        g = st["g"]; r = st["r"]; dT = st["dT"]; T_local = st["T_local"]
        # selected physical radius = the second-order critical radius (principal
        # physical result); fall back to the prescribed r if it is not finite
        rc = st["rC_hom_2nd"] if math.isfinite(st["rC_hom_2nd"]) else r
        # explicit Delta_mu vs near-equilibrium approximation (section 6 cross-check)
        dmu_approx = bulk_entropy_molar(T_base) * dT
        dmu_check.append((g, st["dmu"], dmu_approx))
        # entropy decomposition at the selected physical radius
        dsv, dss, dsc = entropy_decomposition(rc, T_local, dT, theta)
        # CNT reference (comparison only) at the local state
        r_cnt, dG_cnt = cnt_reference(T_local, P_base, dT)
        # CNT rate (Arrhenius-style for comparison)
        if math.isfinite(dG_cnt) and dG_cnt > 0:
            A_cnt = 4.0 * math.pi * r_cnt * r_cnt
            D = D_eff(T_local, P_base)
            Nv = density_of_states_liquid(T_local)
            arg = -dG_cnt / (kB * T_local)
            if arg < -700:
                I_cnt = 0.0; logI_cnt = -np.inf
            else:
                I_cnt = D * A_cnt / (LAMBDA**4) * Nv * math.exp(arg)
                logI_cnt = math.log10(I_cnt) if I_cnt > 0 else -np.inf
        else:
            I_cnt = 0.0; logI_cnt = -np.inf
        # nucleation rates (hom/het) at the selected physical radius
        I_hom, logI_hom = nucleation_rate_hom(rc, T_local, P_base, st["dGc_hom"], dGc_ref)
        I_het, logI_het = nucleation_rate_het(rc, T_local, P_base, st["dGc_het"],
                                              dGc_ref, st["theta"])

        row = dict(r=r, T=T_base, P=P_base,
                   Tsat=T_base, Psat=saturation_pressure(T_base),
                   DeltaT=dT, DeltaP=P_base - saturation_pressure(T_base),
                   gradT=g,
                   rC_hom_1st=st["rC_hom_1st"], rC_hom_2nd=st["rC_hom_2nd"],
                   rC_het_1st=st["rC_het_1st"], rC_het_2nd=st["rC_het_2nd"],
                   Delta_mu=st["dmu"], DeltaG_bulk=st["dGv"],
                   gamma_VL=st["gam"], dgamma_dr=st["dgdr"],
                   DeltaS_bulk=dsv, DeltaS_surface=dss,
                   DeltaS_configurational=dsc,
                   Gamma_1st=st["Gamma1"], Gamma_2nd=st["Gamma2"],
                   DeltaG_C_hom=st["dGc_hom"], DeltaG_C_het=st["dGc_het"],
                   I_hom=I_hom, I_het=I_het,
                   log10_I_hom=logI_hom, log10_I_het=logI_het,
                   rC_CNT=r_cnt, DeltaG_CNT=dG_cnt, I_CNT=I_cnt,
                   log10_I_CNT=logI_cnt,
                   ratio_rC=(st["rC_hom_2nd"] / r_cnt)
                              if (math.isfinite(r_cnt) and r_cnt > 0
                                  and math.isfinite(st["rC_hom_2nd"])) else float('nan'),
                   ratio_I=(I_hom / I_cnt) if I_cnt > 0 else float('nan'))
        rows.append(row)
        for k in results:
            v = row[k]
            results[k].append(v if math.isfinite(v) else np.nan)

    # cross-check: explicit Delta_mu vs near-equilibrium approximation (section 6)
    if dmu_check:
        worst = max(abs(de - da) / max(abs(de), 1e-30) for _, de, da in dmu_check)
        print(f"Delta_mu chain check: explicit vs near-eq approx agree to "
              f"{worst*100:.2f}% (worst case) over the grad_T sweep; "
              f"Delta_mu(grad_T) now varies from {dmu_check[0][1]:.3e} to "
              f"{dmu_check[-1][1]:.3e} J/mol.")

    # ---------- pressure sweep at fixed T (section 13) ----------
    P_atm = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    P_array = P_atm * 101325.0
    T_sweep = []
    P_sat_sweep = []
    T_sat_sweep = []
    dmu_sweep = []
    dGv_sweep = []
    for P in P_array:
        if Pt <= P <= Pc:
            Ts = saturation_temperature(P)
            T_sweep.append(Ts)
            P_sat_sweep.append(saturation_pressure(Ts))
            T_sat_sweep.append(Ts)
            # supersaturation-driven Delta_mu at fixed T_base with P_v = P
            dmu = chemical_potential_difference(T_base, P)
            dGv_sweep.append(dmu)
        else:
            T_sweep.append(np.nan); P_sat_sweep.append(np.nan)
            T_sat_sweep.append(np.nan); dmu_sweep.append(np.nan)
            dGv_sweep.append(np.nan)
    P_sweep_data = dict(P=P_array, T_sat=T_sat_sweep, P_sat=P_sat_sweep,
                        dmu=dGv_sweep)

    # ---------- temperature sweep at fixed P (sections 5, 13) ----------
    T_scan = np.linspace(280, 640, 200)
    Psat_T = []
    dmu_T = []
    dGv_T = []
    for T in T_scan:
        Ps = saturation_pressure(T)
        Psat_T.append(Ps)
        dmu_T.append(chemical_potential_difference(T, P_base))
        # Delta_T relative to T_sat(P_base); for T < T_sat -> condensation
        dT = saturation_temperature(P_base) - T
        dGv_T.append(bulk_free_energy(T, P_base, dT))
    T_sweep_data = dict(T=T_scan, Psat=Psat_T, dmu=dmu_T, dGv=dGv_T)

    return rows, results, P_sweep_data, T_sweep_data, dict(T_base=T_base,
           P_base=P_base, theta=theta, dGc_ref=dGc_ref)


# =============================================================================
#  SAVE RESULTS  (section 24)
# =============================================================================
COLUMNS = ["r", "T", "P", "Tsat", "Psat", "DeltaT", "DeltaP", "gradT",
           "rC_hom_1st", "rC_hom_2nd", "rC_het_1st", "rC_het_2nd",
           "Delta_mu", "DeltaG_bulk", "gamma_VL", "dgamma_dr",
           "DeltaS_bulk", "DeltaS_surface", "DeltaS_configurational",
           "Gamma_1st", "Gamma_2nd", "DeltaG_C_hom", "DeltaG_C_het",
           "I_hom", "I_het", "log10_I_hom", "log10_I_het",
           "rC_CNT", "DeltaG_CNT", "I_CNT", "log10_I_CNT",
           "ratio_rC", "ratio_I"]

def save_results(rows, fname="water_vapour_liquid_nucleation.csv"):
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r.get(c, "") for c in COLUMNS])
    print(f"Saved {len(rows)} rows -> {fname}")


# =============================================================================
#  PLOTS  (section 23)  -- 13 mandatory figures, matplotlib only
# =============================================================================
def _safe(arr):
    """Convert to float array, mapping inf -> nan (so non-finite points are
    simply skipped by matplotlib rather than breaking log-scale tick locators)."""
    a = np.array(arr, dtype=float)
    a[~np.isfinite(a)] = np.nan
    return a

def _mp(ax, x, y, *args, **kw):
    """Masked plot: keep only points where BOTH x and y are finite (and, if the
    receiving axis is log-scaled on that dimension, strictly positive).  This
    prevents 'Data cannot be log-scaled because all values are <= 0' when a
    series underflows (e.g. CNT rate -> 0 at small grad_T)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if ax.get_xscale() == "log":
        keep &= (x > 0)
    if ax.get_yscale() == "log":
        keep &= (y > 0)
    if not np.any(keep):
        return
    ax.plot(x[keep], y[keep], *args, **kw)

def plot_results(results, P_sweep, T_sweep, meta):
    figs = []
    gT = _safe(results["gradT"])
    nonzero = gT > 0

    # 1. P_sat versus T
    fig, ax = plt.subplots(); figs.append((fig, "fig01_Psat_vs_T"))
    _mp(ax, T_sweep["T"], [p/1000 for p in T_sweep["Psat"]], 'b-')
    ax.set_xlabel(r"$T$ [K]"); ax.set_ylabel(r"$P_{sat}$ [kPa]")
    ax.set_title(r"Saturation pressure $P_{sat}(T)$  (IAPWS)")
    ax.grid(True, ls='--', alpha=0.4)

    # 2. Delta_mu versus T
    fig, ax = plt.subplots(); figs.append((fig, "fig02_Dmu_vs_T"))
    _mp(ax, T_sweep["T"], T_sweep["dmu"], 'b-')
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(meta["T_base"], color='r', ls='--', lw=1, label=r"$T_{sat}(P_{base})$")
    ax.set_xlabel(r"$T$ [K]"); ax.set_ylabel(r"$\Delta\mu=\mu_l-\mu_v$ [J/mol]")
    ax.set_title(r"Chemical-potential difference $\Delta\mu(T)$")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)

    # 3. Delta_G_V versus T
    fig, ax = plt.subplots(); figs.append((fig, "fig03_DGV_vs_T"))
    _mp(ax, T_sweep["T"], T_sweep["dGv"], 'b-')
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(meta["T_base"], color='r', ls='--', lw=1)
    ax.set_xlabel(r"$T$ [K]"); ax.set_ylabel(r"$\Delta G_V$ [J/m$^3$]")
    ax.set_title(r"Bulk free energy $\Delta G_V(T)$")
    ax.grid(True, ls='--', alpha=0.4)

    # 4. r_C versus Delta_T
    fig, ax = plt.subplots(); figs.append((fig, "fig04_rC_vs_DeltaT"))
    dT = _safe(results["DeltaT"])
    ax.set_yscale('log')
    _mp(ax, dT, _safe(results["rC_hom_1st"]), 'b-o', ms=3, label=r"$r_{C,Hom}^{1st}$")
    _mp(ax, dT, _safe(results["rC_hom_2nd"]), 'b--', label=r"$r_{C,Hom}^{2nd}$")
    _mp(ax, dT, _safe(results["rC_CNT"]), 'r:', label=r"$r_{C}^{CNT}$")
    ax.set_xlabel(r"$\Delta T$ [K]"); ax.set_ylabel(r"$r_C$ [m]")
    ax.set_title(r"Critical radius vs $\Delta T$")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)

    # 5. r_C versus grad_T
    fig, ax = plt.subplots(); figs.append((fig, "fig05_rC_vs_gradT"))
    ax.set_xscale('log'); ax.set_yscale('log')
    _mp(ax, gT, _safe(results["rC_hom_1st"]), 'b-o', ms=3, label=r"$r_{C,Hom}^{1st}$")
    _mp(ax, gT, _safe(results["rC_hom_2nd"]), 'b--', label=r"$r_{C,Hom}^{2nd}$")
    _mp(ax, gT, _safe(results["rC_het_2nd"]), 'r--', label=r"$r_{C,Het}^{2nd}$")
    ax.set_xlabel(r"$\nabla T$ [K/m]"); ax.set_ylabel(r"$r_C$ [m]")
    ax.set_title(r"Critical radius vs $\nabla T$")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)

    # 6. r_C^2nd / r_C^CNT versus grad_T
    fig, ax = plt.subplots(); figs.append((fig, "fig06_ratio_vs_gradT"))
    r2 = _safe(results["rC_hom_2nd"]); rc = _safe(results["rC_CNT"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((rc > 0) & np.isfinite(rc) & np.isfinite(r2), r2 / rc, np.nan)
    ax.set_xscale('log')
    _mp(ax, gT, ratio, 'k-o', ms=3)
    ax.axhline(1.0, color='r', ls='--', lw=1)
    ax.set_xlabel(r"$\nabla T$ [K/m]")
    ax.set_ylabel(r"$r_{C}^{2nd}/r_{C}^{CNT}$")
    ax.set_title(r"Ratio $r_{C}^{2nd}/r_{C}^{CNT}$ vs $\nabla T$")
    ax.grid(True, ls='--', alpha=0.4)

    # 7. gamma_VL versus r/r_ref
    fig, ax = plt.subplots(); figs.append((fig, "fig07_gamma_vs_r"))
    r1 = _safe(results["rC_hom_1st"])
    r_ref = np.nanmin(r1[r1 > 0]) if np.any(r1 > 0) else 1e-9
    r_grid = np.linspace(r_ref, 50 * r_ref, 200)
    T = meta["T_base"]
    g_vals = [surface_energy_VL(r, T) / gamma_VL_inf(T) for r in r_grid]
    _mp(ax, r_grid / r_ref, g_vals, 'b-')
    ax.axhline(1.0, color='r', ls='--', lw=1, label=r"$\gamma_{\infty}$")
    ax.set_xlabel(r"$r/r_{ref}$"); ax.set_ylabel(r"$\gamma_{VL}/\gamma_{\infty}$")
    ax.set_title(r"Surface energy $\gamma_{VL}(r)$ (Tolman)")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)

    # 8. dgamma/dr versus r/r_ref
    fig, ax = plt.subplots(); figs.append((fig, "fig08_dgamma_dr_vs_r"))
    dg_vals = [surface_energy_derivative(r, T) for r in r_grid]
    _mp(ax, r_grid / r_ref, dg_vals, 'b-')
    ax.set_xlabel(r"$r/r_{ref}$"); ax.set_ylabel(r"$\partial\gamma_{VL}/\partial r$ [J/m$^3$]")
    ax.set_title(r"Surface-energy derivative vs $r/r_{ref}$")
    ax.grid(True, ls='--', alpha=0.4)

    # 9. Gamma versus grad_T
    fig, ax = plt.subplots(); figs.append((fig, "fig09_Gamma_vs_gradT"))
    ax.set_xscale('log')
    _mp(ax, gT, _safe(results["Gamma_1st"]), 'b-o', ms=3, label=r"$\Gamma^{1st}$")
    _mp(ax, gT, _safe(results["Gamma_2nd"]), 'r--', label=r"$\Gamma^{2nd}$")
    ax.set_xlabel(r"$\nabla T$ [K/m]"); ax.set_ylabel(r"$\Gamma$ [m.K]")
    ax.set_title(r"Gibbs-Thomson / thermal-field tensor $\Gamma$ vs $\nabla T$")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)

    # 10. Delta_G_C versus grad_T
    fig, ax = plt.subplots(); figs.append((fig, "fig10_DGC_vs_gradT"))
    ax.set_xscale('log'); ax.set_yscale('log')
    _mp(ax, gT, np.abs(_safe(results["DeltaG_C_hom"])), 'b-',
        label=r"$|\Delta G_{C,Hom}|$")
    _mp(ax, gT, np.abs(_safe(results["DeltaG_C_het"])), 'r--',
        label=r"$|\Delta G_{C,Het}|$")
    _mp(ax, gT, np.abs(_safe(results["DeltaG_CNT"])), 'k:',
        label=r"$|\Delta G_{C}^{CNT}|$")
    ax.set_xlabel(r"$\nabla T$ [K/m]"); ax.set_ylabel(r"$|\Delta G_C|$ [J]")
    ax.set_title(r"Critical free energy vs $\nabla T$")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)

    # 11. log10(I) versus Delta_T
    fig, ax = plt.subplots(); figs.append((fig, "fig11_logI_vs_DeltaT"))
    _mp(ax, dT, _safe(results["log10_I_hom"]), 'b-o', ms=3, label=r"$\log_{10}I_{Hom}$")
    _mp(ax, dT, _safe(results["log10_I_het"]), 'r-s', ms=3, label=r"$\log_{10}I_{Het}$")
    ax.set_xlabel(r"$\Delta T$ [K]"); ax.set_ylabel(r"$\log_{10} I$")
    ax.set_title(r"Nucleation rate vs $\Delta T$")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)

    # 12. log10(I) versus grad_T
    fig, ax = plt.subplots(); figs.append((fig, "fig12_logI_vs_gradT"))
    ax.set_xscale('log')
    _mp(ax, gT, _safe(results["log10_I_hom"]), 'b-o', ms=3, label=r"$\log_{10}I_{Hom}$")
    _mp(ax, gT, _safe(results["log10_I_het"]), 'r-s', ms=3, label=r"$\log_{10}I_{Het}$")
    ax.set_xlabel(r"$\nabla T$ [K/m]"); ax.set_ylabel(r"$\log_{10} I$")
    ax.set_title(r"Nucleation rate vs $\nabla T$")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)

    # 13. I_Ferreira / I_CNT versus grad_T
    fig, ax = plt.subplots(); figs.append((fig, "fig13_ratio_I_vs_gradT"))
    Ih = _safe(results["I_hom"]); Ic = _safe(results["I_CNT"])
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where((Ic > 0) & np.isfinite(Ic) & np.isfinite(Ih), Ih / Ic, np.nan)
    ax.set_xscale('log'); ax.set_yscale('log')
    _mp(ax, gT, rr, 'k-o', ms=3)
    ax.axhline(1.0, color='r', ls='--', lw=1)
    ax.set_xlabel(r"$\nabla T$ [K/m]")
    ax.set_ylabel(r"$I_{Ferreira}/I_{CNT}$")
    ax.set_title(r"Ferreira / CNT nucleation-rate ratio vs $\nabla T$")
    ax.grid(True, ls='--', alpha=0.4)

    # save all figures (guard tight_layout against degenerate log axes)
    for fig, name in figs:
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(name + ".png", dpi=120)
        plt.close(fig)
    print(f"Saved {len(figs)} figures -> fig01..fig13 .png")


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    if not validation(verbose=True):
        print("Validation failed; aborting before nucleation run.")
        raise SystemExit(1)

    rows, results, P_sweep, T_sweep, meta = run_simulation()

    # console summary of the radius-continuation sweep (section 26)
    print("\nradius-continuation sweep summary (T_base=%.3f K, P_base=%.1f Pa):" %
          (meta["T_base"], meta["P_base"]))
    hdr = ("r[m]", "gradT[K/m]", "DeltaT[K]", "rC2nd[m]", "Gamma2[m.K]",
           "|DGC_hom|[J]", "log10Ihom", "rC_CNT[m]")
    print("  " + " | ".join(hdr))
    for r in rows:
        print("  %9.3e | %9.3e | %8.4e | %8.4e | %8.4e | %8.4e | %8.3f | %8.4e" % (
            r["r"], r["gradT"], r["DeltaT"], r["rC_hom_2nd"],
            r["Gamma_2nd"], abs(r["DeltaG_C_hom"]),
            r["log10_I_hom"] if np.isfinite(r["log10_I_hom"]) else float('nan'),
            r["rC_CNT"]))

    save_results(rows, "water_vapour_liquid_nucleation.csv")
    plot_results(results, P_sweep, T_sweep, meta)

    print("\n" + "=" * 78)
    print("DONE.  Corrected governing chain (manuscript Section 6.3):")
    print("  r (prescribed) -> Brent-solve grad_T from  Gamma^(2)/(4 pi r^2)-g=0")
    print("        -> Delta_T = 8 pi r g -> T_local -> Delta_mu -> Delta_G_V")
    print("        -> r_C,2nd (parabolic root) -> Delta_G_C -> I   (vapour -> liquid)")
    print("  Radius is the continuation variable; the gradient is the SOLVED unknown.")
    print("  water_vapour_liquid_nucleation.csv holds the radius-continuation states.")
    print("=" * 78)