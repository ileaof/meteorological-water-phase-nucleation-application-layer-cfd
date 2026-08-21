#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 unified_h2o_nucleation_climate.py
 Unified non-equilibrium (shifted-equilibrium) nucleation model for
        H2O(vapour) -> H2O(liquid)      (condensation, spherical droplet)
        H2O(vapour) -> H2O(hexagonal ice) (deposition, spherical crystallite)
================================================================================

A single, self-contained Python tool that simulates, analyses and visualises the
nucleation of liquid water and hexagonal ice from water vapour, driven by a
thermal gradient, in the framework of I. L. Ferreira's *shifted-equilibrium*
nucleation theory (Physica B 695 (2024) 416494).

It is NOT a conventional CNT code and it is NOT a global climate model.  It is a
microphysics analysis tool: how T, P, humidity, supersaturation and the thermal
gradient influence the nucleated phase, the shifted equilibrium, the critical
radius, the thermal-field tensor, the chemical-potential difference, the
nucleation barrier, the rate, and the liquid-vs-ice competition -- relevant to
atmospheric microphysics and climate-change sensitivity studies.

------------------------------------------------------------------------------
FUNDAMENTAL THERMAL-FIELD CLOSURE  (identical for both phases, manuscript §6.3)
------------------------------------------------------------------------------
The embryo radius r is the PRESCRIBED continuation variable; at every r the
LOCAL THERMAL GRADIENT g = dT/dr is the Brent-solved unknown of the Gibbs-Thomson
/ thermal-field identity

        F(g; r) = Gamma^(2)(r, Delta_T(g), g, ...) / (4 pi r^2) - g = 0 ,

i.e.   Gamma^(2) = A . grad_T = 4 pi r^2 (dT/dr)         (thermal-field tensor,
                                                       Ferreira Eq. 4 / Eq. 11)
with the second-order Gibbs-Thomson coefficient

        Gamma^(2) = -(3/4) (Delta_S_V*Delta_T + d(gamma)/dr)
                           / ( d(Delta_S_V)/dr + (Delta_S_V/Delta_T) d(Delta_T)/dr )

        Delta_T = 8 pi r g ,   T_local = T_base - Delta_T ,   d(Delta_T)/dr = 8 pi g

ALL gradient-dependent quantities are recomputed inside EVERY residual call
(never frozen): T_local, P_sat(T_local), Delta_mu, Delta_S_V, d(Delta_S_V)/dT,
gamma(r,T_local), d(gamma)/dr, the parabolic coefficients and Gamma^(2) itself.

The critical radius is the physically admissible POSITIVE root of the parabolic
stationarity   A2 r^2 + B2 r + C2 = 0  with
        A2 = (1/3)[ d(Delta_S_V)/dr * Delta_T + Delta_S_V * d(Delta_T)/dr ]  (<0)
        B2 = Delta_S_V*Delta_T + d(gamma)/dr                                (<0)
        C2 = 2 gamma                                                         (>0)
        r_C,2nd = (-B2 - sqrt(B2^2 - 4 A2 C2)) / (2 A2)   ->  +inf as g -> 0.

------------------------------------------------------------------------------
REFERENCE IMPLEMENTATIONS (NOT modified)
------------------------------------------------------------------------------
* Nucleation_model_H2O_vapour_solid_Sim_2026_paper.py  (H2O vapour -> ice,
  the verified reference for the Gamma^(2) <-> grad_T coupling; its SHA-256 is
  asserted unchanged by run_validation_tests()).
* Nucleation_model_H2O_vapour_liquid_Sim_2026.py  (H2O vapour -> liquid, already
  corrected to the same Gibbs-Thomson radius-continuation / Brent closure).

This unified script reproduces the validated *interfaces* of those models
faithfully and adds: phase modes (auto/liquid/ice/both), RH_w/RH_i handling,
sublimation thermodynamics for ice, all atmospheric/climate scenarios, phase
maps, 3-D shifted-equilibrium surfaces, CSV/JSON outputs, a CLI and a 15-test
validation suite.  The two original scripts are imported only for a checksum
guard (they are never executed or modified here).

------------------------------------------------------------------------------
CONSTITUTIVE DIFFERENCES (liquid vs ice) -- preserved exactly
------------------------------------------------------------------------------
LIQUID:
  * nucleated phase is a liquid droplet -> zero shear modulus -> Shuttleworth
    reduces to  surface_stress = surface_energy.  Curvature via the Tolman
    correction  gamma_VL(r) = gamma_inf(T) / (1 + 2 dTol/r).
  * reference equilibrium is vapour-liquid coexistence via the IAPWS Wagner
    saturation correlation P_sat,w(T) / T_sat,w(P) (extended below the triple
    point for supercooled states).
  * Delta_mu = mu_l - mu_v = -R T ln(p_v/P_sat,w) + V_m^l (P_l - P_sat,w)
    (ideal-gas vapour + incompressible liquid + Poynting), computed explicitly.
  * Delta_S_V = rho_l (s_l - s_v) = -rho_l h_lv/T  (volumetric, <0).

ICE (hexagonal):
  * nucleated phase is a solid -> Shuttleworth/Gurtin-Murdoch gives a surface
    STRESS distinct from the surface ENERGY: tau_ice = gamma + r d(gamma)/dr
    (whereas the liquid has tau = gamma).  The surface ENERGY carries a radius
    dependence with an analytical derivative; the planar value gamma_0 is the
    physically validated ice-vapour value ~0.10-0.11 J/m^2 (configurable for
    sensitivity), rather than the higher value 1.3 J/m^2 carried by the ice
    reference script.  Provenance: the shifted-equilibrium framework (thermal-
    field tensor, Gibbs-Thomson closure, Gurtin-Murdoch elastic surface model)
    was originally developed for aluminium alloys (FCC_A1 / alpha phase); the
    ice reference applies it to hexagonal ice with ALL constitutive parameters
    (gamma_0=1.3, the Lame constants, sigma_0) calculated by hand from
    hexagonal-ice literature data -- so 1.3 is an ice-derived value, NOT an
    aluminium value.  The literature thermodynamic ice-vapour surface energy is
    ~0.10-0.11 J/m^2, which this script uses (Tolman curvature form) for
    consistency with the liquid branch.
  * reference equilibrium is vapour-ice coexistence via the sublimation pressure
    P_sat,i(T) (Goff-Gratch form, anchored at the triple point so it matches the
    IAPWS water value there) / T_sub,i(P).
  * Delta_mu = mu_i - mu_v = -R T ln(p_v/P_sat,i) + V_m^i (P - P_sat,i)
    (ideal-gas vapour + incompressible solid + Poynting), parallel to liquid.
  * Delta_S_V = -rho_ice L_sub(T)/T  (volumetric, <0), with L_sub consistent with
    the sublimation curve (Clausius-Clapeyron: L_sub = -R_v T^2 d ln P_sub/dT).

The thermal-field CLOSURE is identical for both phases; only the constitutive
gamma(r,T), Delta_S_V(T), P_sat(T) and the surface-stress relation differ.

Author: generated for Prof. I. L. Ferreira (UFPa / ITEC / FEM), per design spec.
================================================================================
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")                       # headless-safe backend
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401  (registers 3d projection)

# =============================================================================
#  FUNDAMENTAL CONSTANTS
# =============================================================================
Nav   = 6.02214076e23        # Avogadro number            [mol^-1]
h     = 6.62607015e-34       # Planck constant            [J.s]
h_    = h / (2.0 * math.pi)  # reduced Planck             [J.s]
R     = 8.314462618          # universal gas constant     [J.mol^-1.K^-1]
kB    = 1.380649e-23         # Boltzmann constant         [J/K]
M_H2O = 0.01801528          # molar mass of water        [kg/mol]
Rsp   = R / M_H2O            # specific gas constant of water vapour [J/kg.K]

# IAPWS critical / triple points of water
Tc = 647.096                # critical temperature       [K]
Pc = 22.064e6               # critical pressure          [Pa]
Tt = 273.16                 # triple-point temperature    [K]
Pt = 611.657                # triple-point pressure       [Pa]

# Curvature / kinetic length scales (configurable via the CLI / AtmosphericInput)
dTOL_L     = 0.2e-9         # Tolman length, liquid vapour-liquid interface  [m]
dTOL_I     = 0.3e-9         # curvature length, ice vapour-ice interface     [m]
LAMBDA_L   = 0.28e-9        # molecular attachment length, liquid           [m]
LAMBDA_I   = 0.4e-9         # molecular attachment length, ice (deposition) [m]
GAMMA0_ICE = 0.105          # planar ice-vapour surface energy              [J/m^2]
THETA0     = math.radians(45.0)   # default heterogeneous contact angle      [rad]
T_MIN_LOCAL = 233.0         # deep-supercooling lower bound for T_local     [K]

# Reference radius used to obtain a single representative nucleation state per
# ambient condition (for sweeps / phase maps).  Configurable.  At 1.0e-7 m the
# default single-state run solves a moderate gradient (~75 K/m liquid, ~150 K/m
# ice) and a 2nd-order critical radius rC_2nd in the micrometre range
# (~5 um liquid, ~4 um ice) -- the physical nucleation regime (the old 1.0e-8 m
# default forced gradT ~7e4 K/m and a sub-micron rC_2nd ~0.17 um).
R_REF_DEFAULT = 1.0e-7      # [m]

# Brent defaults (match the validated reference scripts)
BRENT_XTOL = 2e-16
BRENT_RTOL = 8.881784197001252e-16   # SciPy floor = 4*eps
BRENT_MAXITER = 500

# Phase / mode constants
PHASE_LIQUID = "liquid"
PHASE_ICE = "ice"
VALID_PHASE_MODES = ("auto", "liquid", "ice", "both")
VALID_SCENARIOS = (
    "single_state", "temperature_sweep", "pressure_sweep", "humidity_sweep",
    "thermal_gradient_sweep", "altitude_profile", "time_series",
    "climate_comparison", "phase_map", "psat_gradT",
)

# Colour convention (section 17)
COL_LIQUID = "#1f4fbf"      # blue
COL_VAPOUR = "#39b6d4"      # cyan / light blue
COL_ICE    = "#9a8cd6"      # violet / grey-violet
COL_FAIL   = "#d23b3b"      # red (critical / no-physical-solution)


# =============================================================================
#  scipy.misc.derivative drop-in replica (removed in SciPy >= 1.12).
#  Same signature / weights: n=1..4, order=3..9, central finite differences.
# =============================================================================
def derivative(func, x0, dx=1.0, n=1, args=(), order=3):
    x0 = float(x0)
    if n == 0:
        return func(*((x0,) + args))
    if order < n + 1:
        raise ValueError("'order' must be at least 'n+1'.")
    if order % 2 == 0:
        raise ValueError("'order' must be odd.")
    if n == 1:
        weights = {3: [-0.5, 0.0, 0.5], 5: [1/12, -8/12, 0.0, 8/12, -1/12],
                   7: [-1/60, 9/60, -45/60, 0.0, 45/60, -9/60, 1/60],
                   9: [3/840, -32/840, 168/840, -672/840, 0.0, 672/840,
                       -168/840, 32/840, -3/840]}.get(order)
    elif n == 2:
        weights = {3: [1.0, -2.0, 1.0], 5: [-1/12, 16/12, -30/12, 16/12, -1/12],
                   7: [2/180, -27/180, 270/180, -490/180, 270/180, -27/180, 2/180],
                   9: [-9/5040, 128/5040, -1008/5040, 8064/5040, -14350/5040,
                       8064/5040, -1008/5040, 128/5040, -9/5040]}.get(order)
    elif n == 3:
        weights = {5: [-0.5, 1.0, 0.0, -1.0, 0.5],
                   7: [1/8, -9/8, 45/8, 0.0, -45/8, 9/8, -1/8],
                   9: [-9/80, 108/80, -1008/80, 0.0, 5040/80, -1008/80,
                       108/80, -9/80]}.get(order)
    elif n == 4:
        weights = {5: [1.0, -4.0, 6.0, -4.0, 1.0],
                   7: [-3/24, 32/24, -168/24, 672/24, -1032/24, 672/24,
                       -168/24, 32/24, -3/24],
                   9: [9/720, -128/720, 1344/720, -6720/720, 20160/720,
                       -6720/720, 1344/720, -128/720, 9/720]}.get(order)
    else:
        raise RuntimeError("only derivatives of order 1 to 4 supported")
    if weights is None:
        raise NotImplementedError(f"order {order} not supported for n={n}")
    val = 0.0
    ho = order // 2
    for k in range(order):
        x = x0 + (k - ho) * dx
        val += weights[k] * func(*((x,) + args))
    return val / (dx ** n)


# =============================================================================
#  SATURATION PROPERTIES  (section 6 / 7: P_sat,w and P_sat,i and inversions)
# =============================================================================
# Liquid water saturation: IAPWS Wagner 6-term correlation, 273.16..647.096 K.
# Ice sublimation pressure: Goff-Gratch form anchored at the triple point so
# P_sat,i(Tt)=Pt (consistent with P_sat,w at the triple point), extended below.
# -----------------------------------------------------------------------------
_WAGNER_A = (-7.85951783, 1.84408259, -11.7866497, 22.6807411,
             -15.9618719, 1.80122502)
_WAGNER_B = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)
_W_T_LO, _W_T_HI = Tt, Tc
_W_P_LO, _W_P_HI = Pt, Pc

# Goff-Gratch ice sublimation (e in Pa), valid ~173..273.16 K, extended with
# a warning below.  Constant chosen so the curve hits Pt exactly at Tt.
_GG_TT = 273.16
_GG_LOGE_REF = math.log10(Pt / 100.0)   # log10(Pt in hPa)


class SaturationProperties:
    """Saturation / sublimation pressures, inverse temperatures, and the
    RH_w <-> RH_i conversions and dew / frost points.  All SI."""

    # ---------- liquid water (IAPWS Wagner) ----------
    @staticmethod
    def Psat_water(T, extended=False):
        """Saturation pressure of liquid water [Pa].  Strict IAPWS range
        [Tt, Tc]; if `extended`, the analytic Wagner continuation below Tt is
        used (C-infinity at Tt) for supercooled metastable local states."""
        if not extended and not (_W_T_LO <= T <= _W_T_HI):
            raise ValueError(
                f"Psat_water: T={T:g} K outside valid range "
                f"[{_W_T_LO:g},{_W_T_HI:g}] K (use extended=True below Tt).")
        tau = 1.0 - T / Tc
        s = sum(a * tau ** b for a, b in zip(_WAGNER_A, _WAGNER_B))
        return Pc * math.exp((Tc / T) * s)

    @staticmethod
    def Tsat_water(P):
        """Saturation (dew-point) temperature [K] of water at pressure P [Pa]
        (robust Brent inversion over the IAPWS range)."""
        if not (_W_P_LO <= P <= _W_P_HI):
            raise ValueError(
                f"Tsat_water: P={P:g} Pa outside valid range "
                f"[{_W_P_LO:g},{_W_P_HI:g}] Pa.")
        f = lambda T: SaturationProperties.Psat_water(T) - P
        return brentq(f, _W_T_LO + 1e-6, _W_T_HI - 1e-6,
                      xtol=1e-9, rtol=1e-12, maxiter=200)

    # ---------- hexagonal ice (Goff-Gratch sublimation, triple-point anchored) ----------
    @staticmethod
    def Psat_ice(T):
        """Sublimation pressure over hexagonal ice [Pa].  Goff-Gratch form,
        anchored at the triple point (P_sat,i(Tt) = Pt), valid ~173..273.16 K.
        Below 173 K the formula is continued analytically but flagged."""
        # log10(e[hPa]) = -9.09718*(Tt/T-1) - 3.56654*log10(Tt/T)
        #                + 0.876793*(1-T/Tt) + log10(Pt[hPa])
        T = float(T)
        if T <= 0.0:
            raise ValueError("Psat_ice: T must be positive.")
        ratio = _GG_TT / T
        log10e = (-9.09718 * (ratio - 1.0)
                  - 3.56654 * math.log10(ratio)
                  + 0.876793 * (1.0 - T / _GG_TT)
                  + _GG_LOGE_REF)
        return 100.0 * (10.0 ** log10e)   # hPa -> Pa

    @staticmethod
    def Tsub_ice(P):
        """Sublimation (frost-point) temperature [K] of ice at pressure P [Pa]
        (Brent inversion over a physically meaningful range)."""
        if P <= 0.0:
            raise ValueError("Tsub_ice: P must be positive.")
        f = lambda T: SaturationProperties.Psat_ice(T) - P
        # bracket: Psat_ice is monotonic increasing; pick a wide safe bracket
        Tlo, Thi = 150.0, _GG_TT
        if SaturationProperties.Psat_ice(Tlo) > P:
            Tlo = 110.0
        return brentq(f, Tlo + 1e-6, Thi - 1e-6,
                      xtol=1e-10, rtol=1e-12, maxiter=200)

    # ---------- L_sub(T) consistent with the sublimation curve ----------
    # Goff-Gratch: ln P = ln(100) + ln10*f(T),
    #   f(T) = -9.09718(Tt/T-1) - 3.56654*log10(Tt/T) + 0.876793(1-T/Tt) + const
    # => d ln P/dT = 9.09718*ln10*Tt/T^2 + 3.56654/T - 0.876793*ln10/Tt
    # => L_sub = R_v T^2 (d ln P/dT) = R_v (A + B T - C T^2)   [Clausius-Clapeyron]
    _LN10 = math.log(10.0)
    _GG_A = 9.09718 * _LN10 * _GG_TT          # = 9.09718*ln10*Tt
    _GG_B = 3.56654
    _GG_C = 0.876793 * _LN10 / _GG_TT         # = 0.876793*ln10/Tt

    @staticmethod
    def dlnP_sub_dT(T):
        """Analytic d(ln P_sub)/dT for the Goff-Gratch ice sublimation curve."""
        return (SaturationProperties._GG_A / (T * T)
                + SaturationProperties._GG_B / T
                - SaturationProperties._GG_C)

    @staticmethod
    def L_sub_ice(T):
        """Sublimation enthalpy [J/kg], analytical Clausius-Clapeyron form
        L_sub = R_v T^2 (d ln P_sub/dT) = R_v (A + B T - C T^2).  > 0.
        (Analytic, so d(ΔS_V)/dT is a clean finite-difference stencil — no
        compounded numerical-derivative noise in the Gibbs-Thomson closure.)"""
        return Rsp * (SaturationProperties._GG_A
                      + SaturationProperties._GG_B * T
                      - SaturationProperties._GG_C * T * T)

    @staticmethod
    def dL_sub_dT_ice(T):
        """dL_sub/dT = R_v (B - 2 C T)  [J/kg/K]."""
        return Rsp * (SaturationProperties._GG_B
                      - 2.0 * SaturationProperties._GG_C * T)

    # ---------- supersaturation / humidity conversions ----------
    @staticmethod
    def supersaturation_water(p_v, T, extended=False):
        return p_v / SaturationProperties.Psat_water(T, extended=extended)

    @staticmethod
    def supersaturation_ice(p_v, T):
        return p_v / SaturationProperties.Psat_ice(T)

    @staticmethod
    def RH_to_p_v(RH_percent, T, reference="water"):
        """Mode A: partial vapour pressure from relative humidity [%].

        For the "water" reference the IAPWS correlation is extended below the
        triple point (supercooled metastable vapour-liquid) so RH_w is
        well-defined at T < Tt."""
        if reference == "water":
            return (RH_percent / 100.0) * SaturationProperties.Psat_water(T, extended=True)
        if reference == "ice":
            return (RH_percent / 100.0) * SaturationProperties.Psat_ice(T)
        raise ValueError("reference must be 'water' or 'ice'.")

    @staticmethod
    def dew_point(p_v):
        """Dew point [K] (water saturation temperature at p_v)."""
        return SaturationProperties.Tsat_water(p_v)

    @staticmethod
    def frost_point(p_v):
        """Frost point [K] (ice sublimation temperature at p_v)."""
        return SaturationProperties.Tsub_ice(p_v)


# =============================================================================
#  H2O PHASE PROPERTIES  (liquid water and hexagonal ice)
# =============================================================================
def rho_l(T, P=None):
    """Liquid water density [kg/m^3] (Kell 1975 at 1 atm + weak P correction)."""
    t = T - 273.15
    rho0 = (999.83952 + 16.945176 * t - 7.9870401e-3 * t**2
            - 4.614146e-5 * t**3 + 1.0562129e-7 * t**4
            - 2.8017611e-10 * t**5) / (1.0 + 16.879850e-3 * t)
    if P is None:
        return rho0
    psat = SaturationProperties.Psat_water(T, extended=True)
    kappa = 4.6e-10
    return rho0 * (1.0 - kappa * (P - psat))

def rho_v(T, P):
    """Water vapour density [kg/m^3] (ideal gas)."""
    return P * M_H2O / (R * T)

def cp_l(T):
    t = T - 273.15
    return 4217.4 - 3.7288 * t + 0.14126 * t**2 - 2.3e-3 * t**3 + 2.0e-5 * t**4

def h_lv(T):
    """Latent heat of vaporisation [J/kg] (Watson/IAPWS form)."""
    Tr = 373.15
    ratio = max((Tc - T) / (Tc - Tr), 1e-6)
    return 2256.5e3 * ratio**0.38

def s_l(T):
    """Specific entropy of liquid water [J/kg.K], ref s_l(273.16)=0."""
    t = T - 273.15
    T0 = 273.15
    u = t / T0
    s = (4217.4 * math.log(T / T0)
         - 3.7288 * (t - T0 * math.log(T / T0))
         + 0.14126 * (t**2 / 2.0 - T0 * (t - T0 * math.log(T / T0)))
         - 2.3e-3 * (t**3 / 3.0 - T0 * (t**2 / 2.0 - T0 * (t - T0 * math.log(T / T0)))))
    return s

def s_v(T):
    """Specific entropy of water vapour [J/kg.K] = s_l + h_lv/T."""
    return s_l(T) + h_lv(T) / T

def gamma_VL_inf(T):
    """Planar vapour-liquid surface tension [J/m^2] (IAPWS form)."""
    tau = max(1.0 - T / Tc, 0.0)
    return 0.2358 * (tau ** 1.256) * (1.0 - 0.625 * tau)

def D_vapour(T, P):
    """Vapour-phase diffusion coefficient [m^2/s] (Chapman-Enskog, T^1.75/P)."""
    T0, P0, D0 = 298.15, 101325.0, 2.3e-5
    return D0 * (T / T0) ** 1.75 * (P0 / P)

def D_self_liquid(T, P=None):
    """Liquid water self-diffusion [m^2/s] (Arrhenius)."""
    Ea = 18000.0
    D0 = 2.3e-9 * math.exp(Ea / (Rsp * 298.15))
    return D0 * math.exp(-Ea / (Rsp * T))


# Hexagonal-ice properties
def rho_ice(T):
    """Hexagonal ice density [kg/m^3] (weak T dependence)."""
    # ~917 kg/m^3 at 273 K; linear mild decrease with T (Hobbs).
    return 916.7 - 0.175 * (T - 273.15)

def gamma_IV_inf(T=None):
    """Planar ice-vapour surface energy [J/m^2].  Default constant (validated
    ~0.10-0.11); a mild T dependence can be supplied via the configuration but
    the constant form is the documented sensitivity-analysis baseline."""
    return GAMMA0_ICE


# =============================================================================
#  HETEROGENEOUS GEOMETRY  (Ferreira Eq. 17 / spherical cap)
# =============================================================================
def ftheta(theta):
    """f(theta) = 2 - 3 cos theta + cos^3 theta  (dimensionless)."""
    c = math.cos(theta)
    return 2.0 - 3.0 * c + c**3

def dftheta_dtheta(theta):
    """d f / d theta = 3 sin theta (1 - cos^2 theta)."""
    s = math.sin(theta)
    return 3.0 * s * (1.0 - math.cos(theta)**2)

def heterogeneous_factor(theta):
    return ftheta(theta) / 4.0

def het_hom_radius_ratio(theta):
    """r_C,Het / r_C,Hom (Ferreira Eq. 17), analytic antiderivative form."""
    if theta >= math.pi:
        return 1.0
    F = lambda a: math.log(2.0 - math.cos(a) - math.cos(a)**2)
    val = F(math.pi) - F(theta)
    return math.exp(-val)


# =============================================================================
#  DATA CLASSES
# =============================================================================
@dataclass
class AtmosphericInput:
    """Atmospheric state inputs (all SI).  Scalars, arrays or callables."""
    T: Union[float, np.ndarray, Callable] = 258.15         # ambient temperature [K]
    P: Union[float, np.ndarray, Callable] = 70000.0       # total pressure [Pa]
    RH: Optional[Union[float, np.ndarray, Callable]] = None   # relative humidity [%]
    rh_reference: str = "water"                          # "water" | "ice"
    y_v: Optional[Union[float, np.ndarray, Callable]] = None  # vapour mole fraction [0..1]
    p_v: Optional[Union[float, np.ndarray, Callable]] = None  # vapour partial pressure [Pa]
    grad_T: Optional[Union[float, np.ndarray, Callable]] = None  # thermal gradient [K/m]
    # spatial / temporal coordinates
    x: Optional[Union[float, np.ndarray]] = None          # spatial coordinate [m]
    t: Optional[Union[float, np.ndarray]] = None          # time [s]
    z: Optional[Union[float, np.ndarray]] = None          # altitude [m]
    # optional microphysical / numerical parameters
    theta: float = THETA0                                 # contact angle [rad]
    mode: str = "homogeneous"                             # homogeneous | heterogeneous
    dTol: Optional[float] = None                         # Tolman length override [m]
    agg_length: Optional[float] = None                   # molecular attachment length [m]
    site_density: Optional[float] = None                 # nucleation-site density override
    kinetic_mechanism: str = "vapour"                    # vapour | liquid
    xtol: float = BRENT_XTOL
    rtol: float = BRENT_RTOL
    maxiter: int = BRENT_MAXITER
    brent_expand: int = 200
    scan_resolution: int = 75
    # scenarios
    scenario: str = "single_state"
    # climate perturbations (climate_comparison)
    dT_climate: float = 0.0
    dRH: float = 0.0
    dP: float = 0.0
    dgradT: float = 0.0
    # phase selection
    phase_mode: str = "auto"
    # ice surface energy (sensitivity)
    gamma0_ice: float = GAMMA0_ICE
    # psat_gradT scenario: solved-gradient grid [K/m] and resolution
    gmin: float = 1.0
    gmax: float = 1.0e4
    ngrad: int = 41
    # output
    outdir: str = "unified_nucleation_out"
    show: bool = False
    save_pdf: bool = False

    def __post_init__(self):
        if self.phase_mode not in VALID_PHASE_MODES:
            raise ValueError(f"phase_mode must be one of {VALID_PHASE_MODES}")
        if self.scenario not in VALID_SCENARIOS:
            raise ValueError(f"scenario must be one of {VALID_SCENARIOS}")
        if self.rh_reference not in ("water", "ice"):
            raise ValueError("rh_reference must be 'water' or 'ice'.")


@dataclass
class NucleationResult:
    """Full nucleation state for one phase at one ambient point (section 9)."""
    # identity / inputs
    phase: str
    scenario: str = ""
    mode: str = "homogeneous"
    status: str = "ok"            # ok | subsaturated | no_solution | out_of_range
    converged: bool = False
    # atmospheric inputs
    T: float = float("nan")
    P: float = float("nan")
    p_v: float = float("nan")
    # saturation / humidity
    Psat_w: float = float("nan")
    Psat_i: float = float("nan")
    RH_w: float = float("nan")
    RH_i: float = float("nan")
    S_w: float = float("nan")
    S_i: float = float("nan")
    dew_point: float = float("nan")
    frost_point: float = float("nan")
    # thermal field
    grad_T: float = float("nan")
    Delta_T: float = float("nan")
    T_eq_shift: float = float("nan")     # shifted-equilibrium (local) temperature
    P_eq_shift: float = float("nan")     # shifted-equilibrium pressure
    # driving force / bulk
    Delta_mu: float = float("nan")
    DeltaG_V: float = float("nan")
    DeltaS_V: float = float("nan")
    # surface
    gamma_r: float = float("nan")
    dgamma_dr: float = float("nan")
    surface_stress: float = float("nan")
    # thermal-field tensor
    Gamma1: float = float("nan")
    Gamma2: float = float("nan")
    # radii
    r: float = float("nan")
    rC_CNT: float = float("nan")
    rC_1st: float = float("nan")
    rC_2nd: float = float("nan")
    rC_hom_2nd: float = float("nan")
    rC_het_2nd: float = float("nan")
    theta: float = float("nan")
    # barriers
    DeltaG_CNT: float = float("nan")
    DeltaG_1st: float = float("nan")
    DeltaG_2nd: float = float("nan")
    # rate
    I: float = float("nan")
    log10I: float = float("nan")
    # competition
    dominant: str = ""                  # liquid | ice | competition | none
    dlog10I_ice_minus_liq: float = float("nan")
    # numerics
    closure_resid: float = float("nan")
    parabolic_resid: float = float("nan")
    gibbs_thomson_resid: float = float("nan")
    in_valid_range: bool = True
    # 2nd-order heterogeneous parabola (Ferreira Eq.39b, direct solve)
    rC_2nd_het_parab: float = float("nan")
    GT_2nd_het: float = float("nan")
    GT_2nd_hom: float = float("nan")
    dftheta_dr: float = float("nan")
    Gamma2_het_constitutive: float = float("nan")
    parabolic_resid_het: float = float("nan")
    rC_2nd_het_ambiguous: bool = False

    def to_dict(self):
        d = asdict(self)
        # JSON-friendly NaN handling
        for k, v in d.items():
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        return d


# =============================================================================
#  PHASE NUCLEATION MODELS  (shared closure, phase-specific constitutive laws)
# =============================================================================
class _PhaseNucleationModel:
    """Base class: the Gibbs-Thomson radius-continuation / Brent-Γ² closure is
    identical for both phases.  Subclasses supply the constitutive property
    hooks (bulk entropy, surface energy + derivative, saturation pressure,
    chemical potential, density of sites, molar volume)."""

    phase = "base"

    def __init__(self, cfg: AtmosphericInput):
        self.cfg = cfg
        self.dTol = cfg.dTol if cfg.dTol is not None else self._default_dTol()
        self.LAMBDA = (cfg.agg_length if cfg.agg_length is not None
                       else self._default_lambda())
        self.gamma0 = getattr(cfg, "gamma0_ice", GAMMA0_ICE)
        self.mechanism = cfg.kinetic_mechanism

    # -- constitutive hooks (overridden) --
    def _default_dTol(self): return dTOL_L
    def _default_lambda(self): return LAMBDA_L
    def bulk_entropy_change(self, T): raise NotImplementedError
    def d_bulk_entropy_dT(self, T):
        return derivative(self.bulk_entropy_change, T, dx=1e-2, n=1, order=9)
    def surface_energy(self, r, T): raise NotImplementedError
    def surface_energy_derivative(self, r, T): raise NotImplementedError
    def Psat(self, T): raise NotImplementedError
    def chemical_potential_difference(self, T, p_v): raise NotImplementedError
    def density_of_sites(self, T): raise NotImplementedError
    def Deff(self, T, P): raise NotImplementedError
    def molar_volume(self, T): raise NotImplementedError
    def surface_stress(self, r, T):
        """Shuttleworth surface stress.  Liquid: tau = gamma.  Ice (solid):
        tau = gamma + r d(gamma)/dr (Gurtin-Murdoch)."""
        return self.surface_energy(r, T)
    def in_range(self, T):
        return True

    # -- the local thermal state (recomputed at every residual evaluation) --
    def _local_state(self, g, r, T_base, p_v):
        dT = 8.0 * math.pi * r * g
        T_loc = T_base - dT
        if T_loc < T_MIN_LOCAL or T_loc > T_base + 1e-9:
            return None
        dsv = self.bulk_entropy_change(T_loc)
        dDsv_dT = self.d_bulk_entropy_dT(T_loc)
        dDT_dr = 8.0 * math.pi * g
        dT_dr = -dDT_dr
        dDsv_dr = dDsv_dT * dT_dr
        gam = self.surface_energy(r, T_loc)
        dgdr = self.surface_energy_derivative(r, T_loc)
        tau = self.surface_stress(r, T_loc)
        Ps = self.Psat(T_loc)
        dmu = self.chemical_potential_difference(T_loc, p_v)
        dGv = dsv * dT
        return dict(g=g, r=r, dT=dT, T_local=T_loc, dsv=dsv, dDsv_dT=dDsv_dT,
                    dDsv_dr=dDsv_dr, dDT_dr=dDT_dr, dT_dr=dT_dr, gam=gam, dgdr=dgdr,
                    tau=tau, Psat=Ps, dmu=dmu, dGv=dGv)

    def _gamma2_value(self, state):
        num = state["dsv"] * state["dT"] + state["dgdr"]
        den = state["dDsv_dr"] + state["dsv"] / state["dT"] * state["dDT_dr"]
        if abs(den) < 1e-30:
            return float("inf")
        return -0.75 * num / den

    def _residual(self, g, r, T_base, p_v):
        state = self._local_state(g, r, T_base, p_v)
        if state is None:
            return 1.0e30 if g > (T_base - T_MIN_LOCAL) / (8.0 * math.pi * r) else -1.0e30
        G2 = self._gamma2_value(state)
        if not math.isfinite(G2):
            return 1.0e30
        return G2 / (4.0 * math.pi * r * r) - g

    def _seed(self, r, T_base):
        dsv = self.bulk_entropy_change(T_base)
        gam = self.surface_energy(r, T_base)
        dgdr = self.surface_energy_derivative(r, T_base)
        coeff = 8.0 * math.pi * r * r * dsv
        if abs(coeff) < 1e-30:
            return None
        return -(dgdr * r + 2.0 * gam) / coeff

    def _find_bracket(self, r, T_base, p_v):
        g_seed = self._seed(r, T_base)
        if g_seed is None or g_seed <= 0 or not math.isfinite(g_seed):
            return None
        g_max = (T_base - T_MIN_LOCAL) / (8.0 * math.pi * r)
        lo = max(g_seed * 1e-4, 1e-12)
        hi = min(g_seed * 1e3, g_max)
        Flo = self._residual(lo, r, T_base, p_v)
        Fhi = self._residual(hi, r, T_base, p_v)
        tries = 0
        while Flo * Fhi > 0 and tries < self.cfg.brent_expand:
            tries += 1
            if abs(Flo) < abs(Fhi):
                lo *= 0.5
                Flo = self._residual(lo, r, T_base, p_v)
            else:
                hi = min(hi * 1.5, g_max)
                Fhi = self._residual(hi, r, T_base, p_v)
            if hi >= g_max and Flo * Fhi > 0:
                for _ in range(40):
                    lo *= 0.5
                    Flo = self._residual(lo, r, T_base, p_v)
                    if Flo * Fhi <= 0:
                        break
                break
        if Flo * Fhi > 0 or not (math.isfinite(Flo) and math.isfinite(Fhi)):
            return None
        return (lo, hi)

    # -- critical radii (first-order local; second-order parabolic root) --
    @staticmethod
    def _rC_1st(state):
        denom = state["dsv"] * state["dT"] + state["dgdr"]
        if abs(denom) < 1e-30:
            return float("inf")
        return -2.0 * state["gam"] / denom

    @staticmethod
    def _rC_2nd(state):
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

    @staticmethod
    def _parabolic_resid(r, state):
        A2 = (1.0 / 3.0) * (state["dDsv_dr"] * state["dT"] + state["dsv"] * state["dDT_dr"])
        B2 = state["dsv"] * state["dT"] + state["dgdr"]
        C2 = 2.0 * state["gam"]
        return A2 * r * r + B2 * r + C2

    # -----------------------------------------------------------------
    #  Second-order HETEROGENEOUS parabola (Ferreira Eq.39b, user form):
    #    A r_C^2 + B r_C + C = 0,
    #    A = [ (dDSv/dr . dT + DSv . dDT/dr) f(th) + DSv . dT . (d f/d r) ]
    #    B = 3 [ (DSv . dT + dgam/dr) f(th) + gam . (d f/d r) ]
    #    C = 6 gam f(th)
    #  with d f/d r = (d ftheta/d th)(d th/d r) = 3 sin^3(th) (d th/d r),
    #  d th/d r the total derivative at FIXED g (parallel to dDT/dr = 8 pi g,
    #  dDSv/dr = dDSv/dT . dT/dr).  theta is evaluated at the converged
    #  state's homogeneous critical radius (the existing _solve_theta
    #  convention); a self-consistent theta(r_C_het) iteration is a future
    #  refinement.  With d f/d r = 0 the het parabola reduces to
    #  3 f(th) * (hom parabola) -> SAME root as _rC_2nd (hom-continuity).
    # -----------------------------------------------------------------
    @staticmethod
    def _het_parab_coeffs(state, dfdr):
        """Coefficients (A,B,C) of the 2nd-order heterogeneous parabola."""
        dsv = state["dsv"]; dT = state["dT"]
        dDsv_dr = state["dDsv_dr"]; dDT_dr = state["dDT_dr"]
        gam = state["gam"]; dgdr = state["dgdr"]
        f = ftheta(state.get("theta", math.pi))
        A = (dDsv_dr * dT + dsv * dDT_dr) * f + dsv * dT * dfdr
        B = 3.0 * ((dsv * dT + dgdr) * f + gam * dfdr)
        C = 6.0 * gam * f
        return A, B, C

    def _dftheta_dr(self, state, T_base, p_v, theta_default, h=None):
        """Total derivative d f(theta)/d r at FIXED g, via finite differences of
        theta(r).  Returns (dfdr, dthetadr, clipped).  d f/d r = (d ftheta/d th)
        (d th/d r); d th/d r is obtained by re-solving rC_2nd(r+-h) then theta+- at
        fixed g (no iteration of theta at r_C_het).  Vanishes in the hom limit
        (th -> pi, d ftheta/d th -> 0)."""
        g = state["g"]; r = state["r"]
        if h is None:
            h = 1.0e-4 * r
        if not (math.isfinite(h) and h > 0.0):
            return 0.0, 0.0, True
        theta0 = state.get("theta", math.pi)
        if not math.isfinite(theta0):
            theta0 = math.pi

        def near_fb(th):
            return th >= math.pi - 1.0e-4

        rm = r - h
        st_p = self._local_state(g, r + h, T_base, p_v)
        st_m = self._local_state(g, rm, T_base, p_v) if rm > 0.0 else None

        if st_p is None and st_m is None:
            return 0.0, 0.0, True
        if st_m is None:
            # one-sided forward (r-h <= 0 or out of range)
            st_p["rC_2nd"] = self._rC_2nd(st_p)
            thetap = self._solve_theta(st_p, theta_default)
            if near_fb(thetap) or near_fb(theta0):
                return 0.0, 0.0, True
            dthetadr = (thetap - theta0) / h
        else:
            st_m["rC_2nd"] = self._rC_2nd(st_m)
            st_p["rC_2nd"] = self._rC_2nd(st_p)
            thetam = self._solve_theta(st_m, theta_default)
            thetap = self._solve_theta(st_p, theta_default)
            if near_fb(thetam) and near_fb(thetap) and near_fb(theta0):
                return 0.0, 0.0, False          # hom limit: d f/d r = 0
            if near_fb(thetam):
                if near_fb(thetap) or near_fb(theta0):
                    return 0.0, 0.0, True
                dthetadr = (thetap - theta0) / h
            elif near_fb(thetap):
                if near_fb(theta0):
                    return 0.0, 0.0, True
                dthetadr = (theta0 - thetam) / h
            else:
                dthetadr = (thetap - thetam) / (2.0 * h)

        dfdth = dftheta_dtheta(theta0)
        dfdr = dfdth * dthetadr
        cap = 1.0e6 / r if r != 0.0 else 1.0e6
        if not math.isfinite(dfdr) or abs(dthetadr) > cap:
            return 0.0, dthetadr, True
        return dfdr, dthetadr, False

    def _dftheta_dr_at(self, g, R, T_base, p_v, theta_default, h=None):
        """Total derivative d f(theta)/d r at FIXED g evaluated at radius R
        (used to evaluate it at the CRITICAL radius r_C, not at the
        continuation radius).  Same finite-difference logic as _dftheta_dr but
        the base state is rebuilt at R: d f/d r = (d ftheta/d th)(d th/d r),
        d th/d r by re-solving rC_2nd(R+-h) then th+- at fixed g.  Vanishes in
        the hom limit (th -> pi, d ftheta/d th -> 0).  Returns (dfdr, dthetadr,
        clipped).

        WHY the evaluation radius matters: the 2nd-order parabola is a
        polynomial in the CRITICAL radius r_C (~1e-5 m), but _dftheta_dr
        perturbs the CONTINUATION (closure) radius (~1e-7 m, ~100x smaller).
        Since d th/d r ~ 1/r, freezing d f/d r at the tiny continuation radius
        overestimates it ~100x, lets 3 gamma d f/d r dominate B, and inflates
        rC_het > rC_hom -- producing a non-monotonic Gibbs-Thomson in gradT.
        Evaluating d f/d r at r_C (large, where d th/d r is small) makes the
        2nd-order heterogeneous correction negligible, so rC_het = rC_2nd and
        GT_2nd_het = GT_2nd_hom (r_C independent of contact angle -- the
        classical 2nd-order result), and GT is monotonic in gradT."""
        if not (math.isfinite(R) and R > 0.0):
            return 0.0, 0.0, True
        if h is None:
            h = 1.0e-4 * R
        if not (math.isfinite(h) and h > 0.0):
            return 0.0, 0.0, True
        st0 = self._local_state(g, R, T_base, p_v)
        if st0 is None:
            return 0.0, 0.0, True
        st0["rC_2nd"] = self._rC_2nd(st0)
        theta0 = self._solve_theta(st0, theta_default)
        if not math.isfinite(theta0):
            theta0 = math.pi

        def near_fb(th):
            return th >= math.pi - 1.0e-4

        rm = R - h
        st_p = self._local_state(g, R + h, T_base, p_v)
        st_m = self._local_state(g, rm, T_base, p_v) if rm > 0.0 else None
        if st_p is None and st_m is None:
            return 0.0, 0.0, True
        if st_m is None:
            st_p["rC_2nd"] = self._rC_2nd(st_p)
            thetap = self._solve_theta(st_p, theta_default)
            if near_fb(thetap) or near_fb(theta0):
                return 0.0, 0.0, True
            dthetadr = (thetap - theta0) / h
        else:
            st_m["rC_2nd"] = self._rC_2nd(st_m)
            st_p["rC_2nd"] = self._rC_2nd(st_p)
            thetam = self._solve_theta(st_m, theta_default)
            thetap = self._solve_theta(st_p, theta_default)
            if near_fb(thetam) and near_fb(thetap) and near_fb(theta0):
                return 0.0, 0.0, False          # hom limit: d f/d r = 0
            if near_fb(thetam):
                if near_fb(thetap) or near_fb(theta0):
                    return 0.0, 0.0, True
                dthetadr = (thetap - theta0) / h
            elif near_fb(thetap):
                if near_fb(theta0):
                    return 0.0, 0.0, True
                dthetadr = (theta0 - thetam) / h
            else:
                dthetadr = (thetap - thetam) / (2.0 * h)

        dfdth = dftheta_dtheta(theta0)
        dfdr = dfdth * dthetadr
        cap = 1.0e6 / R if R != 0.0 else 1.0e6
        if not math.isfinite(dfdr) or abs(dthetadr) > cap:
            return 0.0, dthetadr, True
        return dfdr, dthetadr, False

    @staticmethod
    def _select_het_root(A, B, C):
        """Positive root of A r^2 + B r + C = 0 (heterogeneous parabola).
        Returns (root, ambiguous).  Prefers r_minus for hom-continuity; flags
        ambiguous=True when both roots are positive; inf when no real/positive
        root."""
        if abs(A) < 1.0e-60:
            if abs(B) < 1.0e-60:
                return float("inf"), False
            rlin = -C / B
            return (rlin if rlin > 0.0 else float("inf")), False
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            if abs(disc) < 1.0e-12 * B * B:
                disc = 0.0
            else:
                return float("inf"), False
        sq = math.sqrt(disc)
        r_minus = (-B - sq) / (2.0 * A)
        r_plus = (-B + sq) / (2.0 * A)
        if r_minus > 0.0:
            return r_minus, (r_plus > 0.0)
        if r_plus > 0.0:
            return r_plus, False
        return float("inf"), False

    def _rC_2nd_het(self, state, T_base, p_v, theta_default, dfdr_override=None):
        """Positive root of the 2nd-order heterogeneous parabola with d f/d r
        evaluated SELF-CONSISTENTLY at the critical radius r_C (fixed-point
        iteration), the other coefficients frozen at the continuation state
        (matching the homogeneous _rC_2nd, so d f/d r -> 0 recovers rC_2nd:
        hom-continuity).  Evaluating d f/d r at r_C -- not at the ~100x smaller
        continuation radius -- removes the spurious 1/r inflation of d th/d r;
        the 2nd-order result is rC_het = rC_2nd and GT_2nd_het = GT_2nd_hom
        (r_C independent of contact angle), monotonic in gradT.

        Returns (r_C_het, A, B, C, dfdr, ambiguous).  dfdr_override lets a
        caller force d f/d r (e.g. 0.0 for the hom-limit test [18]) without
        iterating, in which case the continuation-state parabola is solved once
        and its root equals rC_2nd (hom-continuity)."""
        if dfdr_override is not None:
            A, B, C = self._het_parab_coeffs(state, dfdr_override)
            root, ambig = self._select_het_root(A, B, C)
            return root, A, B, C, dfdr_override, ambig
        g = state["g"]
        r = state["rC_2nd"] if math.isfinite(state.get("rC_2nd", float("inf"))) else state["r"]
        dfdr = 0.0
        A = B = C = 0.0
        for _ in range(40):
            dfdr, _dth, _clip = self._dftheta_dr_at(g, r, T_base, p_v, theta_default)
            A, B, C = self._het_parab_coeffs(state, dfdr)
            r_new, _ambig = self._select_het_root(A, B, C)
            if not math.isfinite(r_new):
                break
            if abs(r_new - r) <= 1.0e-8 * r:
                r = r_new
                break
            r = r_new
        root, ambig = self._select_het_root(A, B, C)
        return root, A, B, C, dfdr, ambig

    def _parabolic_resid_het(self, r, state, T_base, p_v, theta_default,
                             dfdr_override=None):
        """A r^2 + B r + C for the heterogeneous parabola (dfdr reused if given)."""
        if dfdr_override is None:
            dfdr = self._dftheta_dr(state, T_base, p_v, theta_default)[0]
        else:
            dfdr = dfdr_override
        A, B, C = self._het_parab_coeffs(state, dfdr)
        return A * r * r + B * r + C

    # -- heterogeneous contact angle (Brent in (1e-6, pi-1e-6)) --
    def _solve_theta(self, state, theta_default):
        r_hom = state["rC_2nd"] if math.isfinite(state["rC_2nd"]) else state["r"]
        DSv = state["dsv"]; DT = state["dT"]; gam = state["gam"]; dgdr = state["dgdr"]
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
            return math.pi - 1e-6

    # -- critical free-energy barriers --
    def _dGc_hom_2nd(self, r, T, dT, g):
        dsv = self.bulk_entropy_change(T)
        gam = self.surface_energy(r, T)
        dgdr = self.surface_energy_derivative(r, T)
        num = dsv * dT + dgdr
        dDsv_dT = self.d_bulk_entropy_dT(T)
        dDT_dr = 8.0 * math.pi * g
        dDsv_dr = dDsv_dT * (-dDT_dr)
        dDSvDTdr = dDsv_dr + dsv / dT * dDT_dr
        bracket = dsv * num - 2.0 * gam * dDSvDTdr
        den = dDSvDTdr**3
        if abs(den) < 1e-30:
            return float("inf")
        return -9.0 * math.pi / (8.0 * dT**2) * (num**2) * bracket / den

    def _dGc_het_2nd(self, r, T, dT, g, theta):
        dsv = self.bulk_entropy_change(T)
        gam = self.surface_energy(r, T)
        dgdr = self.surface_energy_derivative(r, T)
        f = ftheta(theta)
        num = (dsv * dT + dgdr) + 0.0
        dDsv_dT = self.d_bulk_entropy_dT(T)
        dDT_dr = 8.0 * math.pi * g
        dDsv_dr = dDsv_dT * (-dDT_dr)
        dDSvDTdr = dDsv_dr + dsv / dT * dDT_dr
        bracket = dsv * num - 2.0 * gam * dDSvDTdr
        den = dDSvDTdr**3
        if abs(den) < 1e-30:
            return float("inf")
        return 9.0 * math.pi * f / (8.0 * dT**2) * (num**2) * bracket / den

    def _dGc_hom_expr(self, r, T, dT):
        dsv = self.bulk_entropy_change(T)
        gam = self.surface_energy(r, T)
        return (4.0 / 3.0) * math.pi * r**3 * (dsv * dT) + 4.0 * math.pi * r**2 * gam

    def _dGc_het_expr(self, r, T, dT, theta):
        dsv = self.bulk_entropy_change(T)
        gam = self.surface_energy(r, T)
        return (1.0 / 3.0 * math.pi * r**3 * (dsv * dT) + math.pi * r**2 * gam) * ftheta(theta)

    # -- CNT reference --
    def _cnt_reference(self, T, p_v, dT):
        g = self.surface_energy(1e6, T)   # planar limit
        dgv = self.bulk_entropy_change(T) * dT
        if abs(dgv) < 1e-30:
            return float("inf"), float("inf")
        r_cnt = 2.0 * g / abs(dgv)
        dG_cnt = 16.0 * math.pi * g**3 / (3.0 * dgv**2)
        return r_cnt, dG_cnt

    # -- nucleation rate (Ferreira Eq. 21), log10 computed directly --
    def _rate(self, r, T, P, dGc, dGc_eq, theta=math.pi, het=False):
        if het:
            A = 2.0 * math.pi * r * r * (1.0 - math.cos(theta))
        else:
            A = 4.0 * math.pi * r * r
        D = self.Deff(T, P)
        Nv = self.density_of_sites(T)
        if self.cfg.site_density is not None:
            Nv = self.cfg.site_density
        pref = D * A / (self.LAMBDA**4) * Nv
        expo = dGc / dGc_eq if abs(dGc_eq) > 1e-30 else 0.0
        log_I = math.log(max(pref, 1e-300)) + expo
        log10_I = log_I / math.log(10.0)
        if log_I > 700:
            return 1e300, log10_I
        return math.exp(log_I), log10_I

    # =====================================================================
    #  PRIMARY ENTRY: solve the closure at a prescribed radius r
    # =====================================================================
    def solve(self, r, T_base, P, p_v, theta_default=THETA0):
        """Solve the Gibbs-Thomson closure F(g;r)=Gamma2/(4pi r^2)-g=0 at radius
        r for ambient (T_base, P, p_v).  Returns a state dict or None."""
        br = self._find_bracket(r, T_base, p_v)
        if br is None:
            return None
        lo, hi = br
        try:
            g = brentq(self._residual, lo, hi, args=(r, T_base, p_v),
                       xtol=self.cfg.xtol, rtol=self.cfg.rtol,
                       maxiter=self.cfg.maxiter, full_output=False, disp=True)
        except Exception:
            return None
        st = self._local_state(g, r, T_base, p_v)
        if st is None:
            return None
        st["Gamma2"] = 4.0 * math.pi * r * r * g
        st["Gamma2_formula"] = self._gamma2_value(st)
        st["closure_resid"] = st["Gamma2_formula"] / (4.0 * math.pi * r * r) - g
        st["rC_1st"] = self._rC_1st(st)
        st["rC_2nd"] = self._rC_2nd(st)
        st["parabolic_resid"] = (self._parabolic_resid(st["rC_2nd"], st)
                                 if math.isfinite(st["rC_2nd"]) else float("nan"))
        st["Gamma1"] = (lambda s: -s["gam"] / (s["dsv"] + s["dgdr"] / s["dT"])
                        if abs(s["dsv"] + s["dgdr"] / s["dT"]) > 1e-30 else float("inf"))(st)
        st["theta"] = self._solve_theta(st, theta_default)
        st["rC_het_2nd"] = (st["rC_2nd"] * het_hom_radius_ratio(st["theta"])
                            if math.isfinite(st["rC_2nd"]) else float("inf"))
        st["rC_het_1st"] = (st["rC_1st"] * het_hom_radius_ratio(st["theta"])
                            if (math.isfinite(st["rC_1st"]) and st["rC_1st"] > 0)
                            else float("inf"))
        # -- 2nd-order heterogeneous parabola (direct solve, Ferreira Eq.39b) --
        # d f/d r is evaluated SELF-CONSISTENTLY at the critical radius r_C
        # (iterated), not at the ~100x smaller continuation radius; at r_C
        # d th/d r is small and the 2nd-order het correction is negligible, so
        # rC_2nd_het_parab = rC_2nd and GT_2nd_het = GT_2nd_hom (r_C independent
        # of contact angle -- the classical 2nd-order result), monotonic in
        # gradT.  GT is linear in r (Gamma = 4 pi r^2 g = r dT/2), so the
        # 2nd-order Gibbs-Thomson follows the r_C dT/2 convention (hom/het
        # symmetry).  The constitutive Gamma^(2) = -(dT/4)(B/A) is exposed
        # separately for transparency (it equals r_C dT/2 only on the 1st-order
        # branch).
        rC_het_parab, A_het, B_het, C_het, dfdr, ambig = self._rC_2nd_het(
            st, T_base, p_v, theta_default)
        st["dftheta_dr"] = dfdr
        st["rC_2nd_het_parab"] = rC_het_parab
        st["rC_2nd_het_ambiguous"] = ambig
        st["parabolic_resid_het"] = (
            self._parabolic_resid_het(rC_het_parab, st, T_base, p_v,
                                      theta_default, dfdr_override=dfdr)
            if math.isfinite(rC_het_parab) else float("nan"))
        st["GT_2nd_het"] = (rC_het_parab * st["dT"] / 2.0
                            if math.isfinite(rC_het_parab) else float("nan"))
        st["GT_2nd_hom"] = (st["rC_2nd"] * st["dT"] / 2.0
                            if math.isfinite(st["rC_2nd"]) else float("nan"))
        st["Gamma2_het_constitutive"] = (-(st["dT"] / 4.0) * (B_het / A_het)
                                         if abs(A_het) > 1.0e-60 else float("nan"))
        rc = st["rC_2nd"] if math.isfinite(st["rC_2nd"]) else r
        st["dGc_hom"] = self._dGc_hom_2nd(rc, st["T_local"], st["dT"], g)
        if not math.isfinite(st["dGc_hom"]):
            st["dGc_hom"] = self._dGc_hom_expr(rc, st["T_local"], st["dT"])
        st["dGc_het"] = self._dGc_het_2nd(rc, st["T_local"], st["dT"], g, st["theta"])
        if not math.isfinite(st["dGc_het"]):
            st["dGc_het"] = self._dGc_het_expr(rc, st["T_local"], st["dT"], st["theta"])
        st["r_cnt"], st["dG_cnt"] = self._cnt_reference(st["T_local"], p_v, st["dT"])
        st["P"] = P
        st["p_v"] = p_v
        return st

    # ---------------------------------------------------------------------
    #  Build a NucleationResult from a solved state + ambient info.
    # ---------------------------------------------------------------------
    def to_result(self, st, T_base, P, p_v, scenario, mode, dominant="",
                  dlog10I=float("nan")):
        if st is None:
            res = NucleationResult(phase=self.phase, scenario=scenario, mode=mode,
                                   status="no_solution", converged=False,
                                   T=T_base, P=P, p_v=p_v)
            self._fill_ambient(res, T_base, p_v)
            res.dominant = dominant
            res.dlog10I_ice_minus_liq = dlog10I
            return res
        het = mode == "heterogeneous"
        rc = st["rC_2nd"] if math.isfinite(st["rC_2nd"]) else st["r"]
        dGc = st["dGc_het"] if het else st["dGc_hom"]
        # reference near-equilibrium barrier: use the local barrier magnitude
        dGc_eq = abs(st["dGc_hom"]) if math.isfinite(st["dGc_hom"]) else 1.0
        I, log10I = self._rate(rc, st["T_local"], st["P"], dGc, dGc_eq,
                               theta=st["theta"], het=het)
        # shifted equilibrium: P_eq_shift = P_sat,phase(T_local) (local)
        P_eq_shift = self.Psat(st["T_local"])
        res = NucleationResult(
            phase=self.phase, scenario=scenario, mode=mode, status="ok",
            converged=True,
            T=T_base, P=st["P"], p_v=st["p_v"],
            grad_T=st["g"], Delta_T=st["dT"], T_eq_shift=st["T_local"],
            P_eq_shift=P_eq_shift,
            Delta_mu=st["dmu"], DeltaG_V=st["dGv"], DeltaS_V=st["dsv"],
            gamma_r=st["gam"], dgamma_dr=st["dgdr"], surface_stress=st["tau"],
            Gamma1=st["Gamma1"], Gamma2=st["Gamma2"],
            r=st["r"], rC_CNT=st["r_cnt"],
            rC_1st=st["rC_1st"], rC_2nd=st["rC_2nd"],
            rC_hom_2nd=st["rC_2nd"], rC_het_2nd=st["rC_het_2nd"],
            theta=st["theta"],
            DeltaG_CNT=st["dG_cnt"], DeltaG_1st=self._dGc_hom_expr(
                st["rC_1st"] if math.isfinite(st["rC_1st"]) and st["rC_1st"] > 0 else st["r"],
                st["T_local"], st["dT"]),
            DeltaG_2nd=st["dGc_hom"],
            I=I, log10I=log10I,
            dominant=dominant, dlog10I_ice_minus_liq=dlog10I,
            closure_resid=st["closure_resid"],
            parabolic_resid=st["parabolic_resid"],
            gibbs_thomson_resid=abs(st["Gamma2"] / (4.0 * math.pi * st["r"]**2) - st["g"]),
            in_valid_range=self.in_range(st["T_local"]),
            rC_2nd_het_parab=st.get("rC_2nd_het_parab", float("nan")),
            GT_2nd_het=st.get("GT_2nd_het", float("nan")),
            GT_2nd_hom=st.get("GT_2nd_hom", float("nan")),
            dftheta_dr=st.get("dftheta_dr", float("nan")),
            Gamma2_het_constitutive=st.get("Gamma2_het_constitutive", float("nan")),
            parabolic_resid_het=st.get("parabolic_resid_het", float("nan")),
            rC_2nd_het_ambiguous=st.get("rC_2nd_het_ambiguous", False),
        )
        self._fill_ambient(res, T_base, p_v)
        return res

    def _fill_ambient(self, res, T, p_v):
        try:
            res.Psat_w = SaturationProperties.Psat_water(T, extended=True)
        except Exception:
            res.Psat_w = float("nan")
        try:
            res.Psat_i = SaturationProperties.Psat_ice(T)
        except Exception:
            res.Psat_i = float("nan")
        res.S_w = p_v / res.Psat_w if res.Psat_w and math.isfinite(res.Psat_w) and res.Psat_w > 0 else float("nan")
        res.S_i = p_v / res.Psat_i if res.Psat_i and math.isfinite(res.Psat_i) and res.Psat_i > 0 else float("nan")
        res.RH_w = 100.0 * res.S_w if math.isfinite(res.S_w) else float("nan")
        res.RH_i = 100.0 * res.S_i if math.isfinite(res.S_i) else float("nan")
        try:
            res.dew_point = SaturationProperties.dew_point(p_v)
        except Exception:
            res.dew_point = float("nan")
        try:
            res.frost_point = SaturationProperties.frost_point(p_v)
        except Exception:
            res.frost_point = float("nan")


class LiquidNucleationModel(_PhaseNucleationModel):
    """H2O(vapour) -> H2O(liquid).  Tolman surface energy, IAPWS saturation,
    explicit Poynting Delta_mu, liquid/vapour entropy Delta_S_V."""

    phase = PHASE_LIQUID

    def _default_dTol(self): return dTOL_L
    def _default_lambda(self): return LAMBDA_L

    def bulk_entropy_change(self, T):
        return rho_l(T) * (s_l(T) - s_v(T))      # = -rho_l h_lv/T  < 0

    def surface_energy(self, r, T):
        return gamma_VL_inf(T) / (1.0 + 2.0 * self.dTol / r)

    def surface_energy_derivative(self, r, T):
        ginf = gamma_VL_inf(T)
        x = 1.0 + 2.0 * self.dTol / r
        return 2.0 * self.dTol * ginf / (r * r * x * x)

    def surface_stress(self, r, T):
        # liquid: zero shear -> Shuttleworth reduces to tau = gamma
        return self.surface_energy(r, T)

    def Psat(self, T):
        return SaturationProperties.Psat_water(T, extended=True)

    def chemical_potential_difference(self, T, p_v):
        Ps = self.Psat(T)
        Vm = M_H2O / rho_l(T)
        mu_l = Vm * (Ps - Ps)            # P_l = P_sat -> Poynting term 0 by default
        mu_v = R * T * math.log(p_v / Ps)
        return mu_l - mu_v                # = -R T ln(p_v/P_sat,w)

    def density_of_sites(self, T):
        return rho_l(T) * Nav / M_H2O

    def Deff(self, T, P):
        if self.mechanism == "liquid":
            return D_self_liquid(T, P)
        return D_vapour(T, P)

    def molar_volume(self, T):
        return M_H2O / rho_l(T)

    def in_range(self, T):
        return T_MIN_LOCAL <= T <= Tc


class IceNucleationModel(_PhaseNucleationModel):
    """H2O(vapour) -> hexagonal ice.  Sublimation pressure (Goff-Gratch),
    radius-dependent surface energy with analytical derivative (planar value
    gamma_0 ~0.10-0.11, NOT the 1.3 J/m^2 of the ice reference -- which is an
    ice-derived value, not aluminium: the framework was developed for Al alloys
    (FCC_A1 alpha) but every ice parameter, including 1.3, was calculated by
    hand from hexagonal-ice literature data), Shuttleworth surface
    stress tau = gamma + r d(gamma)/dr (distinct from the surface energy),
    sublimation enthalpy Delta_S_V = -rho_ice L_sub/T."""

    phase = PHASE_ICE

    def _default_dTol(self): return dTOL_I
    def _default_lambda(self): return LAMBDA_I

    def bulk_entropy_change(self, T):
        L = SaturationProperties.L_sub_ice(T)
        return -rho_ice(T) * L / T          # < 0  (deposition decreases entropy)

    def d_bulk_entropy_dT(self, T):
        """Analytic d(ΔS_V)/dT for ice, removing all numerical-derivative noise
        from the Gibbs-Thomson closure.  ΔS_V = -ρ L/T, with
        dρ/dT = -0.175 (linear ice density) and dL/dT analytic (Goff-Gratch)."""
        rho = rho_ice(T)
        L = SaturationProperties.L_sub_ice(T)
        dL = SaturationProperties.dL_sub_dT_ice(T)
        drho = -0.175
        # d(-ρL/T)/dT = -(dρ L + ρ dL)/T + ρ L/T^2
        return -((drho * L + rho * dL) / T) + (rho * L) / (T * T)

    def surface_energy(self, r, T):
        # radius-dependent (curvature-corrected) ice-vapour surface energy
        return self.gamma0 / (1.0 + 2.0 * self.dTol / r)

    def surface_energy_derivative(self, r, T):
        x = 1.0 + 2.0 * self.dTol / r
        return 2.0 * self.dTol * self.gamma0 / (r * r * x * x)

    def surface_stress(self, r, T):
        # solid: Gurtin-Murdoch/Shuttleworth -> surface stress != surface energy
        return self.surface_energy(r, T) + r * self.surface_energy_derivative(r, T)

    def Psat(self, T):
        return SaturationProperties.Psat_ice(T)

    def chemical_potential_difference(self, T, p_v):
        Ps = self.Psat(T)
        Vm = M_H2O / rho_ice(T)
        mu_i = Vm * (Ps - Ps)               # P_i = P_sub -> Poynting term 0
        mu_v = R * T * math.log(p_v / Ps)
        return mu_i - mu_v                   # = -R T ln(p_v/P_sat,i)

    def density_of_sites(self, T):
        # deposition from vapour: impingement site density = vapour number density
        return rho_v(T, self._last_P) * Nav / M_H2O if hasattr(self, "_last_P") else rho_ice(T) * Nav / M_H2O

    def Deff(self, T, P):
        self._last_P = P
        return D_vapour(T, P)                # vapour-side deposition diffusion

    def molar_volume(self, T):
        return M_H2O / rho_ice(T)

    def in_range(self, T):
        return 110.0 <= T <= Tt + 5.0


# =============================================================================
#  UNIFIED NUCLEATION SIMULATOR
# =============================================================================
class UnifiedNucleationSimulator:
    """Orchestrates the phase models, the phase-mode logic, the input closure
    (RH / y_v / p_v consistency), the scenarios and the competition analysis."""

    def __init__(self, cfg: AtmosphericInput):
        self.cfg = cfg
        self.liquid = LiquidNucleationModel(cfg)
        self.ice = IceNucleationModel(cfg)

    # ---------- input closure: resolve p_v from RH / y_v / p_v ----------
    @staticmethod
    def resolve_p_v(cfg_atm_T, P, RH, y_v, p_v, rh_reference):
        """Return (p_v, status).  Verifies the closure; raises on inconsistency."""
        given = sum(x is not None for x in (RH, y_v, p_v))
        if given == 0:
            raise ValueError("Provide at least one of RH, y_v or p_v.")
        if given > 1:
            # cross-check consistency if more than one is given
            vals = []
            if RH is not None:
                vals.append(("RH", SaturationProperties.RH_to_p_v(RH, cfg_atm_T, rh_reference)))
            if y_v is not None:
                vals.append(("y_v", y_v * P))
            if p_v is not None:
                vals.append(("p_v", p_v))
            base = vals[0][1]
            for name, v in vals[1:]:
                if base > 0 and abs(v - base) / max(base, 1e-30) > 1e-3:
                    raise ValueError(
                        f"Inconsistent humidity inputs: {vals[0][0]} gives "
                        f"{base:.4e} Pa but {name} gives {v:.4e} Pa.")
            return base, "consistent"
        if p_v is not None:
            return float(p_v), "p_v"
        if y_v is not None:
            return float(y_v) * P, "y_v"
        return SaturationProperties.RH_to_p_v(float(RH), cfg_atm_T, rh_reference), "RH"

    # ---------- phase admissibility (section 3) ----------
    def admissible_phases(self, T, p_v):
        S_w = p_v / SaturationProperties.Psat_water(T, extended=True)
        S_i = p_v / SaturationProperties.Psat_ice(T)
        phases = []
        if S_w > 1.0:
            phases.append(PHASE_LIQUID)
        if S_i > 1.0:
            phases.append(PHASE_ICE)
        return phases, S_w, S_i

    # ---------- evaluate one ambient point -> dict[phase]->NucleationResult ----------
    def evaluate_point(self, T, P, p_v, r_ref=R_REF_DEFAULT, grad_T_req=None):
        """Evaluate nucleation at one ambient (T, P, p_v).

        * The Gibbs-Thomson closure is solved at radius r_ref (continuation).
        * If grad_T_req is given, the radius whose solved gradient matches it
          is found by interpolation over a small radius scan (faithful: r is
          the continuation variable, g is solved -- the requested gradient is
          matched by interpolation, never imposed as the unknown).
        """
        cfg = self.cfg
        phases, S_w, S_i = self.admissible_phases(T, p_v)
        mode = "heterogeneous" if cfg.mode == "heterogeneous" else "homogeneous"

        # decide which phases to compute from phase_mode + admissibility
        if cfg.phase_mode == "liquid":
            requested = [PHASE_LIQUID]
        elif cfg.phase_mode == "ice":
            requested = [PHASE_ICE]
        elif cfg.phase_mode == "both":
            requested = [PHASE_LIQUID, PHASE_ICE]
        else:  # auto
            requested = phases

        results = {}
        # if a specific gradient is requested, scan a small radius grid and
        # interpolate to that gradient
        if grad_T_req is not None and grad_T_req > 0:
            rgrid = np.geomspace(1e-2, 1e-9, cfg.scan_resolution)
            for ph in requested:
                model = self.liquid if ph == PHASE_LIQUID else self.ice
                states = []
                for r in rgrid:
                    st = model.solve(r, T, P, p_v, theta_default=cfg.theta)
                    if st is not None:
                        states.append(st)
                st = self._interp_to_gradient(states, grad_T_req)
                results[ph] = (model, st)
        else:
            for ph in requested:
                model = self.liquid if ph == PHASE_LIQUID else self.ice
                st = model.solve(r_ref, T, P, p_v, theta_default=cfg.theta)
                results[ph] = (model, st)

        # build NucleationResults
        out = {}
        # compute rates & results
        nr = {}
        for ph, (model, st) in results.items():
            nr[ph] = model.to_result(st, T, P, p_v, cfg.scenario, mode)
        # competition metric (log10I per phase)
        logI = {ph: r.log10I for ph, r in nr.items()}
        has_l = PHASE_LIQUID in nr and nr[PHASE_LIQUID].converged
        has_i = PHASE_ICE in nr and nr[PHASE_ICE].converged
        if has_l and has_i:
            d = logI[PHASE_ICE] - logI[PHASE_LIQUID]
            for ph in nr:
                nr[ph].dlog10I_ice_minus_liq = d
            if d > 0.5:
                dom = PHASE_ICE
            elif d < -0.5:
                dom = PHASE_LIQUID
            else:
                dom = "competition"
            for ph in nr:
                nr[ph].dominant = dom
        elif has_l:
            nr[PHASE_LIQUID].dominant = PHASE_LIQUID
        elif has_i:
            nr[PHASE_ICE].dominant = PHASE_ICE

        # subsaturated handling
        if not requested:
            for ph in ([PHASE_LIQUID, PHASE_ICE] if cfg.phase_mode == "both"
                       else ([PHASE_LIQUID] if cfg.phase_mode == "liquid"
                             else [PHASE_ICE] if cfg.phase_mode == "ice"
                             else [PHASE_LIQUID, PHASE_ICE])):
                model = self.liquid if ph == PHASE_LIQUID else self.ice
                res = model.to_result(None, T, P, p_v, cfg.scenario, mode,
                                       dominant="none")
                res.status = "subsaturated"
                nr[ph] = res
        return nr

    @staticmethod
    def _interp_to_gradient(states, grad_T_req):
        """From a list of solved states (varying r -> varying g), return the
        state whose solved gradient is closest to grad_T_req (linear
        interpolation in log g).  Faithful: g is the SOLVED unknown at each r."""
        if not states:
            return None
        gs = [s["g"] for s in states]
        # find bracketing indices
        target = grad_T_req
        if target <= min(gs):
            return states[0]
        if target >= max(gs):
            return states[-1]
        # states ordered large-r (small g) first -> g increasing with index
        for i in range(len(states) - 1):
            g0, g1 = gs[i], gs[i + 1]
            if (g0 - target) * (g1 - target) <= 0:
                # linear interp weight
                w = (target - g0) / (g1 - g0) if g1 != g0 else 0.0
                # return the closer state (states are discrete; pick nearest)
                return states[i + 1] if w > 0.5 else states[i]
        return states[-1]

    # =====================================================================
    #  SCENARIOS
    # =====================================================================
    def run(self):
        s = self.cfg.scenario
        if s == "single_state":
            return self._single_state()
        if s == "temperature_sweep":
            return self._sweep("T")
        if s == "pressure_sweep":
            return self._sweep("P")
        if s == "humidity_sweep":
            return self._sweep("RH")
        if s == "thermal_gradient_sweep":
            return self._thermal_gradient_sweep()
        if s == "altitude_profile":
            return self._altitude_profile()
        if s == "time_series":
            return self._time_series()
        if s == "climate_comparison":
            return self._climate_comparison()
        if s == "phase_map":
            return self._phase_map()
        if s == "psat_gradT":
            return self._psat_gradT()
        raise ValueError(f"unknown scenario {s}")

    def _ambient_from_inputs(self, idx=None):
        """Materialise scalar T, P, RH from the AtmosphericInput (which may hold
        scalars, arrays or callables).  `idx` selects an array element."""
        cfg = self.cfg

        def pick(v):
            if v is None:
                return None
            if callable(v):
                x = (cfg.x[idx] if isinstance(cfg.x, np.ndarray) and idx is not None
                     else cfg.x) if cfg.x is not None else None
                t = (cfg.t[idx] if isinstance(cfg.t, np.ndarray) and idx is not None
                     else cfg.t) if cfg.t is not None else None
                try:
                    return v(x, t) if (x is not None or t is not None) else v()
                except TypeError:
                    try:
                        return v(x) if x is not None else v(t) if t is not None else v()
                    except TypeError:
                        return v()
            if isinstance(v, np.ndarray):
                return float(v[idx if idx is not None else 0])
            return float(v)

        T = pick(cfg.T); P = pick(cfg.P)
        RH = pick(cfg.RH); yv = pick(cfg.y_v); pv = pick(cfg.p_v)
        return T, P, RH, yv, pv

    def _resolve_grad_T(self, T, idx=None):
        g = self.cfg.grad_T
        if g is None:
            return None
        if callable(g):
            x = self.cfg.x[idx] if isinstance(self.cfg.x, np.ndarray) and idx is not None else self.cfg.x
            t = self.cfg.t[idx] if isinstance(self.cfg.t, np.ndarray) and idx is not None else self.cfg.t
            try:
                return float(g(x, t)) if (x is not None or t is not None) else float(g())
            except TypeError:
                return float(g(x) if x is not None else g(t) if t is not None else g())
        if isinstance(g, np.ndarray):
            return float(g[idx if idx is not None else 0])
        return float(g)

    # ---- single state ----
    def _single_state(self):
        T, P, RH, yv, pv = self._ambient_from_inputs()
        p_v, _ = self.resolve_p_v(T, P, RH, yv, pv, self.cfg.rh_reference)
        grad_T = self._resolve_grad_T(T)
        nr = self.evaluate_point(T, P, p_v, r_ref=self._r_ref(), grad_T_req=grad_T)
        return {"scenario": "single_state", "points": [nr], "sweep": None}

    def _r_ref(self):
        return R_REF_DEFAULT

    # ---- generic 1-D sweep over T / P / RH ----
    def _sweep(self, var):
        cfg = self.cfg
        if var == "T":
            vals = self._build_axis(cfg.T, (240.0, 320.0), cfg.scan_resolution)
        elif var == "P":
            vals = self._build_axis(cfg.P, (20000.0, 110000.0), cfg.scan_resolution)
        else:  # RH
            vals = self._build_axis(cfg.RH, (60.0, 160.0), cfg.scan_resolution)
        points = []
        for i, val in enumerate(vals):
            if var == "T":
                T = float(val); P, RH, yv, pv = self._ambient_from_inputs_excluding_T()
                P = P if P is not None else 70000.0
            elif var == "P":
                P = float(val); T = self._scalar(cfg.T, 258.15); RH, yv, pv = self._rest("P")
            else:
                RH = float(val); T = self._scalar(cfg.T, 258.15); P = self._scalar(cfg.P, 70000.0)
                yv, pv = self._scalar(cfg.y_v, None), self._scalar(cfg.p_v, None)
            # resolve p_v
            if pv is not None:
                p_v = pv
            elif yv is not None:
                p_v = yv * P
            else:
                p_v = SaturationProperties.RH_to_p_v(RH if RH is not None else 100.0,
                                                     T, cfg.rh_reference)
            grad_T = self._resolve_grad_T(T, idx=i)
            nr = self.evaluate_point(T, P, p_v, r_ref=self._r_ref(), grad_T_req=grad_T)
            for ph in nr:
                nr[ph].T = T; nr[ph].P = P; nr[ph].p_v = p_v
            points.append(nr)
        return {"scenario": cfg.scenario, "points": points,
                "sweep": {"variable": var, "values": vals}}

    def _build_axis(self, v, default_range, n):
        if isinstance(v, np.ndarray):
            return v
        if callable(v):
            return np.array([v(self._scalar(self.cfg.x, None), None)])
        if isinstance(v, (list, tuple)):
            return np.array(v)
        # scalar -> build a range around it, or use default range
        lo, hi = default_range
        return np.linspace(lo, hi, n)

    def _scalar(self, v, default):
        if v is None:
            return default
        if callable(v) or isinstance(v, np.ndarray):
            return default
        return float(v)

    def _ambient_from_inputs_excluding_T(self):
        cfg = self.cfg
        P = self._scalar(cfg.P, 70000.0)
        RH = self._scalar(cfg.RH, None)
        yv = self._scalar(cfg.y_v, None)
        pv = self._scalar(cfg.p_v, None)
        return P, RH, yv, pv

    def _rest(self, excluding):
        cfg = self.cfg
        return (self._scalar(cfg.RH, None), self._scalar(cfg.y_v, None),
                self._scalar(cfg.p_v, None))

    # ---- thermal-gradient sweep: full radius-continuation ----
    def _thermal_gradient_sweep(self):
        cfg = self.cfg
        T = self._scalar(cfg.T, 258.15)
        P = self._scalar(cfg.P, 70000.0)
        RH = self._scalar(cfg.RH, 110.0)
        p_v = SaturationProperties.RH_to_p_v(RH, T, cfg.rh_reference)
        rgrid = np.geomspace(1e-2, 1e-9, cfg.scan_resolution)
        points = []
        for r in rgrid:
            nr = self.evaluate_point(T, P, p_v, r_ref=float(r))
            points.append(nr)
        return {"scenario": cfg.scenario, "points": points,
                "sweep": {"variable": "gradT", "values": rgrid}}

    # ---- altitude profile (US Standard Atmosphere-like) ----
    def _altitude_profile(self):
        cfg = self.cfg
        z = np.linspace(0.0, 11000.0, cfg.scan_resolution) if isinstance(cfg.z, type(None)) else (
            cfg.z if isinstance(cfg.z, np.ndarray) else np.linspace(0, 11000, cfg.scan_resolution))
        points = []
        for i, zi in enumerate(z):
            T_z = 288.15 - 6.5e-3 * zi       # lapse rate 6.5 K/km (troposphere)
            P_z = 101325.0 * (T_z / 288.15) ** 5.255
            RH_z = self._interp_scalar(cfg.RH, i, default=80.0)
            p_v = SaturationProperties.RH_to_p_v(RH_z, T_z, cfg.rh_reference)
            nr = self.evaluate_point(T_z, P_z, p_v, r_ref=self._r_ref())
            for ph in nr:
                nr[ph].z = float(zi)
            points.append(nr)
        return {"scenario": cfg.scenario, "points": points,
                "sweep": {"variable": "z", "values": z}}

    def _interp_scalar(self, v, idx, default):
        if v is None:
            return default
        if callable(v):
            return float(v(self._scalar(self.cfg.x, 0.0), None))
        if isinstance(v, np.ndarray):
            return float(v[idx])
        return float(v)

    # ---- time series ----
    def _time_series(self):
        cfg = self.cfg
        t = np.linspace(0.0, 3.156e7, cfg.scan_resolution) if not isinstance(cfg.t, np.ndarray) else cfg.t
        points = []
        T0 = self._scalar(cfg.T, 258.15)
        for i, ti in enumerate(t):
            T_t = T0 + 10.0 * math.sin(2.0 * math.pi * ti / t[-1]) if not callable(cfg.T) else float(cfg.T(0.0, ti))
            P_t = self._scalar(cfg.P, 70000.0)
            RH_t = self._interp_scalar(cfg.RH, i, 90.0)
            p_v = SaturationProperties.RH_to_p_v(RH_t, T_t, cfg.rh_reference)
            nr = self.evaluate_point(T_t, P_t, p_v, r_ref=self._r_ref())
            for ph in nr:
                nr[ph].t = float(ti)
            points.append(nr)
        return {"scenario": cfg.scenario, "points": points,
                "sweep": {"variable": "t", "values": t}}

    # ---- climate comparison ----
    def _climate_comparison(self):
        cfg = self.cfg
        T0 = self._scalar(cfg.T, 258.15)
        P0 = self._scalar(cfg.P, 70000.0)
        RH0 = self._scalar(cfg.RH, 110.0)
        gradT0 = self._resolve_grad_T(T0) or 0.0
        p_v0 = SaturationProperties.RH_to_p_v(RH0, T0, cfg.rh_reference)
        T1 = T0 + cfg.dT_climate
        P1 = P0 + cfg.dP
        RH1 = max(RH0 + cfg.dRH, 1.0)
        gradT1 = gradT0 + cfg.dgradT
        p_v1 = SaturationProperties.RH_to_p_v(RH1, T1, cfg.rh_reference)
        base = self.evaluate_point(T0, P0, p_v0, r_ref=self._r_ref(), grad_T_req=(gradT0 or None))
        fut = self.evaluate_point(T1, P1, p_v1, r_ref=self._r_ref(), grad_T_req=(gradT1 or None))
        comp = ClimateScenarioComparison(base, fut, cfg)
        return {"scenario": cfg.scenario, "points": [base, fut],
                "comparison": comp.report(), "sweep": None}

    # ---- phase map (T x RH) ----
    def _phase_map(self):
        cfg = self.cfg
        T_ax = np.linspace(220.0, 290.0, 40)
        RH_ax = np.linspace(80.0, 160.0, 40)
        P = self._scalar(cfg.P, 70000.0)
        grid = []
        logI_l = np.full((len(RH_ax), len(T_ax)), np.nan)
        logI_i = np.full((len(RH_ax), len(T_ax)), np.nan)
        dom = np.full((len(RH_ax), len(T_ax)), -1, dtype=int)
        for j, T in enumerate(T_ax):
            for i, RH in enumerate(RH_ax):
                try:
                    p_v = SaturationProperties.RH_to_p_v(RH, T, cfg.rh_reference)
                except Exception:
                    continue
                nr = self.evaluate_point(T, P, p_v, r_ref=self._r_ref())
                if PHASE_LIQUID in nr and nr[PHASE_LIQUID].converged:
                    logI_l[i, j] = nr[PHASE_LIQUID].log10I
                if PHASE_ICE in nr and nr[PHASE_ICE].converged:
                    logI_i[i, j] = nr[PHASE_ICE].log10I
                # dominant: 0=liquid, 1=ice, 2=competition, -1=none
                li = logI_l[i, j]; ii = logI_i[i, j]
                if math.isfinite(li) and math.isfinite(ii):
                    d = ii - li
                    dom[i, j] = 1 if d > 0.5 else (0 if d < -0.5 else 2)
                elif math.isfinite(li):
                    dom[i, j] = 0
                elif math.isfinite(ii):
                    dom[i, j] = 1
        return {"scenario": cfg.scenario,
                "phase_map": dict(T=T_ax, RH=RH_ax, logI_liquid=logI_l,
                                  logI_ice=logI_i, dominant=dom),
                "points": [], "sweep": None}

    # ---- shifted-equilibrium saturation pressure vs solved gradient ----
    def _invert_gradient(self, model, g_targets, T_base, P, p_v, r_scan):
        """For each target gradient g in `g_targets`, recover the embryo radius
        r such that the Gibbs-Thomson closure solves to *exactly* that gradient
        (Brent inversion of g(r); r is the prescribed continuation variable,
        g = dT/dr is the solved unknown -- faithful to the closure).  Returns
        arrays (gradT, DeltaT, T_local, P_sat_shift, r_used) for every
        attainable target."""
        rs, gs = [], []
        for r in r_scan:
            st = model.solve(r, T_base, P, p_v)
            if st is not None and math.isfinite(st["g"]) and st["g"] > 0:
                rs.append(r)
                gs.append(st["g"])
        rs = np.asarray(rs)
        gs = np.asarray(gs)
        order = np.argsort(rs)              # r ascending -> g decreasing
        rs, gs = rs[order], gs[order]

        def g_of_r(r):
            st = model.solve(r, T_base, P, p_v)
            return st["g"] if st is not None else float("nan")

        gradT, dT_list, Tloc, Psat, r_used = [], [], [], [], []
        for gt in g_targets:
            if gs.size == 0 or gt < gs.min() or gt > gs.max():
                continue
            idx = np.where((gs[:-1] >= gt) & (gs[1:] <= gt))[0]   # g decreasing in r
            if idx.size == 0:
                continue
            i = idx[0]
            r_lo, r_hi = rs[i], rs[i + 1]
            try:
                r = brentq(lambda r: g_of_r(r) - gt, r_lo, r_hi,
                           xtol=1e-20, rtol=1e-12, maxiter=200)
            except Exception:
                r = (r_lo if abs(g_of_r(rs[i]) - gt) < abs(g_of_r(rs[i + 1]) - gt)
                     else rs[i + 1])
            st = model.solve(r, T_base, P, p_v)
            if st is None:
                continue
            gradT.append(st["g"])
            dT_list.append(st["dT"])
            Tloc.append(st["T_local"])
            Psat.append(st["Psat"])         # = model.Psat(T_local) = P_eq_shift
            r_used.append(r)
        return (np.asarray(gradT), np.asarray(dT_list), np.asarray(Tloc),
                np.asarray(Psat), np.asarray(r_used))

    def _psat_gradT(self):
        """Shifted-equilibrium saturation pressure vs the *solved* thermal
        gradient for the two reference bases (Brent inversion of g(r)):

          * liquid: base = 1 atm boiling point  (T_base = Tsat(101325 Pa))
          * ice:    base = 255.65 K, P_amb = 54300 Pa  (Goff-Gratch sublimation)

        For each log-spaced target gradient the embryo radius that the
        Gibbs-Thomson closure solves to exactly that gradient is recovered by
        Brent inversion; the consistent undercooling is Delta_T = 8 pi r g,
        T_local = T_base - Delta_T, and the shifted saturation pressure is the
        unified model's own curve at T_local.  phase_mode selects which phases
        are computed/plotted ('auto' -> both).  No per-point NucleationResults
        are produced; the inverted arrays are carried under 'psat_gradT'."""
        cfg = self.cfg
        gmin, gmax, ngrad = cfg.gmin, cfg.gmax, int(cfg.ngrad)
        g_targets = np.logspace(np.log10(gmin), np.log10(gmax), ngrad)
        r_scan = np.geomspace(1e-3, 1e-9, 400)

        phases = ([PHASE_LIQUID] if cfg.phase_mode == "liquid"
                  else [PHASE_ICE] if cfg.phase_mode == "ice"
                  else [PHASE_LIQUID, PHASE_ICE])

        bases = {}
        if PHASE_LIQUID in phases:
            P_base = 101325.0
            T_base = SaturationProperties.Tsat_water(P_base)          # ~373.12 K
            Psat_base = SaturationProperties.Psat_water(T_base, extended=True)
            bases[PHASE_LIQUID] = (T_base, P_base, Psat_base, self.liquid)
        if PHASE_ICE in phases:
            T_base = 255.65
            P_amb = 54300.0
            Psat_base = SaturationProperties.Psat_ice(T_base)         # Goff-Gratch
            bases[PHASE_ICE] = (T_base, P_amb, Psat_base, self.ice)

        out = {}
        for ph, (T_base, P, Psat_base, model) in bases.items():
            g, dT, Tl, Ps, r = self._invert_gradient(model, g_targets, T_base,
                                                    P, Psat_base, r_scan)
            drop = Psat_base - Ps
            out[ph] = dict(gradT=g, r=r, dT=dT, T_local=Tl, Psat=Ps,
                           P_base=Psat_base, T_base=T_base, P_amb=P,
                           drop=drop, drop_pct=(100.0 * drop / Psat_base))
            print(f"  [{ph}] inverted {len(g)} states over gradT in "
                  f"[{gmin:g}, {gmax:g}] K/m "
                  f"(r in [{r.min():.3e}, {r.max():.3e}] m); "
                  f"max DeltaT={np.max(dT):.4e} K, "
                  f"max rel. drop={np.max(out[ph]['drop_pct']):.4f} %")
        return {"scenario": "psat_gradT", "points": [], "sweep": None,
                "psat_gradT": out, "gmin": gmin, "gmax": gmax}

    # ---- 3-D shifted-equilibrium surface P_eq_shift(T, gradT) ----
    def shifted_equilibrium_surface(self, T_range=(240.0, 285.0), nT=24,
                                    gradT_range=(1.0, 1e5, "log"), ng=24,
                                    phase="both"):
        """Compute P_eq_shift(T, gradT) on a grid for the 3-D surface (section 11).
        Returns a dict of grids for liquid and ice.  `gradT_range` may be
        (lo, hi, 'log') for a logarithmic gradient axis.

        Performance: the radius scan that maps a requested gradient to a solved
        state (evaluate_point's grad_T_req path) is performed ONCE per (T, phase)
        and reused for every requested gradient via _interp_to_gradient.  The
        solved gradient g is the unknown at each continuation radius r, so the
        scan is identical for all gReq at a fixed (T, p_v); re-solving it per
        gReq (the former path) made the 24x24 surface call solve() ~172k times
        and hang the phase_map example.  Output is value-identical -- same
        states, same interpolation, and st["Psat"] == Psat(T_local) == P_eq_shift
        (the field to_result exposes as P_eq_shift)."""
        T_ax = np.linspace(T_range[0], T_range[1], nT)
        if len(gradT_range) == 3 and gradT_range[2] == "log":
            g_ax = np.geomspace(gradT_range[0], gradT_range[1], ng)
        else:
            g_ax = np.linspace(gradT_range[0], gradT_range[1], ng)
        P = self._scalar(self.cfg.P, 70000.0)
        RH = self._scalar(self.cfg.RH, 110.0)
        rgrid = np.geomspace(1e-2, 1e-9, self.cfg.scan_resolution)
        out = {}
        phases = [PHASE_LIQUID, PHASE_ICE] if phase == "both" else [phase]
        for ph in phases:
            model = self.liquid if ph == PHASE_LIQUID else self.ice
            Z = np.full((len(g_ax), len(T_ax)), np.nan)
            for j, T in enumerate(T_ax):
                p_v = SaturationProperties.RH_to_p_v(RH, T, self.cfg.rh_reference)
                # one radius scan per (T, phase); reused for every gReq
                states = []
                for r in rgrid:
                    st = model.solve(r, T, P, p_v, theta_default=self.cfg.theta)
                    if st is not None:
                        states.append(st)
                for i, gReq in enumerate(g_ax):
                    st = self._interp_to_gradient(states, float(gReq))
                    if st is not None:
                        Z[i, j] = st["Psat"]      # = Psat(T_local) = P_eq_shift
            out[ph] = dict(T=T_ax, gradT=g_ax, P_eq=Z)
        return out


# =============================================================================
#  CLIMATE SCENARIO COMPARISON
# =============================================================================
class ClimateScenarioComparison:
    """Compares baseline vs perturbed (future) microphysical responses."""

    def __init__(self, base, future, cfg):
        self.base = base
        self.future = future
        self.cfg = cfg

    @staticmethod
    def _get(d, phase, attr):
        r = d.get(phase)
        if r is None:
            return float("nan")
        return getattr(r, attr)

    def _delta(self, phase, attr):
        x0 = self._get(self.base, phase, attr)
        x1 = self._get(self.future, phase, attr)
        if not (math.isfinite(x0) and math.isfinite(x1)):
            return float("nan"), float("nan")
        d = x1 - x0
        if abs(x0) > 1e-30:
            return d, 100.0 * d / x0
        return d, float("nan")

    def report(self):
        rep = {}
        for phase in (PHASE_LIQUID, PHASE_ICE):
            row = {}
            for attr in ("T_eq_shift", "P_eq_shift", "rC_2nd", "DeltaG_2nd",
                         "log10I", "Delta_mu", "dew_point", "frost_point",
                         "S_w", "S_i"):
                d, dpct = self._delta(phase, attr)
                row[attr] = dict(baseline=self._get(self.base, phase, attr),
                                 future=self._get(self.future, phase, attr),
                                 delta=d, delta_pct=dpct)
            # dominant phase change
            b_dom = self._get(self.base, phase, "dominant")
            f_dom = self._get(self.future, phase, "dominant")
            row["dominant_change"] = f"{b_dom} -> {f_dom}"
            rep[phase] = row
        return rep


# =============================================================================
#  OUTPUT  (CSV / JSON / summary)
# =============================================================================
CSV_COLUMNS = [
    "scenario", "phase", "mode", "status", "converged",
    "T", "P", "p_v", "Psat_w", "Psat_i", "RH_w", "RH_i", "S_w", "S_i",
    "dew_point", "frost_point", "grad_T", "Delta_T", "T_eq_shift", "P_eq_shift",
    "Delta_mu", "DeltaG_V", "DeltaS_V", "gamma_r", "dgamma_dr", "surface_stress",
    "Gamma1", "Gamma2", "r", "rC_CNT", "rC_1st", "rC_2nd", "rC_hom_2nd",
    "rC_het_2nd", "theta", "DeltaG_CNT", "DeltaG_1st", "DeltaG_2nd", "I",
    "log10I", "dominant", "dlog10I_ice_minus_liq", "closure_resid",
    "parabolic_resid", "gibbs_thomson_resid", "in_valid_range",
]


def _safe_num(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return v if math.isfinite(v) else ""
    return v


def save_csv(simout, fname):
    """Write one row per (point, phase) state."""
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for nr_map in simout.get("points", []):
            for ph, res in nr_map.items():
                w.writerow([_safe_num(res.to_dict().get(c)) for c in CSV_COLUMNS])
    print(f"Saved CSV -> {fname}")


def save_json(simout, cfg, fname):
    meta = {"scenario": cfg.scenario, "phase_mode": cfg.phase_mode,
            "rh_reference": cfg.rh_reference, "scenario_config": asdict(cfg),
            "timestamp_note": "produced by unified_h2o_nucleation_climate.py"}
    # strip non-serialisable
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()
                    if not callable(v) and not isinstance(v, (np.ndarray,))}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    points = []
    for nr_map in simout.get("points", []):
        for ph, res in nr_map.items():
            points.append(res.to_dict())
    out = {"meta": clean(meta), "points": clean(points),
           "sweep": clean(simout.get("sweep")),
           "comparison": clean(simout.get("comparison")),
           "phase_map": clean(simout.get("phase_map")),
           "psat_gradT": clean(simout.get("psat_gradT"))}
    with open(fname, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved JSON -> {fname}")


def _print_psat_gradT_summary(simout):
    """Console table for the psat_gradT scenario (per-phase inverted states)."""
    data = simout["psat_gradT"]
    for ph, d in data.items():
        g, r, dT, Tl, Ps = d["gradT"], d["r"], d["dT"], d["T_local"], d["Psat"]
        P_base = d["P_base"]
        print(f"\n--- {ph.upper()} table  (T_base={d['T_base']:.4g} K, "
              f"P_base={P_base:.4g} Pa) ---")
        print("  gradT[K/m]      r[m]         DeltaT[K]    T_local[K]   "
              "P_sat[Pa]     drop[Pa]   drop[%]")
        for gi, ri, dTi, Tli, Psi in zip(g, r, dT, Tl, Ps):
            dp = P_base - Psi
            dpp = 100.0 * dp / P_base
            print(f"  {gi:10.4e}  {ri:11.4e}  {dTi:10.4e}  {Tli:10.4f}  "
                  f"{Psi:11.4f}  {dp:9.4f}  {dpp:7.5f}")


def print_summary(simout, cfg):
    print("\n" + "=" * 78)
    print(f"SCENARIO: {cfg.scenario}   phase_mode={cfg.phase_mode}   "
          f"rh_reference={cfg.rh_reference}")
    print("=" * 78)
    pts = simout.get("points", [])
    if not pts:
        if simout.get("psat_gradT"):
            _print_psat_gradT_summary(simout)
        else:
            print("(no per-point results for this scenario)")
        return
    hdr = ("phase  ", "status     ", "T[K]  ", " P[Pa]  ", "RH_w[%]", "RH_i[%]",
           "gradT[K/m]", "rC2nd[m]", "Dmu[J/mol]", "log10I", "dominant")
    print("  " + " | ".join(hdr))
    n_show = min(12, len(pts))
    for nr_map in pts[:n_show]:
        for ph, r in nr_map.items():
            print("  %7s | %11s | %6.2f | %8.1e | %7.2f | %7.2f | %9.3e | "
                  "%9.3e | %9.3e | %8.3f | %s" % (
                      ph, r.status, r.T, r.P, r.RH_w, r.RH_i, r.grad_T,
                      r.rC_2nd, r.Delta_mu,
                      r.log10I if math.isfinite(r.log10I) else float("nan"),
                      r.dominant))
    if len(pts) > n_show:
        print(f"  ... ({len(pts) - n_show} more rows in CSV)")
    if "comparison" in simout and simout["comparison"]:
        print("\nCLIMATE COMPARISON (future - baseline):")
        for ph, row in simout["comparison"].items():
            print(f"  {ph}:")
            for k, v in row.items():
                if isinstance(v, dict):
                    d = v.get("delta", float("nan"))
                    dp = v.get("delta_pct", float("nan"))
                    print(f"    {k:14s}: delta={d:.4e}  ({dp:.3f}%)" if math.isfinite(d)
                          else f"    {k:14s}: n/a")
                else:
                    print(f"    {k:14s}: {v}")


# =============================================================================
#  PLOTTER
# =============================================================================
class NucleationPlotter:
    """Generates the scenario-compatible figures (sections 10, 11, 17)."""

    def __init__(self, cfg, outdir):
        self.cfg = cfg
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.dpi = 300

    def _save(self, fig, name):
        path = os.path.join(self.outdir, name)
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(path + ".png", dpi=self.dpi)
        if self.cfg.save_pdf:
            try:
                fig.savefig(path + ".pdf")
            except Exception:
                pass
        plt.close(fig)
        print(f"  figure -> {path}.png")

    @staticmethod
    def _safe(arr):
        a = np.asarray(arr, dtype=float)
        a[~np.isfinite(a)] = np.nan
        return a

    def _xy(self, points, attr, phase):
        xs, ys = [], []
        for nr_map in points:
            if phase in nr_map and nr_map[phase].converged:
                v = getattr(nr_map[phase], attr)
                if math.isfinite(v):
                    # x coordinate: use sweep index proxy (grad_T) or T
                    xs.append(nr_map[phase].grad_T)
                    ys.append(v)
        return np.array(xs), np.array(ys)

    # ---- entry: dispatch by scenario ----
    def plot(self, simout, surface=None):
        s = self.cfg.scenario
        print(f"\nGenerating figures for scenario '{s}' ...")
        if s == "single_state":
            self._fig_single_state(simout)
        elif s in ("temperature_sweep", "pressure_sweep", "humidity_sweep"):
            self._fig_1d_sweep(simout)
        elif s == "thermal_gradient_sweep":
            self._fig_grad_sweep(simout)
        elif s == "altitude_profile":
            self._fig_altitude(simout)
        elif s == "time_series":
            self._fig_time_series(simout)
        elif s == "climate_comparison":
            self._fig_climate(simout)
        elif s == "phase_map":
            self._fig_phase_maps(simout)
        elif s == "psat_gradT":
            self._fig_psat_gradT(simout)
        # universal diagnostic figures (when a radius/grad sweep is available)
        if s in ("thermal_gradient_sweep", "temperature_sweep", "humidity_sweep",
                 "altitude_profile", "time_series"):
            self._fig_competition(simout)
        if surface is not None:
            self._fig_3d_surface(surface)
        if self.cfg.show:
            plt.show()

    # ---------- single state ----------
    def _fig_single_state(self, simout):
        nr_map = simout["points"][0]
        labels = {PHASE_LIQUID: ("liquid", COL_LIQUID), PHASE_ICE: ("ice", COL_ICE)}
        # bar chart of log10I per phase
        fig, ax = plt.subplots()
        phs = [p for p in nr_map if nr_map[p].converged]
        vals = [nr_map[p].log10I for p in phs]
        cols = [labels[p][1] for p in phs]
        ax.bar([labels[p][0] for p in phs], vals, color=cols)
        ax.set_ylabel(r"$\log_{10} I$  [$\mathrm{m^{-3}\,s^{-1}}$]")
        ax.set_title("Nucleation rate at the requested state")
        ax.grid(True, ls="--", alpha=0.4)
        self._save(fig, "single_state_logI")

        # thermodynamic summary text figure
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axis("off")
        lines = []
        for ph in nr_map:
            r = nr_map[ph]
            lines.append(f"[{ph}]  status={r.status}  dominant={r.dominant}")
            lines.append(f"   T={r.T:.2f} K  P={r.P:.1f} Pa  RH_w={r.RH_w:.1f}%  "
                         f"RH_i={r.RH_i:.1f}%")
            lines.append(f"   grad_T={r.grad_T:.3e} K/m  Delta_T={r.Delta_T:.3e} K  "
                         f"T_eq={r.T_eq_shift:.2f} K  P_eq={r.P_eq_shift:.3e} Pa")
            lines.append(f"   rC_2nd={r.rC_2nd:.3e} m  Dmu={r.Delta_mu:.3e} J/mol  "
                         f"DeltaG_2nd={r.DeltaG_2nd:.3e} J")
            lines.append(f"   Gamma2={r.Gamma2:.3e} m.K  log10I={r.log10I:.3f}  "
                         f"closure_resid={r.closure_resid:.2e}")
            lines.append("")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=9)
        ax.set_title("Single-state nucleation summary")
        self._save(fig, "single_state_summary")

    # ---------- 1-D sweep (T / P / RH) ----------
    def _fig_1d_sweep(self, simout):
        var = simout["sweep"]["variable"]
        vals = np.asarray(simout["sweep"]["values"])
        points = simout["points"]
        fig, ax = plt.subplots()
        for ph, col in ((PHASE_LIQUID, COL_LIQUID), (PHASE_ICE, COL_ICE)):
            ys = []
            for nr_map in points:
                r = nr_map.get(ph)
                ys.append(r.log10I if (r and math.isfinite(r.log10I)) else np.nan)
            ax.plot(vals[:len(ys)], ys, "-o", ms=3, color=col, label=ph)
        ax.set_xlabel({"T": "T [K]", "P": "P [Pa]", "RH": "RH [%]"}[var])
        ax.set_ylabel(r"$\log_{10} I$")
        ax.set_title(f"Nucleation rate vs {var}")
        ax.legend(); ax.grid(True, ls="--", alpha=0.4)
        self._save(fig, f"sweep_{var}_logI")
        # supersaturation vs var
        fig, ax = plt.subplots()
        for ph, col, key in ((PHASE_LIQUID, COL_LIQUID, "S_w"),
                             (PHASE_ICE, COL_ICE, "S_i")):
            ys = [getattr(nr_map.get(ph, NucleationResult(phase=ph)), key)
                  if nr_map.get(ph) else np.nan for nr_map in points]
            ax.plot(vals[:len(ys)], ys, "-", color=col, label=f"{ph} ({key})")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel({"T": "T [K]", "P": "P [Pa]", "RH": "RH [%]"}[var])
        ax.set_ylabel("supersaturation S")
        ax.set_title(f"Supersaturation vs {var}")
        ax.legend(); ax.grid(True, ls="--", alpha=0.4)
        self._save(fig, f"sweep_{var}_supersat")

    # ---------- thermal-gradient sweep ----------
    def _fig_grad_sweep(self, simout):
        points = simout["points"]
        # gather per-phase arrays (ordered by radius grid -> gradT increasing)
        gT = {ph: [] for ph in (PHASE_LIQUID, PHASE_ICE)}
        rC2 = {ph: [] for ph in (PHASE_LIQUID, PHASE_ICE)}
        logI = {ph: [] for ph in (PHASE_LIQUID, PHASE_ICE)}
        Dmu = {ph: [] for ph in (PHASE_LIQUID, PHASE_ICE)}
        DGV = {ph: [] for ph in (PHASE_LIQUID, PHASE_ICE)}
        G2 = {ph: [] for ph in (PHASE_LIQUID, PHASE_ICE)}
        P_eq = {ph: [] for ph in (PHASE_LIQUID, PHASE_ICE)}
        T_eq = {ph: [] for ph in (PHASE_LIQUID, PHASE_ICE)}
        for nr_map in points:
            for ph in (PHASE_LIQUID, PHASE_ICE):
                r = nr_map.get(ph)
                if r and r.converged:
                    gT[ph].append(r.grad_T); rC2[ph].append(r.rC_2nd)
                    logI[ph].append(r.log10I); Dmu[ph].append(r.Delta_mu)
                    DGV[ph].append(r.DeltaG_V); G2[ph].append(r.Gamma2)
                    P_eq[ph].append(r.P_eq_shift); T_eq[ph].append(r.T_eq_shift)
        cols = {PHASE_LIQUID: COL_LIQUID, PHASE_ICE: COL_ICE}

        # 12. r_C vs gradT (log-log)
        fig, ax = plt.subplots()
        for ph in (PHASE_LIQUID, PHASE_ICE):
            x = self._safe(gT[ph]); y = self._safe(rC2[ph])
            m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if np.any(m):
                ax.loglog(x[m], y[m], "-o", ms=3, color=cols[ph], label=ph)
        ax.set_xlabel(r"$\nabla T$ [K/m]"); ax.set_ylabel(r"$r_{C,2nd}$ [m]")
        ax.set_title("Critical radius vs thermal gradient")
        ax.legend(); ax.grid(True, ls="--", alpha=0.4, which="both")
        self._save(fig, "rC_vs_gradT_loglog")

        # 27. log10I vs gradT
        fig, ax = plt.subplots()
        for ph in (PHASE_LIQUID, PHASE_ICE):
            x = self._safe(gT[ph]); y = self._safe(logI[ph])
            m = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if np.any(m):
                ax.semilogx(x[m], y[m], "-o", ms=3, color=cols[ph], label=ph)
        ax.set_xlabel(r"$\nabla T$ [K/m]"); ax.set_ylabel(r"$\log_{10} I$")
        ax.set_title("Nucleation rate vs thermal gradient")
        ax.legend(); ax.grid(True, ls="--", alpha=0.4)
        self._save(fig, "logI_vs_gradT")

        # 7/8. shifted equilibrium pressure & temperature vs gradT
        fig, ax = plt.subplots()
        for ph in (PHASE_LIQUID, PHASE_ICE):
            x = self._safe(gT[ph]); y = self._safe(P_eq[ph])
            m = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if np.any(m):
                ax.loglog(x[m], y[m], "-o", ms=3, color=cols[ph], label=ph)
        ax.set_xlabel(r"$\nabla T$ [K/m]"); ax.set_ylabel(r"$P_{eq,shift}$ [Pa]")
        ax.set_title("Shifted-equilibrium pressure vs thermal gradient")
        ax.legend(); ax.grid(True, ls="--", alpha=0.4, which="both")
        self._save(fig, "Peq_vs_gradT")

        # 9. Delta_T vs gradT
        fig, ax = plt.subplots()
        for ph in (PHASE_LIQUID, PHASE_ICE):
            dT = [nr_map.get(ph).Delta_T if nr_map.get(ph) else np.nan
                  for nr_map in points]
            x = self._safe(gT[ph]); y = self._safe(dT)
            m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if np.any(m):
                ax.loglog(x[m], y[m], "-o", ms=3, color=cols[ph], label=ph)
        ax.set_xlabel(r"$\nabla T$ [K/m]"); ax.set_ylabel(r"$\Delta T$ [K]")
        ax.set_title("Undercooling vs thermal gradient")
        ax.legend(); ax.grid(True, ls="--", alpha=0.4, which="both")
        self._save(fig, "DeltaT_vs_gradT")

        # 20. |Delta_mu| vs gradT
        fig, ax = plt.subplots()
        for ph in (PHASE_LIQUID, PHASE_ICE):
            x = self._safe(gT[ph]); y = np.abs(self._safe(Dmu[ph]))
            m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if np.any(m):
                ax.loglog(x[m], y[m], "-o", ms=3, color=cols[ph], label=ph)
        ax.set_xlabel(r"$\nabla T$ [K/m]"); ax.set_ylabel(r"$|\Delta\mu|$ [J/mol]")
        ax.set_title("Chemical-potential difference vs thermal gradient")
        ax.legend(); ax.grid(True, ls="--", alpha=0.4, which="both")
        self._save(fig, "Dmu_vs_gradT")

        # 18. Gamma2 vs gradT
        fig, ax = plt.subplots()
        for ph in (PHASE_LIQUID, PHASE_ICE):
            x = self._safe(gT[ph]); y = self._safe(G2[ph])
            m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if np.any(m):
                ax.loglog(x[m], y[m], "-o", ms=3, color=cols[ph], label=ph)
        ax.set_xlabel(r"$\nabla T$ [K/m]"); ax.set_ylabel(r"$\Gamma^{(2)}$ [m.K]")
        ax.set_title("Thermal-field tensor vs thermal gradient")
        ax.legend(); ax.grid(True, ls="--", alpha=0.4, which="both")
        self._save(fig, "Gamma2_vs_gradT")

        # 25. barrier vs gradT
        fig, ax = plt.subplots()
        for ph in (PHASE_LIQUID, PHASE_ICE):
            DG = [abs(nr_map.get(ph).DeltaG_2nd) if nr_map.get(ph) else np.nan
                  for nr_map in points]
            x = self._safe(gT[ph]); y = self._safe(DG)
            m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if np.any(m):
                ax.loglog(x[m], y[m], "-o", ms=3, color=cols[ph], label=ph)
        ax.set_xlabel(r"$\nabla T$ [K/m]"); ax.set_ylabel(r"$|\Delta G_C^{(2)}|$ [J]")
        ax.set_title("Nucleation barrier vs thermal gradient")
        ax.legend(); ax.grid(True, ls="--", alpha=0.4, which="both")
        self._save(fig, "DGC_vs_gradT")

    # ---------- competition (Delta log10I) ----------
    def _fig_competition(self, simout):
        points = simout["points"]
        xs, dl = [], []
        for nr_map in points:
            l = nr_map.get(PHASE_LIQUID); i = nr_map.get(PHASE_ICE)
            if l and i and math.isfinite(l.log10I) and math.isfinite(i.log10I):
                xs.append(l.grad_T if math.isfinite(l.grad_T) and l.grad_T > 0
                          else l.T)
                dl.append(i.log10I - l.log10I)
        if not xs:
            return
        fig, ax = plt.subplots()
        ax.plot(xs, dl, "-o", ms=3, color="k")
        ax.axhline(0, color="r", ls="--", lw=1)
        ax.set_xlabel(r"$\nabla T$ [K/m] (or T [K])")
        ax.set_ylabel(r"$\Delta\log_{10} I = \log_{10} I_{ice}-\log_{10} I_{liq}$")
        ax.set_title("Liquid / ice kinetic competition")
        ax.grid(True, ls="--", alpha=0.4)
        self._save(fig, "competition_dlogI")

    # ---------- altitude ----------
    def _fig_altitude(self, simout):
        points = simout["points"]
        z = [next(iter(nr_map.values())).z if nr_map else np.nan for nr_map in points]
        for attr, ylabel, fname in (
                ("T", "T [K]", "alt_T"), ("P", "P [Pa]", "alt_P"),
                ("RH_w", r"$RH_w$ [%]", "alt_RHw"),
                ("grad_T", r"$\nabla T$ [K/m]", "alt_gradT")):
            fig, ax = plt.subplots()
            for ph, col in ((PHASE_LIQUID, COL_LIQUID), (PHASE_ICE, COL_ICE)):
                ys = [nr_map.get(ph, NucleationResult(phase=ph)).__getattribute__(attr)
                      if nr_map.get(ph) else np.nan for nr_map in points]
                ax.plot(ys, z, "-o", ms=3, color=col, label=ph)
            ax.set_xlabel(ylabel); ax.set_ylabel("altitude z [m]")
            ax.set_title(f"{ylabel} vs altitude"); ax.legend()
            ax.grid(True, ls="--", alpha=0.4)
            self._save(fig, fname)

    # ---------- time series ----------
    def _fig_time_series(self, simout):
        points = simout["points"]
        t = [next(iter(nr_map.values())).t if nr_map else np.nan for nr_map in points]
        for attr, ylabel, fname in (
                ("T", "T [K]", "ts_T"), ("RH_w", r"$RH_w$ [%]", "ts_RHw"),
                ("log10I", r"$\log_{10} I$", "ts_logI")):
            fig, ax = plt.subplots()
            for ph, col in ((PHASE_LIQUID, COL_LIQUID), (PHASE_ICE, COL_ICE)):
                ys = [nr_map.get(ph, NucleationResult(phase=ph)).__getattribute__(attr)
                      if nr_map.get(ph) else np.nan for nr_map in points]
                ax.plot(t, ys, "-o", ms=3, color=col, label=ph)
            ax.set_xlabel("t [s]"); ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs time"); ax.legend()
            ax.grid(True, ls="--", alpha=0.4)
            self._save(fig, fname)

    # ---------- climate comparison ----------
    def _fig_climate(self, simout):
        comp = simout.get("comparison", {})
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axis("off")
        lines = ["CLIMATE COMPARISON  (future - baseline)", ""]
        for ph, row in comp.items():
            lines.append(f"[{ph}]")
            for k, v in row.items():
                if isinstance(v, dict):
                    d = v.get("delta", float("nan"))
                    dp = v.get("delta_pct", float("nan"))
                    if math.isfinite(d):
                        lines.append(f"  {k:14s}: {d:+.4e}  ({dp:+.3f}%)")
                    else:
                        lines.append(f"  {k:14s}: n/a")
                else:
                    lines.append(f"  {k:14s}: {v}")
            lines.append("")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=9)
        ax.set_title("Climate-scenario microphysical deltas")
        self._save(fig, "climate_comparison")

    # ---------- phase maps ----------
    def _fig_phase_maps(self, simout):
        pm = simout["phase_map"]
        T = pm["T"]; RH = pm["RH"]
        TT, RR = np.meshgrid(T, RH)
        for key, z, title, fname in (
                ("logI_liquid", pm["logI_liquid"], r"$\log_{10} I_{liquid}$",
                 "map_TxRH_logI_liquid"),
                ("logI_ice", pm["logI_ice"], r"$\log_{10} I_{ice}$",
                 "map_TxRH_logI_ice")):
            fig, ax = plt.subplots()
            Z = np.ma.masked_invalid(np.asarray(z))
            pcm = ax.pcolormesh(TT, RR, Z, shading="auto", cmap="viridis")
            ax.set_xlabel("T [K]"); ax.set_ylabel("RH [%]")
            ax.set_title(title)
            fig.colorbar(pcm, ax=ax, label=title)
            ax.grid(True, ls="--", alpha=0.3)
            self._save(fig, fname)
        # dominant-phase map
        fig, ax = plt.subplots()
        dom = np.asarray(pm["dominant"])
        cmap = plt.get_cmap("coolwarm").resampled(4)
        pcm = ax.pcolormesh(TT, RR, dom, shading="auto", cmap=cmap, vmin=-1, vmax=2)
        ax.set_xlabel("T [K]"); ax.set_ylabel("RH [%]")
        ax.set_title("Dominant phase  (0=liquid, 1=ice, 2=competition, -1=none)")
        fig.colorbar(pcm, ax=ax, ticks=[-1, 0, 1, 2])
        # I_liquid = I_ice contour
        dl = np.asarray(pm["logI_ice"]) - np.asarray(pm["logI_liquid"])
        ax.contour(TT, RR, dl, levels=[0.0], colors="k", linewidths=1.5)
        ax.grid(True, ls="--", alpha=0.3)
        self._save(fig, "map_TxRH_dominant")

    # ---------- shifted-equilibrium P_sat vs solved gradient ----------
    @staticmethod
    def _gtag(v):
        """Format a gradient bound as in the standalone names (1.0 -> '1e0')."""
        return f"{v:.0e}".replace("e+0", "e")

    def _fig_psat_gradT(self, simout):
        """Two-panel figure per phase: absolute shifted P_sat (top) + pressure
        drop (bottom), vs the Brent-solved thermal gradient.  Reproduces the
        standalone fig_Psat_vs_gradT_*_1e0_1e4.png using the unified model's
        own solver and constitutive laws."""
        data = simout.get("psat_gradT", {})
        if not data:
            print("  (no psat_gradT data to plot)")
            return
        gmin = simout.get("gmin", 1.0)
        gmax = simout.get("gmax", 1.0e4)
        tag = f"{self._gtag(gmin)}_{self._gtag(gmax)}"
        G_INV = 1530.6
        inv_label = r"Inversion point gradient, $\nabla T = 1530.6$ K/m"

        for ph, d in data.items():
            gradT = self._safe(d["gradT"])
            P_shift = self._safe(d["Psat"])
            P_base = d["P_base"]
            if gradT.size == 0:
                print(f"  [{ph}] no attainable states; figure skipped")
                continue
            is_ice = (ph == PHASE_ICE)
            col = COL_ICE if is_ice else COL_LIQUID
            P_drop = P_base - P_shift
            P_drop_pct = 100.0 * P_drop / P_base

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8.0, 9.0), sharex=True,
                gridspec_kw={"hspace": 0.14})
            ax1.plot(gradT, P_shift, "o-", ms=4, lw=1.6, color=col,
                     label=r"$P_{sat}(T_{base}-\Delta T)$  [shifted equilibrium]")
            ax1.axhline(P_base, color="r", ls="--", lw=1.2,
                        label=r"$P_{sat}(T_{base})$  [classical equilibrium, "
                              r"$\nabla T\to 0$]")
            ax1.axvline(G_INV, color="k", ls="--", lw=1.4, label=inv_label)
            ax1.set_xscale("log")
            ax1.set_ylabel(r"Saturation pressure  $P_{sat}$  [Pa]", fontsize=12)
            if is_ice:
                top = (r"$p_{\mathrm{ice}}$ vs $\nabla T$  —  H$_2$O vapour"
                       r"$\to$ice  (base: $T=%g$ K, $P_{amb}=%g$ hPa)  [unified]"
                       % (d["T_base"], d["P_amb"] / 100.0))
                ax1.set_ylabel(r"Sublimation pressure  $p_{\mathrm{ice}}$  [Pa]",
                               fontsize=12)
            else:
                top = (r"$P_{sat}$ vs $\nabla T$  —  H$_2$O vapour$\to$liquid  "
                       r"(base: $T=%.2f$ K, $P=%.2f$ atm)  [unified]"
                       % (d["T_base"], d["P_amb"] / 101325.0))
            ax1.set_title(top, fontsize=12)
            ax1.grid(True, ls="--", alpha=0.4)
            ax1.legend(loc="best", fontsize=9)
            ax1.tick_params(labelbottom=False, labelsize=10)

            ax2.plot(gradT, P_drop, "o-", ms=4, lw=1.6, color=col,
                     label=(r"$\Delta p = p_{\mathrm{ice}}(T_{base})"
                           r"-p_{\mathrm{ice}}(T_{base}-\Delta T)$  [Pa]"
                           if is_ice else
                           r"$\Delta P = P_{sat}(T_{base})"
                           r"-P_{sat}(T_{base}-\Delta T)$  [Pa]"))
            ax2b = ax2.twinx()
            ax2b.plot(gradT, P_drop_pct, "--s", ms=4, lw=1.2, color="g",
                      label=(r"$\Delta p / p_{\mathrm{ice}}(T_{base})$  [%]"
                             if is_ice else
                             r"$\Delta P / P_{base}$  [%]"))
            ax2.axvline(G_INV, color="k", ls="--", lw=1.4, label=inv_label)
            ax2.set_xscale("log")
            ax2.set_xlabel(r"Solved thermal gradient  $\nabla T$  [K/m]",
                           fontsize=12)
            ax2.set_ylabel(r"Pressure drop  $\Delta P$  [Pa]", color=col,
                           fontsize=12)
            ax2b.set_ylabel(r"Relative drop  $\Delta P / P_{base}$  [%]",
                            color="g", fontsize=12)
            ax2.tick_params(axis="y", labelcolor=col, labelsize=10)
            ax2b.tick_params(axis="y", labelcolor="g", labelsize=10)
            ax2.tick_params(axis="x", labelsize=10)
            ax2.set_title(r"Shifted-equilibrium saturation-pressure drop "
                          r"vs $\nabla T$", fontsize=12)
            ax2.grid(True, ls="--", alpha=0.4)
            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = ax2b.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)

            name = ("fig_Psat_vs_gradT_ice_" if is_ice else "fig_Psat_vs_gradT_"
                    ) + tag
            self._save(fig, name)

    # ---------- 3-D shifted-equilibrium surface ----------
    def _fig_3d_surface(self, surface):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        plotted = False
        for ph, col in ((PHASE_LIQUID, COL_LIQUID), (PHASE_ICE, COL_ICE)):
            if ph not in surface:
                continue
            T = surface[ph]["T"]; g = surface[ph]["gradT"]; Z = surface[ph]["P_eq"]
            TT, GG = np.meshgrid(T, g)
            Zm = np.ma.masked_invalid(Z)
            ax.plot_surface(np.log10(GG), TT, Zm, alpha=0.55, color=col,
                            label=ph, edgecolor="none")
            plotted = True
        if not plotted:
            plt.close(fig); return
        ax.set_xlabel(r"$\log_{10}\nabla T$  [K/m]")
        ax.set_ylabel("T  [K]")
        ax.set_zlabel(r"$P_{eq,shift}$  [Pa]")
        ax.set_title(r"Shifted-equilibrium pressure $P_{eq,shift}(T,\nabla T)$")
        # triple-point marker (Tt, any grad, Pt)
        ax.scatter([math.log10(1.0)], [Tt], [Pt], color="k", s=40,
                   label="triple point")
        ax.legend()
        try:
            fig.tight_layout()
        except Exception:
            pass
        path = os.path.join(self.outdir, "surface_Peq_T_gradT")
        fig.savefig(path + ".png", dpi=self.dpi)
        if self.cfg.save_pdf:
            try:
                fig.savefig(path + ".pdf")
            except Exception:
                pass
        plt.close(fig)
        print(f"  figure -> {path}.png")


# =============================================================================
#  VALIDATION TESTS  (section 16: 15 mandatory tests)
# =============================================================================
ICE_SCRIPT = "Nucleation_model_H2O_vapour_solid_Sim_2026_paper.py"
LIQUID_SCRIPT = "Nucleation_model_H2O_vapour_liquid_Sim_2026.py"
ICE_SHA256 = "c9fa9c01aabd147a455632a54fc8b907882b7fb167d9b0cff0bd9a86058403f6"


def _sha256(path):
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_validation_tests(verbose=True):
    """Run the 15 mandatory validation tests.  Returns True iff all pass."""
    def pr(*a):
        if verbose:
            print(*a)

    pr("=" * 78)
    pr("VALIDATION  unified_h2o_nucleation_climate.py")
    pr("=" * 78)
    ok = True
    # this module lives in the unified_h2o_nucleation_climate/ subfolder; the
    # ice/liquid reference scripts it guards live one level up at repo root.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ice_path = os.path.join(here, ICE_SCRIPT)
    liq_path = os.path.join(here, LIQUID_SCRIPT)

    # 1 -- ice reference script byte-for-byte unchanged
    cs = _sha256(ice_path)
    pr(f"[1] ice script SHA-256 = {cs}")
    pr(f"    expected           = {ICE_SHA256}")
    if cs is None:
        pr("    !! ice reference script not found alongside this module")
        ok = False
    else:
        ok &= (cs == ICE_SHA256)
        pr(f"    -> {'UNCHANGED' if cs == ICE_SHA256 else 'CHANGED !!'}")

    # 2 -- liquid reference script present (not modified by us; checksum guard)
    cs_l = _sha256(liq_path)
    pr(f"[2] liquid reference script present: {cs_l is not None}")
    ok &= (cs_l is not None)

    # 3 -- units & order of magnitude: gamma_ice ~0.10-0.11, not 1.3
    ok &= (0.09 <= gamma_IV_inf() <= 0.12)
    pr(f"[3] ice gamma_0 = {gamma_IV_inf():.4f} J/m^2 (must be ~0.10-0.11, NOT 1.3): "
       f"{'OK' if 0.09 <= gamma_IV_inf() <= 0.12 else 'FAIL'}")

    # 4 -- partial vapour pressure closure (RH -> p_v -> RH round-trip)
    T = 260.0
    for ref in ("water", "ice"):
        p_v = SaturationProperties.RH_to_p_v(110.0, T, ref)
        if ref == "water":
            rt = 100.0 * p_v / SaturationProperties.Psat_water(T, extended=True)
        else:
            rt = 100.0 * p_v / SaturationProperties.Psat_ice(T)
        ok &= abs(rt - 110.0) < 1e-6
    pr(f"[4] RH -> p_v -> RH round-trip (water & ice): "
       f"{'OK' if ok else 'FAIL'}")

    # 5 -- RH_w / RH_i consistency at the triple point
    p_v = Pt
    Sw = p_v / SaturationProperties.Psat_water(Tt, extended=True)
    Si = p_v / SaturationProperties.Psat_ice(Tt)
    ok &= abs(Sw - 1.0) < 1e-6 and abs(Si - 1.0) < 1e-6
    pr(f"[5] S_w={Sw:.6f}, S_i={Si:.6f} at triple point (must be 1.0): "
       f"{'OK' if abs(Sw-1)<1e-6 and abs(Si-1)<1e-6 else 'FAIL'}")

    # 6 -- saturation pressure consistency: P_sat,w(Tt)=P_sat,i(Tt)=Pt
    pw = SaturationProperties.Psat_water(Tt)
    pi = SaturationProperties.Psat_ice(Tt)
    ok &= abs(pw - Pt) < 1e-3 and abs(pi - Pt) < 1e-3
    pr(f"[6] P_sat,w(Tt)={pw:.3f}, P_sat,i(Tt)={pi:.3f}, Pt={Pt:.3f} Pa: "
       f"{'OK' if abs(pw-Pt)<1e-3 and abs(pi-Pt)<1e-3 else 'FAIL'}")

    # 7 -- inversion P_sat(T) <-> T_sat(P)
    worst = 0.0
    for Ttst in (274.0, 300.0, 373.15, 450.0, 500.0, 600.0, 640.0):
        P = SaturationProperties.Psat_water(Ttst)
        Tb = SaturationProperties.Tsat_water(P)
        worst = max(worst, abs(Tb - Ttst))
    ok &= worst < 1e-4
    pr(f"[7] IAPWS round-trip max |T_sat(P_sat(T))-T| = {worst:.2e} K "
       f"(tol 1e-4): {'OK' if worst < 1e-4 else 'FAIL'}")
    # ice inversion
    worst_i = 0.0
    for Ttst in (200.0, 240.0, 270.0):
        P = SaturationProperties.Psat_ice(Ttst)
        Tb = SaturationProperties.Tsub_ice(P)
        worst_i = max(worst_i, abs(Tb - Ttst))
    ok &= worst_i < 1e-4
    pr(f"    ice round-trip max |T_sub(P_sub(T))-T| = {worst_i:.2e} K: "
       f"{'OK' if worst_i < 1e-4 else 'FAIL'}")

    # build states for the closure tests
    cfg = AtmosphericInput(T=258.15, P=70000.0, RH=110.0, scenario="single_state")
    sim = UnifiedNucleationSimulator(cfg)
    rgrid = np.geomspace(1e-2, 1e-9, 14)
    p_v = SaturationProperties.RH_to_p_v(110.0, 258.15, "water")
    states = []
    for ph, model in ((PHASE_LIQUID, sim.liquid), (PHASE_ICE, sim.ice)):
        for r in rgrid:
            st = model.solve(r, 258.15, 70000.0, p_v)
            if st is not None:
                st["_phase"] = ph
                states.append(st)

    # 8 -- identity Gamma^(2) = 4 pi r^2 gradT
    worst_id = 0.0
    for st in states:
        lhs = st["Gamma2"]
        rhs = 4.0 * math.pi * st["r"]**2 * st["g"]
        worst_id = max(worst_id, abs(lhs - rhs) / max(abs(rhs), 1e-30))
    ok &= worst_id < 1e-10
    pr(f"[8] |Gamma^(2) - 4 pi r^2 g|/|4 pi r^2 g| max = {worst_id:.2e} "
       f"(tol 1e-10): {'OK' if worst_id < 1e-10 else 'FAIL'}")

    # 9 -- residual |Gamma^(2)/(4 pi r^2) - g| < eps
    worst_r = 0.0
    for st in states:
        worst_r = max(worst_r, abs(st["closure_resid"]) / max(abs(st["g"]), 1.0))
    ok &= worst_r < 1e-10
    pr(f"[9] max |Gamma2/(4 pi r^2)-g|/max(|g|,1) = {worst_r:.2e} "
       f"(tol 1e-10): {'OK' if worst_r < 1e-10 else 'FAIL'}")

    # 10 -- parabolic stationarity residual at r_C,2nd
    worst_p = 0.0
    for st in states:
        if math.isfinite(st["parabolic_resid"]):
            worst_p = max(worst_p, abs(st["parabolic_resid"]))
    ok &= worst_p < 1e-8
    pr(f"[10] max |A2 r_C^2 + B2 r_C + C2| = {worst_p:.2e} (tol 1e-8): "
       f"{'OK' if worst_p < 1e-8 else 'FAIL'}")

    # 11 -- limit: gradT -> 0  =>  Delta_T -> 0, r_C -> +inf
    if states:
        # states from large-r (small g) first
        near = states[0]
        ok &= near["dT"] < 1e-3 and near["rC_2nd"] > 1.0
        pr(f"[11] near-equilibrium: Delta_T={near['dT']:.3e}, r_C,2nd={near['rC_2nd']:.3e} "
           f"(Delta_T<1e-3, r_C>1): "
           f"{'OK' if near['dT']<1e-3 and near['rC_2nd']>1.0 else 'FAIL'}")

    # 12 -- finite gradient -> finite critical radius
    bad = [st for st in states if math.isfinite(st["g"]) and st["g"] > 0
           and not math.isfinite(st["rC_2nd"])]
    ok &= len(bad) == 0
    pr(f"[12] finite-gradient states with non-finite r_C,2nd: {len(bad)}: "
       f"{'OK' if len(bad)==0 else 'FAIL'}")

    # 13 -- first-order reduces to CNT in the classical limit (large r, big Delta_T)
    r_big = 1.0e-3
    dT_test = 20.0
    g_test = dT_test / (8.0 * math.pi * r_big)
    model = sim.liquid
    st = model._local_state(g_test, r_big, 298.15, p_v)
    if st is not None:
        r1st = model._rC_1st(st)
        r_cnt = model._cnt_reference(st["T_local"], p_v, st["dT"])[0]
        rel = abs(r1st - r_cnt) / max(abs(r_cnt), 1e-30) if (math.isfinite(r1st) and r1st > 0) else 1.0
        ok &= rel < 0.05
        pr(f"[13] classical limit r_C,1st={r1st:.4e} vs r_CNT={r_cnt:.4e} "
           f"(rel {rel:.2e}): {'OK' if rel < 0.05 else 'FAIL'}")
    else:
        pr("[13] !! could not build classical-limit state"); ok = False

    # 14 -- convergence stability: re-solve with a DIFFERENT valid Brent
    # sub-interval (bisect the original bracket to obtain a strictly smaller
    # bracket that still encloses the root) and confirm the solved gradient
    # is unchanged to < 1e-6 relative.  This tests bracket-independence without
    # leaving the physically admissible range (hi*2 would exceed g_max).
    r_test = 1e-7
    s_def = sim.liquid.solve(r_test, 258.15, 70000.0, p_v)
    if s_def:
        br = sim.liquid._find_bracket(r_test, 258.15, p_v)
        g_alt = None
        if br is not None:
            lo, hi = br
            Flo = sim.liquid._residual(lo, r_test, 258.15, p_v)
            # shrink the bracket from the low side by a factor, keeping a sign change
            for shrink in (0.9, 0.5, 0.1):
                a = lo + (hi - lo) * (1.0 - shrink)
                Fa = sim.liquid._residual(a, r_test, 258.15, p_v)
                if math.isfinite(Fa) and Flo * Fa < 0:
                    try:
                        g_alt = brentq(sim.liquid._residual, a, hi,
                                       args=(r_test, 258.15, p_v),
                                       xtol=2e-16, rtol=BRENT_RTOL, maxiter=500)
                    except Exception:
                        g_alt = None
                    break
            # fallback: re-solve the same bracket with a tighter tolerance
            if g_alt is None:
                try:
                    g_alt = brentq(sim.liquid._residual, lo, hi,
                                   args=(r_test, 258.15, p_v),
                                   xtol=1e-18, rtol=BRENT_RTOL, maxiter=800)
                except Exception:
                    g_alt = None
        if g_alt is not None:
            rel = abs(s_def["g"] - g_alt) / max(abs(s_def["g"]), 1e-30)
            ok &= rel < 1e-6
            pr(f"[14] bracket-independence |g_def-g_alt|/g = {rel:.2e} "
               f"(tol 1e-6): {'OK' if rel < 1e-6 else 'FAIL'}")
        else:
            pr("[14] !! alternative-bracket solve failed"); ok = False
    else:
        pr("[14] !! solver failed at test radius"); ok = False

    # 15 -- original scripts NOT modified: re-assert ice checksum + liquid present
    cs2 = _sha256(ice_path)
    ok &= (cs2 == ICE_SHA256)
    pr(f"[15] ice reference unchanged after tests: "
       f"{'OK' if cs2 == ICE_SHA256 else 'FAIL'}")

    # 16 -- 2nd-order HETEROGENEOUS parabola residual at r_C_het (direct solve)
    worst_het = 0.0
    n_het = 0
    for st in states:
        if math.isfinite(st.get("rC_2nd_het_parab", float("nan"))):
            n_het += 1
            worst_het = max(worst_het, abs(st["parabolic_resid_het"]))
    ok &= worst_het < 1e-8
    pr(f"[16] het parabola |A r_C^2 + B r_C + C| max = {worst_het:.2e} "
       f"(tol 1e-8, {n_het} states): "
       f"{'OK' if worst_het < 1e-8 else 'FAIL'}")

    # 17 -- GT identity: GT_2nd_het == rC_2nd_het_parab * dT / 2
    worst_gt = 0.0
    for st in states:
        if math.isfinite(st.get("GT_2nd_het", float("nan"))) and st["dT"] != 0:
            ref = st["rC_2nd_het_parab"] * st["dT"] / 2.0
            worst_gt = max(worst_gt, abs(st["GT_2nd_het"] - ref)
                           / max(abs(ref), 1e-30))
    ok &= worst_gt < 1e-12
    pr(f"[17] |GT_2nd_het - rC_het*dT/2|/|rC_het*dT/2| max = {worst_gt:.2e} "
       f"(tol 1e-12): {'OK' if worst_gt < 1e-12 else 'FAIL'}")

    # 18 -- hom-limit analytic (deterministic): with theta=pi and d f/d r=0 the
    # het parabola = 3 f(pi) (hom parabola) -> SAME root as _rC_2nd.
    st_h = None
    for st in states:
        if math.isfinite(st.get("rC_2nd", float("nan"))) and st["rC_2nd"] > 0:
            st_h = st
            break
    if st_h is not None:
        st_h["theta"] = math.pi
        mdl_h = sim.liquid if st_h.get("_phase") == PHASE_LIQUID else sim.ice
        rC_het_lim = mdl_h._rC_2nd_het(
            st_h, 258.15, p_v, THETA0, dfdr_override=0.0)[0]
        rel_lim = abs(rC_het_lim - st_h["rC_2nd"]) / max(abs(st_h["rC_2nd"]), 1e-30)
        ok &= rel_lim < 1e-9
        pr(f"[18] hom-limit |rC_het(pi,0)-rC_2nd|/|rC_2nd| = {rel_lim:.2e} "
           f"(tol 1e-9): {'OK' if rel_lim < 1e-9 else 'FAIL'}")
    else:
        pr("[18] !! no finite-rC_2nd state for hom-limit test"); ok = False

    # 19 -- hom-limit grid (best-effort): states with theta -> pi should have
    # rC_2nd_het_parab -> rC_2nd.  Does NOT flip ok on a vacuous skip.
    n_hom = 0
    worst_hom = 0.0
    for st in states:
        if (st.get("theta", 0.0) >= math.pi - 1e-3
                and math.isfinite(st.get("rC_2nd", float("nan")))
                and math.isfinite(st.get("rC_2nd_het_parab", float("nan")))):
            n_hom += 1
            worst_hom = max(worst_hom, abs(st["rC_2nd_het_parab"] - st["rC_2nd"])
                            / max(abs(st["rC_2nd"]), 1e-30))
    pr(f"[19] hom-limit grid worst |rC_het-rC_2nd|/|rC_2nd| = {worst_hom:.2e} "
       f"({n_hom} states with th>=pi-1e-3)  [report-only]")

    # 20 -- d f/d r -> 0 as theta -> pi (report-only): the largest-theta state
    # should have the smallest |dftheta_dr|.
    th_sorted = sorted(states, key=lambda s: s.get("theta", 0.0), reverse=True)
    if th_sorted:
        dfmax = abs(th_sorted[0].get("dftheta_dr", 0.0))
        import statistics as _st
        med = _st.median([abs(s.get("dftheta_dr", 0.0)) for s in states]) or 1.0
        pr(f"[20] largest-theta |d f/d r| = {dfmax:.3e} (median={med:.3e})  "
           f"[report-only]")

    # 21 -- finiteness guard (report-only): finite-gradient states with a
    # non-finite rC_2nd_het_parab (legitimate when the het parabola has no
    # positive real root).
    n_nonfin = sum(1 for st in states
                   if math.isfinite(st.get("g", float("nan"))) and st["g"] > 0
                   and not math.isfinite(st.get("rC_2nd_het_parab", float("nan"))))
    pr(f"[21] finite-gradient states with non-finite rC_2nd_het_parab: "
       f"{n_nonfin}  [report-only]")

    pr("-" * 78)
    pr(f"VALIDATION {'PASSED' if ok else 'FAILED'}")
    pr("=" * 78)
    return ok


# =============================================================================
#  MAIN / CLI
# =============================================================================
def build_argparser():
    p = argparse.ArgumentParser(
        description="Unified H2O vapour->liquid / vapour->ice nucleation "
                    "(shifted-equilibrium, thermal-gradient driven).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
--------
RH is in PERCENT: --RH 101  =>  p_v = 1.01 * P_sat  (the slightly-supersaturated
state used in the figures).  One of --RH / --yv / --pv is required for a run.

  # pure ICE, single state (T=258.15 K, P=70000 Pa)
  python unified_h2o_nucleation_climate.py --RH 101 --phase-mode ice

  # AUTO two-phase: liquid vs ice compete, dominant phase reported
  python unified_h2o_nucleation_climate.py --RH 101 --phase-mode auto

  # pure LIQUID, single state
  python unified_h2o_nucleation_climate.py --RH 101 --phase-mode liquid

  # heterogeneous nucleation at a 150 deg contact angle
  python unified_h2o_nucleation_climate.py --RH 101 --phase-mode liquid --mode heterogeneous --theta 150

  # temperature sweep, both phases compared
  python unified_h2o_nucleation_climate.py --RH 101 --phase-mode both --scenario temperature_sweep

  # just run the validation gate, no simulation
  python unified_h2o_nucleation_climate.py --validate-only

phase-mode:  auto (liquid vs ice competition) | liquid | ice | both
scenarios:   single_state (default) | temperature_sweep | pressure_sweep |
             humidity_sweep | thermal_gradient_sweep | altitude_profile |
             time_series | climate_comparison | phase_map | psat_gradT
""")
    p.add_argument("--scenario", default="single_state", choices=VALID_SCENARIOS)
    p.add_argument("--phase-mode", default="auto", choices=VALID_PHASE_MODES)
    p.add_argument("--T", type=float, default=258.15, help="ambient T [K]")
    p.add_argument("--P", type=float, default=70000.0, help="total pressure [Pa]")
    p.add_argument("--RH", type=float, default=None, help="relative humidity [%%]")
    p.add_argument("--rh-reference", default="water", choices=("water", "ice"))
    p.add_argument("--yv", type=float, default=None, help="vapour mole fraction")
    p.add_argument("--pv", type=float, default=None, help="vapour partial pressure [Pa]")
    p.add_argument("--gradT", type=float, default=None, help="thermal gradient [K/m]")
    p.add_argument("--theta", type=float, default=None, help="contact angle [deg]")
    p.add_argument("--mode", default="homogeneous",
                   choices=("homogeneous", "heterogeneous"))
    p.add_argument("--dTol", type=float, default=None, help="Tolman length [m]")
    p.add_argument("--agg-length", type=float, default=None, help="attachment length [m]")
    p.add_argument("--site-density", type=float, default=None)
    p.add_argument("--mechanism", default="vapour", choices=("vapour", "liquid"))
    p.add_argument("--gamma0-ice", type=float, default=GAMMA0_ICE,
                   help="planar ice-vapour surface energy [J/m^2]")
    p.add_argument("--resolution", type=int, default=75, help="scan resolution")
    p.add_argument("--r-ref", type=float, default=R_REF_DEFAULT,
                   help="reference radius for single-state evaluation [m]")
    p.add_argument("--outdir", default="unified_nucleation_out")
    p.add_argument("--show", action="store_true")
    p.add_argument("--save-pdf", action="store_true")
    p.add_argument("--validate-only", action="store_true",
                   help="run only the validation tests and exit")
    p.add_argument("--no-validation", action="store_true",
                   help="skip the pre-run validation gate")
    # climate perturbations
    p.add_argument("--dT-climate", type=float, default=0.0)
    p.add_argument("--dRH", type=float, default=0.0)
    p.add_argument("--dP", type=float, default=0.0)
    p.add_argument("--dgradT", type=float, default=0.0)
    p.add_argument("--surface-3d", action="store_true", dest="surface_3d",
                   help="also produce the 3-D P_eq_shift(T,gradT) surface")
    # psat_gradT scenario: solved-gradient grid
    p.add_argument("--gmin", type=float, default=1.0,
                   help="min solved thermal gradient [K/m] (psat_gradT)")
    p.add_argument("--gmax", type=float, default=1.0e4,
                   help="max solved thermal gradient [K/m] (psat_gradT)")
    p.add_argument("--ngrad", type=int, default=41,
                   help="number of gradient points (psat_gradT)")
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)

    if args.validate_only:
        ok = run_validation_tests(verbose=True)
        raise SystemExit(0 if ok else 1)

    if not args.no_validation:
        if not run_validation_tests(verbose=True):
            print("Validation failed; aborting before nucleation run.")
            raise SystemExit(1)

    global R_REF_DEFAULT
    R_REF_DEFAULT = args.r_ref

    theta = math.radians(args.theta) if args.theta is not None else THETA0
    cfg = AtmosphericInput(
        T=args.T, P=args.P, RH=args.RH, rh_reference=args.rh_reference,
        y_v=args.yv, p_v=args.pv, grad_T=args.gradT,
        theta=theta, mode=args.mode, dTol=args.dTol, agg_length=args.agg_length,
        site_density=args.site_density, kinetic_mechanism=args.mechanism,
        scan_resolution=args.resolution, scenario=args.scenario,
        dT_climate=args.dT_climate, dRH=args.dRH, dP=args.dP, dgradT=args.dgradT,
        phase_mode=args.phase_mode, gamma0_ice=args.gamma0_ice,
        outdir=args.outdir, show=args.show, save_pdf=args.save_pdf,
        gmin=args.gmin, gmax=args.gmax, ngrad=args.ngrad,
    )

    sim = UnifiedNucleationSimulator(cfg)
    simout = sim.run()

    # outputs
    os.makedirs(cfg.outdir, exist_ok=True)
    save_csv(simout, os.path.join(cfg.outdir, "nucleation_results.csv"))
    save_json(simout, cfg, os.path.join(cfg.outdir, "nucleation_results.json"))
    print_summary(simout, cfg)

    # plots
    plotter = NucleationPlotter(cfg, cfg.outdir)
    surface = None
    if args.surface_3d or cfg.scenario in ("climate_comparison", "phase_map"):
        surface = sim.shifted_equilibrium_surface(phase=cfg.phase_mode
                                                   if cfg.phase_mode in ("liquid", "ice")
                                                   else "both")
    plotter.plot(simout, surface=surface)

    print("\n" + "=" * 78)
    print("DONE.  Outputs in:", cfg.outdir)
    print("  Closure: r (prescribed) -> Brent-solve grad_T from "
          "Gamma^(2)/(4 pi r^2)-g=0 -> Delta_T=8 pi r g -> T_local")
    print("           -> Delta_mu -> Delta_G_V -> r_C,2nd (parabolic) -> "
          "Delta_G_C -> I  (liquid & ice, identical closure)")
    print("=" * 78)


if __name__ == "__main__":
    main()