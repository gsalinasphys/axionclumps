from math import pi

import mpmath as mp
import numpy as np

G = 1.32712e11                  # Newton's constant (km^3/(M_Sun s^2))
rho_eq = 5.46e-28               # Energy density at MR equality in 10^{-10}*M_Sun/km^3


class AxionMiniclusterNFW:
    def __init__(self, rCM: mp.matrix, vCM: mp.matrix, mass: float = 1.,
                delta: float = 1.55, c: float = 100., vdisptype: str = 'Maxwell-Boltzmann'):
        self.rCM = rCM                      # Position (km) of center of mass
        self.vCM = vCM                      # Velocity (km/s) of center of mass
        self.mass = mass                    # Axion minicluster mass (10^{-10} M_Sun)
        self.delta = delta                  # Parameter delta for NFW profile
        self.c = c                          # Concentration for NFW profile
        self.vdisptype = vdisptype          # Type of velocity dispersion curve: No dispersion (set to None) or Maxwell-Boltzmann

    def rho_s(self) -> float:    # In 10^{-10}*M_Sun/km^3
        return 140 * (1 + self.delta) * self.delta**3 * rho_eq

    def rs(self) -> float:   # In km
        f_NFW = np.log(1 + self.c) - self.c/(1 + self.c)
        return (self.mass/(4*pi*self.rho_s()*f_NFW)) ** (1/3)

    def rtrunc(self) -> float:  # In km
        return self.c*self.rs()

    # Density profile in units of 10^{-10}*M_Sun/km^3
    def rho_prf(self, r: float) -> float:
        if r <= 0. or r > self.rtrunc():
            return 0.
            
        return 4 * pi * r * self.rho_s() / (1/self.rs()*(1 + r/self.rs())**2)

    # def grav_pot(self, positions: np.ndarray) -> np.ndarray:   # In units of (km/s)^2
    #     ds = mag(positions - self.rCM)
    #     return -4e-10*pi*G*self.rho_s()*self.rs()**3/ds*np.log((ds + self.rs())/self.rs())
    
    # # Enclosed mass from a given position, in units of 10^{-10} M_Sun
    # def encl_mass(self, positions: np.ndarray) -> np.ndarray:
    #     ds = mag(positions - self.rCM)
    #     return cases(ds-self.rtrunc(),
    #                 4*pi*self.rho_s()*self.rs()**3*(np.log((ds + self.rs())/self.rs())-ds/(ds + self.rs())),
    #                 self.mass)
        
    # # Escape velocity in km/s
    # def vesc(self, position: np.ndarray) -> float:
    #     return np.sqrt(abs(2*self.grav_pot(position)))

    # # Circular velocity in km/s
    # def vcirc(self, position: np.ndarray) -> float:
        
    #     rstoCM = position - self.rCM
    #     return 1e-5*np.sqrt(G*self.encl_mass(position)/mag(rstoCM)) * np.heaviside(self.rtrunc()-mag(rstoCM), 1.)

    # # Maximum value of velocity dispersion
    # def deltav(self, prec: int = 100_000) -> float:
    #     if not self.vdisptype:
    #         return 0.
        
    #     rs = np.linspace(1e-8, 1., prec) * self.rtrunc()
    #     vscirc = np.empty_like(rs)
    #     for ii in range(prec):
    #         vscirc[ii] = self.vcirc(rs[ii])
            
    #     return max(vscirc)

    # # Velocity dispersion at a given position inside the minicluster
    # def vdisp(self, position: np.ndarray) -> np.ndarray:
    #     found = False
    #     while not found:
    #         if self.vdisptype == 1: # Maxwell-Boltzmann
    #             vesc, vcirc = self.vesc(position), self.vcirc(position)
    #             v_try = np.random.normal(0, vcirc, 3)
    #             if mag(v_try) < vesc:
    #                 found = True
        
    #     return mag(v_try)*randdir3d()
    
    # # Velocity dispersion for many positions inside the minicluster
    # def vsdisp(self, positions: np.ndarray) -> np.ndarray:
    #     for ii in range(len(positions)):
    #         yield self.vdisp(positions[ii])
