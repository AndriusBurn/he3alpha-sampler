#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:00:03 EDT 2023

@author: Andrius Burnelis
"""
import numpy as np
from scipy.special import psi, gamma
import sys; sys.path.append('./')
from unc_base_model import BaseModel
from data_loader import DataLoader
from functools import wraps
import time
import cProfile, pstats

def get_time(func):
    """
    Times any function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        func(*args, *kwargs)

        end_time = time.perf_counter()
        total_time = round(end_time - start_time, 3)

        print('Time: {} seconds'.format(total_time))

    return wrapper






class F_Wave_cbar_lambda_AR(BaseModel):
    def __init__(self, data, norm_grouping, param_prior, norm_prior, use_theory_cov):
        super().__init__(data, norm_grouping, param_prior, norm_prior, use_theory_cov)

        # Get the cross section at LO
        self.cs_LO_values = self.cs_theory(np.array([0, 0, 0, 0, 0, 0, 0]), 0)
        self.y_ref = np.reshape(self.cs_LO_values, (1, len(self.cs_LO_values)))

        # Set up the covariance matrix
        cov_expt_matrix = np.diag(self.err_cs**2)
        cov_theory_matrix = self.cov_theory()
        if use_theory_cov:
            cov_matrix = cov_expt_matrix + cov_theory_matrix
        else:
            cov_matrix = cov_expt_matrix
        self.inv_cov_matrix = np.linalg.inv(cov_matrix)
        self.Q_rest = np.sum(np.log(1.0 / np.sqrt(2.0 * np.pi * np.diag(cov_matrix))))

        # Set the Rutherford amplitude
        self.f_r = (-self.etavalue_cs / (2 * self.kvalue_cs)) * (
            1 / np.sin((self.theta_cs * (np.pi / 180)) / 2)**2)**(1 + (1.j) * self.etavalue_cs)

        #############################################
        # Set information on the 7/2- resonance state
        #############################################
        E_R = (4.57 - 1.5876) # MeV (position)
        Gamma_R = 0.160 # MeV (width)

        # Resonance Momentum
        self.gamma_R = np.sqrt(2 * self.mu * E_R) / self.h_bar_c

        # Get the k_c at resonance
        self.k_cR = (self.Zb * self.Zt * self.alpha * (self.mu + E_R)) / self.h_bar_c

        # Set the Sommerfeld parameter at resonance
        self.eta_R = self.k_cR / self.gamma_R

        # Define the H's at resonance
        self.H_R = psi(1.j * self.eta_R) + (1 / (2 * 1.j * self.eta_R)) - np.log(1.j * self.eta_R)
        self.H_R_prime = self.H_prime_eta(self.eta_R)


        ########################################
        # Precompute values used in cs_theory()
        ########################################
        # For r1plus
        self.r1plus_val1 = 2 * self.gamma1_plus**2 * gamma(2 + self.eta_B_plus)**2
        self.r1plus_val2 = 4 * self.kc_plus * self.H_plus
        self.r1plus_val3 = 2.j * self.kc_plus * self.eta_B_plus * (1 - self.eta_B_plus**2) * self.H_prime_plus

        # For r1minus
        self.r1minus_val1 = 2 * self.gamma1_minus**2 * gamma(2 + self.eta_B_minus)**2
        self.r1minus_val2 = 4 * self.kc_minus * self.H_minus
        self.r1minus_val3 = 2.j * self.kc_minus * self.eta_B_minus * (1 - self.eta_B_minus**2) * self.H_prime_minus

        # Legendre polynomials + derivatives
        self.LP1 = np.cos(self.theta_cs * (np.pi / 180))
        self.DLP1 = np.sin(self.theta_cs * (np.pi / 180))
        self.LP3 = 0.5 * (5 * np.power(np.cos(self.theta_cs * (np.pi / 180)), 3) - 3 * np.cos(self.theta_cs * (np.pi / 180)))
        self.DLP3 = 0.5 * (15 * np.square(np.cos(self.theta_cs * (np.pi / 180))) - 3)

        # Phase terms
        alpha_1 = np.arcsin((self.etavalue_cs / np.sqrt(1 + np.square(self.etavalue_cs))))
        self.phase_1 = np.cos(2 * alpha_1) + (1.j) * np.sin(2 * alpha_1)
        alpha_2 = alpha_1 + np.arcsin((self.etavalue_cs / np.sqrt(4 + np.square(self.etavalue_cs))))
        alpha_3 = alpha_2 + np.arcsin((self.etavalue_cs / np.sqrt(9 + np.square(self.etavalue_cs))))
        self.phase_3 = np.cos(2 * alpha_3) + 1.j * np.sin(2 * alpha_3)
        
        # ERE1
        self.ERE1_val1 = (self.kcvalue_cs**2 + self.kvalue_cs**2) * self.Hvalue_cs
        # f_c
        self.fc_val1 = self.kvalue_cs**2 * self.phase_1 * self.LP1
        # f_i
        self.fi_val1 = self.kvalue_cs**2 * self.phase_1 * self.DLP1

        #### F-WAVES
        # For A3plus
        self.A3plus_val1 = (self.k_cR / 18) * (self.eta_R**6 + 14 * self.eta_R**4 + 49 * self.eta_R**2 + 36) * self.gamma_R**6 * self.H_R.real

        # For C3_2
        self.C3_2value_cs = self.C0_2value_cs * (1 + np.square(self.etavalue_cs)) * (4 + np.square(self.etavalue_cs)) * (9 + np.square(self.etavalue_cs))

        # ERE3plus
        self.ERE3_plus_val1 = (self.kvalue_cs**6 * (9 + self.etavalue_cs**2) * 
                (4 + self.etavalue_cs**2) * (1 + self.etavalue_cs**2) * 
                self.Hvalue_cs)
        
        # f_c
        self.fc_val2 = np.power(self.kvalue_cs, 6) * self.phase_3 * self.LP3
        # f_i
        self.fi_val2 = np.power(self.kvalue_cs, 6) * self.phase_3 * self.DLP3 * np.sin(self.theta_cs * (np.pi / 180))



    def cs_theory(self, params, order):
        """
        This method utilizes the existence of a bound state, the asymptotic normalization coefficients,
        and the 7/2- (denoted 3+) resonance state to compute the theoretical cross section at the energies and 
        angles where we have data. For this version of the code, we will ignore the 3- channel.

        NOTE: C1+/- is really (C1+/-)^2 since we are sampling C1 squared
        """
        # Extract the parameters from params
        A0, r0, C1plus, P1plus, C1minus, P1minus, r3plus = params

        # Compute r1plus and r1minus based on the relation with ANCs
        r1plus = np.real(-self.r1plus_val1 / C1plus + (P1plus * self.gamma1_plus**2) + (self.r1plus_val2) + (self.r1plus_val3))
        r1minus = np.real(-self.r1minus_val1 / C1minus + (P1minus * self.gamma1_minus**2) + (self.r1minus_val2) + (self.r1minus_val3))

        # Compute A1plus and A1minus based on the bound state condition
        A1plus = - (r1plus * self.gamma1_plus**2) / 2 + (0.25 * P1plus * self.gamma1_plus**4) + (
            2 * self.kc_plus * (self.gamma1_plus**2 - self.kc_plus**2)) * self.H_plus
        A1minus = - (r1minus * self.gamma1_minus**2) / 2 + (0.25 * P1minus * self.gamma1_minus**4) + (
            2 * self.kc_minus * (self.gamma1_minus**2 - self.kc_minus**2)) * self.H_minus

        # Compute A3 based on the resonance state condition
        A3plus = (r3plus * self.gamma_R**2 / 2) - self.A3plus_val1

        # Set the parameterization based on the order
        if order == 0:
            # All parameters switched off
            A0 = 0
            r0 = 0
            A1plus = 0
            r1plus = 0
            P1plus = 0
            A1minus = 0
            r1minus = 0
            P1minus = 0
            A3plus = 0
            r3plus = 0
        elif order == 1:
            # Only r0, r1+, P1+/- are non-zero
            A0 = 0
            A1plus = 0
            A1minus = 0
            r1minus = 0
            A3plus = 0
            r3plus = 0
        elif order == 2:
            # Include all parameters
            pass
        else:
            raise Exception('Order {} not implemented!'.format(order))
            
        # Compute the Effective Range Functions
        K_0 = (1 / (2 * self.kcvalue_cs)) * (-A0 + 0.5 * r0 * self.kvalue_cs**2)
        K_1_plus = (1 / (2 * self.kcvalue_cs**3)) * (-A1plus + 
                0.5 * r1plus * self.kvalue_cs**2 + 
                    0.25 * P1plus * self.kvalue_cs**4)
        K_1_minus = (1 / (2 * self.kcvalue_cs**3)) * (-A1minus + 
                0.5 * r1minus * self.kvalue_cs**2 + 
                    0.25 * P1minus * self.kvalue_cs**4)
        
        ############# F-WAVE
        K_3_plus = (1 / (2 * self.kcvalue_cs**7)) * (-A3plus + 
                0.5 * r3plus * self.kvalue_cs**2)

        ERE_0 = (2 * self.kcvalue_cs * (K_0 - self.Hvalue_cs)) / self.C0_2value_cs
        ERE_1_plus = (2 * self.kcvalue_cs / (9 * self.C1_2value_cs)) * (
            self.kcvalue_cs**2 * K_1_plus - self.ERE1_val1)
        ERE_1_minus = (2 * self.kcvalue_cs / (9 * self.C1_2value_cs)) * (
            self.kcvalue_cs**2 * K_1_minus - self.ERE1_val1)

        ############## F-WAVE
        ERE_3_plus = (2 * self.kcvalue_cs / self.C3_2value_cs) * (
            36 * self.kcvalue_cs**6 * K_3_plus - self.ERE3_plus_val1)
        
        # Compute the amplitude of each of the components
        # Rutherford - precomputed in __init__()
        
        # Coulomb
        f_c = self.f_r + (1 / ERE_0) + self.fc_val1 * (
            (2 / ERE_1_plus) + (1 / ERE_1_minus)) + (self.fc_val2 * ((
            4 / ERE_3_plus) + 0.0)) # <-- ERE_3_minus = 0 here

        # Interaction
        f_i = self.fi_val1 * ((1 / ERE_1_minus) - (1 / ERE_1_plus)) + (
            self.fi_val2 * ((1 / ERE_3_plus) - 0.0)) # <-- ERE_3_minus = 0 here

        sigma = 10 * (np.abs(f_c)**2 + np.abs(f_i)**2)
        sigma_R = 10 * np.abs(self.f_r)**2
        sigma_ratio = sigma / sigma_R
        return sigma_ratio






def main():
    pass

if __name__ == "__main__":
    main()