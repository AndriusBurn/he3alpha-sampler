#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:05:13 EDT 2023

@author: Andrius Burnelis
"""
import numpy as np
from scipy.special import psi, gamma
import sys; sys.path.append('./')
from base_model import BaseModel
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


class BS_C(BaseModel):
    def __init__(self, data, norm_grouping, param_prior, norm_prior, use_theory_cov):
        super().__init__(data, norm_grouping, param_prior, norm_prior, use_theory_cov)
        
        # Set the Rutherford amplitude
        self.f_r = (-self.etavalue_cs / (2 * self.kvalue_cs)) * (
            1 / np.sin((self.theta_cs * (np.pi / 180)) / 2)**2)**(1 + (1.j) * self.etavalue_cs)

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
        # Phase terms
        alpha_1 = np.arcsin((self.etavalue_cs / np.sqrt(1 + np.square(self.etavalue_cs))))
        self.phase_1 = np.cos(2 * alpha_1) + (1.j) * np.sin(2 * alpha_1)
        # ERE1
        self.ERE1_val1 = (self.kcvalue_cs**2 + self.kvalue_cs**2) * self.Hvalue_cs
        # f_c
        self.fc_val1 = self.kvalue_cs**2 * self.phase_1 * self.LP1
        # f_i
        self.fi_val1 = self.kvalue_cs**2 * self.phase_1 * self.DLP1

    def cs_theory(self, params):
        """
        This method utilizes the existence of a bound state and also the asymptotic normalization coefficients
        to compute the theoretical cross section at the energies and angles where we have data.

        NOTE: C1+/- is really (C1+/-)^2 since we are sampling C1 squared
        """
        # Extract the parameters from params
        A0, r0, C1plus, P1plus, C1minus, P1minus = params

        # Convert theta to radians
        # theta_rad = self.theta_cs * (np.pi / 180)

        # Compute r1plus and r1minus based on the relation
        r1plus = np.real(-self.r1plus_val1 / C1plus + (P1plus * self.gamma1_plus**2) + (self.r1plus_val2) + (self.r1plus_val3))
        r1minus = np.real(-self.r1minus_val1 / C1minus + (P1minus * self.gamma1_minus**2) + (self.r1minus_val2) + (self.r1minus_val3))

        # Compute A1plus and A1minus based on the bound state condition
        A1plus = - (r1plus * self.gamma1_plus**2) / 2 + (0.25 * P1plus * self.gamma1_plus**4) + (
            2 * self.kc_plus * (self.gamma1_plus**2 - self.kc_plus**2)) * self.H_plus
        A1minus = - (r1minus * self.gamma1_minus**2) / 2 + (0.25 * P1minus * self.gamma1_minus**4) + (
            2 * self.kc_minus * (self.gamma1_minus**2 - self.kc_minus**2)) * self.H_minus

        # Set up the necessary Legendre polynomials
        # P_1 = np.cos(theta_rad)

        # Phase terms
        # alpha_1 = np.arcsin((self.etavalue_cs / np.sqrt(1 + np.square(self.etavalue_cs))))
        # phase_1 = np.cos(2 * alpha_1) + (1.j) * np.sin(2 * alpha_1)

        K_0 = (1 / (2 * self.kcvalue_cs)) * (-A0 + 0.5 * r0 * self.kvalue_cs**2)
        K_1_plus = (1 / (2 * self.kcvalue_cs**3)) * (-A1plus + 
                0.5 * r1plus * self.kvalue_cs**2 + 
                    0.25 * P1plus * self.kvalue_cs**4)
        K_1_minus = (1 / (2 * self.kcvalue_cs**3)) * (-A1minus + 
                0.5 * r1minus * self.kvalue_cs**2 + 
                    0.25 * P1minus * self.kvalue_cs**4)
        
        ERE_0 = (2 * self.kcvalue_cs * (K_0 - self.Hvalue_cs)) / self.C0_2value_cs
        ERE_1_plus = (2 * self.kcvalue_cs / (9 * self.C1_2value_cs)) * (
            self.kcvalue_cs**2 * K_1_plus - self.ERE1_val1)
        ERE_1_minus = (2 * self.kcvalue_cs / (9 * self.C1_2value_cs)) * (
            self.kcvalue_cs**2 * K_1_minus - self.ERE1_val1)

        # Compute the amplitude of each of the components
        # Rutherford - precomputed in __init__()
        
        # Coulomb
        f_c = self.f_r + (1 / ERE_0) + self.fc_val1 * (
            (2 / ERE_1_plus) + (1 / ERE_1_minus))

        # Interaction
        f_i = self.fi_val1 * ((1 / ERE_1_minus) - (1 / ERE_1_plus))

        sigma = 10 * (np.abs(f_c)**2 + np.abs(f_i)**2)
        sigma_R = 10 * np.abs(self.f_r)**2
        sigma_ratio = sigma / sigma_R
        return sigma_ratio
        
    def get_cs_theory(self, params, theta_deg, E_lab):
            """
            This method utilizes the existence of a bound state, and the asymptotic normalization coefficients
            to compute the theoretical cross section at a given angle and energy we choose. 

            NOTE: C1+/- is really (C1+/-)^2 since we are sampling C1 squared
            """
            # Extract the parameters from params
            A0, r0, C1plus, P1plus, C1minus, P1minus = params

            # Convert theta into radians
            theta = theta_deg * (np.pi / 180)

            # Define the necessary parameters
            eta = self.eta(E_lab)
            kc = self.kc(E_lab)
            k = self.k(E_lab)
            H = self.H(E_lab)
            C0_2 = self.C0_2(E_lab)
            C1_2 = self.C1_2(E_lab)


            # Legendre polynomials + derivatives
            LP1 = np.cos(theta)
            DLP1 = np.sin(theta)
            
            # Phase terms
            alpha_1 = np.arcsin((eta / np.sqrt(1 + np.square(eta))))
            phase_1 = np.cos(2 * alpha_1) + (1.j) * np.sin(2 * alpha_1)

            # Compute r1plus and r1minus based on the relation with ANCs
            r1plus = np.real(-self.r1plus_val1 / C1plus + (P1plus * self.gamma1_plus**2) + (self.r1plus_val2) + (self.r1plus_val3))
            r1minus = np.real(-self.r1minus_val1 / C1minus + (P1minus * self.gamma1_minus**2) + (self.r1minus_val2) + (self.r1minus_val3))

            # Compute A1plus and A1minus based on the bound state condition
            A1plus = - (r1plus * self.gamma1_plus**2) / 2 + (0.25 * P1plus * self.gamma1_plus**4) + (
                2 * self.kc_plus * (self.gamma1_plus**2 - self.kc_plus**2)) * self.H_plus
            A1minus = - (r1minus * self.gamma1_minus**2) / 2 + (0.25 * P1minus * self.gamma1_minus**4) + (
                2 * self.kc_minus * (self.gamma1_minus**2 - self.kc_minus**2)) * self.H_minus


            # Compute the Effective Range Functions
            K_0 = (1 / (2 * kc)) * (-A0 + 0.5 * r0 * k**2)
            K_1_plus = (1 / (2 * kc**3)) * (-A1plus + 
                    0.5 * r1plus * k**2 + 
                        0.25 * P1plus * k**4)
            K_1_minus = (1 / (2 * kc**3)) * (-A1minus + 
                    0.5 * r1minus * k**2 + 
                        0.25 * P1minus * k**4)
            

            ERE_0 = (2 * kc * (K_0 - H)) / C0_2
            ERE_1_plus = (2 * kc / (9 * C1_2)) * (
                kc**2 * K_1_plus - (kc**2 + k**2) * H)
            ERE_1_minus = (2 * kc / (9 * C1_2)) * (
                kc**2 * K_1_minus - (kc**2 + k**2) * H)
            

            # Set the Rutherford amplitude
            f_r = (-eta / (2 * k)) * (
                1 / np.sin((theta) / 2)**2)**(1 + (1.j) * eta)
            
            # Coulomb
            f_c = f_r + (1 / ERE_0) + (k**2 * phase_1 * LP1) * (
                (2 / ERE_1_plus) + (1 / ERE_1_minus)) 

            # Interaction
            f_i = (k**2 * phase_1 * DLP1) * ((1 / ERE_1_minus) - (1 / ERE_1_plus)) 

            sigma = 10 * (np.abs(f_c)**2 + np.abs(f_i)**2)
            sigma_R = 10 * np.abs(f_r)**2
            sigma_ratio = sigma / sigma_R
            return sigma_ratio











class F_Wave_AR(BaseModel):
    def __init__(self, data, norm_grouping, param_prior, norm_prior, use_theory_cov):
        super().__init__(data, norm_grouping, param_prior, norm_prior, use_theory_cov)

        # Set the Rutherford amplitude
        self.f_r = (-self.etavalue_cs / (2 * self.kvalue_cs)) * (
            1 / np.sin((self.theta_cs * (np.pi / 180)) / 2)**2)**(1 + (1.j) * self.etavalue_cs)

        ########################################
        # Set information on the 7/2- resonance state
        ########################################
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


    def cs_theory(self, params):
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


    def get_cs_theory(self, params, theta_deg, E_lab):
            """
            This method utilizes the existence of a bound state, the asymptotic normalization coefficients,
            and the 7/2- (denoted 3+) resonance state to compute the theoretical cross section at a given angle
            and energy we choose. For this version of the code, we will ignore the 3- channel.

            NOTE: C1+/- is really (C1+/-)^2 since we are sampling C1 squared
            """
            # Extract the parameters from params
            A0, r0, C1plus, P1plus, C1minus, P1minus, r3plus = params

            # Convert theta into radians
            theta = theta_deg * (np.pi / 180)

            # Define the necessary parameters
            eta = self.eta(E_lab)
            kc = self.kc(E_lab)
            k = self.k(E_lab)
            H = self.H(E_lab)
            C0_2 = self.C0_2(E_lab)
            C1_2 = self.C1_2(E_lab)
            C3_2 = C0_2 * (1 + np.square(eta)) * (4 + np.square(eta)) * (9 + np.square(eta))


            # Legendre polynomials + derivatives
            LP1 = np.cos(theta)
            DLP1 = np.sin(theta)
            LP3 = 0.5 * (5 * np.power(np.cos(theta), 3) - 3 * np.cos(theta))
            DLP3 = 0.5 * (15 * np.square(np.cos(theta)) - 3)
            
            # Phase terms
            alpha_1 = np.arcsin((eta / np.sqrt(1 + np.square(eta))))
            phase_1 = np.cos(2 * alpha_1) + (1.j) * np.sin(2 * alpha_1)
            alpha_2 = alpha_1 + np.arcsin((eta / np.sqrt(4 + np.square(eta))))
            alpha_3 = alpha_2 + np.arcsin((eta / np.sqrt(9 + np.square(eta))))
            phase_3 = np.cos(2 * alpha_3) + 1.j * np.sin(2 * alpha_3)

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


            # Compute the Effective Range Functions
            K_0 = (1 / (2 * kc)) * (-A0 + 0.5 * r0 * k**2)
            K_1_plus = (1 / (2 * kc**3)) * (-A1plus + 
                    0.5 * r1plus * k**2 + 
                        0.25 * P1plus * k**4)
            K_1_minus = (1 / (2 * kc**3)) * (-A1minus + 
                    0.5 * r1minus * k**2 + 
                        0.25 * P1minus * k**4)
            
            ############# F-WAVE
            K_3_plus = (1 / (2 * kc**7)) * (-A3plus + 
                    0.5 * r3plus * k**2)

            ERE_0 = (2 * kc * (K_0 - H)) / C0_2
            ERE_1_plus = (2 * kc / (9 * C1_2)) * (
                kc**2 * K_1_plus - (kc**2 + k**2) * H)
            ERE_1_minus = (2 * kc / (9 * C1_2)) * (
                kc**2 * K_1_minus - (kc**2 + k**2) * H)

            ############## F-WAVE
            ERE_3_plus = (2 * kc / C3_2) * (
                36 * kc**6 * K_3_plus - (
                k**6 * (9 + eta**2) * (4 + eta**2) * (1 + eta**2) * H))
            

            # Set the Rutherford amplitude
            f_r = (-eta / (2 * k)) * (
                1 / np.sin((theta) / 2)**2)**(1 + (1.j) * eta)
            
            # Coulomb
            f_c = f_r + (1 / ERE_0) + (k**2 * phase_1 * LP1) * (
                (2 / ERE_1_plus) + (1 / ERE_1_minus)) + ((k**6 * phase_3 * LP3) * (( 
                4 / ERE_3_plus) + 0.0)) # <-- ERE_3_minus = 0 here

            # Interaction
            f_i = (k**2 * phase_1 * DLP1) * ((1 / ERE_1_minus) - (1 / ERE_1_plus)) + (
                (k**6 * phase_3 * DLP3 * np.sin(theta)) * ((1 / ERE_3_plus) - 0.0)) # <-- ERE_3_minus = 0 here

            sigma = 10 * (np.abs(f_c)**2 + np.abs(f_i)**2)
            sigma_R = 10 * np.abs(f_r)**2
            sigma_ratio = sigma / sigma_R
            return sigma_ratio











class F_Wave_cbar_lambda_AR(BaseModel):
     def __init__(self, data, norm_grouping, param_prior, norm_prior, use_theory_cov):
        super().__init__(data, norm_grouping, param_prior, norm_prior, use_theory_cov)

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

        ########################################
        # Set information on the 7/2- resonance state
        ########################################
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











class BS_C_flatprior(BaseModel):
    def __init__(self, data, norm_grouping, param_prior, norm_prior, use_theory_cov):
        super().__init__(data, norm_grouping, param_prior, norm_prior, use_theory_cov)
        
        # Set the Rutherford amplitude
        self.f_r = (-self.etavalue_cs / (2 * self.kvalue_cs)) * (
            1 / np.sin((self.theta_cs * (np.pi / 180)) / 2)**2)**(1 + (1.j) * self.etavalue_cs)

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
        # Phase terms
        alpha_1 = np.arcsin((self.etavalue_cs / np.sqrt(1 + np.square(self.etavalue_cs))))
        self.phase_1 = np.cos(2 * alpha_1) + (1.j) * np.sin(2 * alpha_1)
        # ERE1
        self.ERE1_val1 = (self.kcvalue_cs**2 + self.kvalue_cs**2) * self.Hvalue_cs
        # f_c
        self.fc_val1 = self.kvalue_cs**2 * self.phase_1 * self.LP1
        # f_i
        self.fi_val1 = self.kvalue_cs**2 * self.phase_1 * self.DLP1


    def cs_theory(self, params):
        """
        This method utilizes the existence of a bound state and also the asymptotic normalization coefficients
        to compute the theoretical cross section at the energies and angles where we have data.

        NOTE: C1+/- is really (C1+/-)^2 since we are sampling C1 squared
        """
        # Extract the parameters from params
        A0, r0, C1plus, P1plus, C1minus, P1minus = params

        # Convert theta to radians
        # theta_rad = self.theta_cs * (np.pi / 180)

        # Compute r1plus and r1minus based on the relation
        r1plus = np.real(-self.r1plus_val1 / C1plus + (P1plus * self.gamma1_plus**2) + (self.r1plus_val2) + (self.r1plus_val3))
        r1minus = np.real(-self.r1minus_val1 / C1minus + (P1minus * self.gamma1_minus**2) + (self.r1minus_val2) + (self.r1minus_val3))

        # Compute A1plus and A1minus based on the bound state condition
        A1plus = - (r1plus * self.gamma1_plus**2) / 2 + (0.25 * P1plus * self.gamma1_plus**4) + (
            2 * self.kc_plus * (self.gamma1_plus**2 - self.kc_plus**2)) * self.H_plus
        A1minus = - (r1minus * self.gamma1_minus**2) / 2 + (0.25 * P1minus * self.gamma1_minus**4) + (
            2 * self.kc_minus * (self.gamma1_minus**2 - self.kc_minus**2)) * self.H_minus

        # Set up the necessary Legendre polynomials
        # P_1 = np.cos(theta_rad)

        # Phase terms
        # alpha_1 = np.arcsin((self.etavalue_cs / np.sqrt(1 + np.square(self.etavalue_cs))))
        # phase_1 = np.cos(2 * alpha_1) + (1.j) * np.sin(2 * alpha_1)

        K_0 = (1 / (2 * self.kcvalue_cs)) * (-A0 + 0.5 * r0 * self.kvalue_cs**2)
        K_1_plus = (1 / (2 * self.kcvalue_cs**3)) * (-A1plus + 
                0.5 * r1plus * self.kvalue_cs**2 + 
                    0.25 * P1plus * self.kvalue_cs**4)
        K_1_minus = (1 / (2 * self.kcvalue_cs**3)) * (-A1minus + 
                0.5 * r1minus * self.kvalue_cs**2 + 
                    0.25 * P1minus * self.kvalue_cs**4)
        
        ERE_0 = (2 * self.kcvalue_cs * (K_0 - self.Hvalue_cs)) / self.C0_2value_cs
        ERE_1_plus = (2 * self.kcvalue_cs / (9 * self.C1_2value_cs)) * (
            self.kcvalue_cs**2 * K_1_plus - self.ERE1_val1)
        ERE_1_minus = (2 * self.kcvalue_cs / (9 * self.C1_2value_cs)) * (
            self.kcvalue_cs**2 * K_1_minus - self.ERE1_val1)

        # Compute the amplitude of each of the components
        # Rutherford - precomputed in __init__()
        
        # Coulomb
        f_c = self.f_r + (1 / ERE_0) + self.fc_val1 * (
            (2 / ERE_1_plus) + (1 / ERE_1_minus))

        # Interaction
        f_i = self.fi_val1 * ((1 / ERE_1_minus) - (1 / ERE_1_plus))

        sigma = 10 * (np.abs(f_c)**2 + np.abs(f_i)**2)
        sigma_R = 10 * np.abs(self.f_r)**2
        sigma_ratio = sigma / sigma_R
        return sigma_ratio
        
    def get_cs_theory(self, params, theta_deg, E_lab):
        """
        This method utilizes the existence of a bound state, and the asymptotic normalization coefficients
        to compute the theoretical cross section at a given angle and energy we choose. 

        NOTE: C1+/- is really (C1+/-)^2 since we are sampling C1 squared
        """
        # Extract the parameters from params
        A0, r0, C1plus, P1plus, C1minus, P1minus = params

        # Convert theta into radians
        theta = theta_deg * (np.pi / 180)

        # Define the necessary parameters
        eta = self.eta(E_lab)
        kc = self.kc(E_lab)
        k = self.k(E_lab)
        H = self.H(E_lab)
        C0_2 = self.C0_2(E_lab)
        C1_2 = self.C1_2(E_lab)


        # Legendre polynomials + derivatives
        LP1 = np.cos(theta)
        DLP1 = np.sin(theta)
        
        # Phase terms
        alpha_1 = np.arcsin((eta / np.sqrt(1 + np.square(eta))))
        phase_1 = np.cos(2 * alpha_1) + (1.j) * np.sin(2 * alpha_1)

        # Compute r1plus and r1minus based on the relation with ANCs
        r1plus = np.real(-self.r1plus_val1 / C1plus + (P1plus * self.gamma1_plus**2) + (self.r1plus_val2) + (self.r1plus_val3))
        r1minus = np.real(-self.r1minus_val1 / C1minus + (P1minus * self.gamma1_minus**2) + (self.r1minus_val2) + (self.r1minus_val3))

        # Compute A1plus and A1minus based on the bound state condition
        A1plus = - (r1plus * self.gamma1_plus**2) / 2 + (0.25 * P1plus * self.gamma1_plus**4) + (
            2 * self.kc_plus * (self.gamma1_plus**2 - self.kc_plus**2)) * self.H_plus
        A1minus = - (r1minus * self.gamma1_minus**2) / 2 + (0.25 * P1minus * self.gamma1_minus**4) + (
            2 * self.kc_minus * (self.gamma1_minus**2 - self.kc_minus**2)) * self.H_minus


        # Compute the Effective Range Functions
        K_0 = (1 / (2 * kc)) * (-A0 + 0.5 * r0 * k**2)
        K_1_plus = (1 / (2 * kc**3)) * (-A1plus + 
                0.5 * r1plus * k**2 + 
                    0.25 * P1plus * k**4)
        K_1_minus = (1 / (2 * kc**3)) * (-A1minus + 
                0.5 * r1minus * k**2 + 
                    0.25 * P1minus * k**4)
        

        ERE_0 = (2 * kc * (K_0 - H)) / C0_2
        ERE_1_plus = (2 * kc / (9 * C1_2)) * (
            kc**2 * K_1_plus - (kc**2 + k**2) * H)
        ERE_1_minus = (2 * kc / (9 * C1_2)) * (
            kc**2 * K_1_minus - (kc**2 + k**2) * H)
        

        # Set the Rutherford amplitude
        f_r = (-eta / (2 * k)) * (
            1 / np.sin((theta) / 2)**2)**(1 + (1.j) * eta)
        
        # Coulomb
        f_c = f_r + (1 / ERE_0) + (k**2 * phase_1 * LP1) * (
            (2 / ERE_1_plus) + (1 / ERE_1_minus)) 

        # Interaction
        f_i = (k**2 * phase_1 * DLP1) * ((1 / ERE_1_minus) - (1 / ERE_1_plus)) 

        sigma = 10 * (np.abs(f_c)**2 + np.abs(f_i)**2)
        sigma_R = 10 * np.abs(f_r)**2
        sigma_ratio = sigma / sigma_R
        return sigma_ratio


    def log_prior(self, parameters):
        """
        Computes the log-prior of a given set of parameters.
        """
        # Cast the parameters to an array
        parameters = np.array(parameters)

        # If all the parameters fall within the bounds of the prior, compute the prior
        if np.logical_and(self.prior_info[:, 0] <= parameters, parameters <= self.prior_info[:, 1]).all():
            log_p = 0.0

            # Sum over all the contributions from the parameters (flat prior)
            for i in range(0, self.erp_dim):
                log_p += self.lp_flat(parameters[i], self.bounds[i])

            # Sum over all the contributions for the normalization parameters (gaussian)
            for i in range(self.erp_dim, self.total_dim):
                mu, sigma = self.prior_info[i, 2:]
                log_p += self.lp_gauss(parameters[i], mu, sigma, self.prior_info[i, :2])

            return log_p
        else:
            return -np.inf
































def main():
    # TEST
    # Load the data and norm info
    # E_min : [0.676, 0.84 , 1.269, 1.741, 2.12 , 2.609, 2.609, 3.586, 4.332, 5.475]
    # E_max : [0.706, 0.868, 1.292, 1.759, 2.137, 2.624, 2.624, 3.598, 4.342, 5.484]
    E_min = 0.676 # MeV
    E_max = 4.342 # MeV
    which_data = 'som'

    loader = DataLoader(E_min, E_max, which_data)
    data = loader.get_data()
    norm_grouping = loader.get_normalization_grouping()


    # Set up the priors
    param_bounds = np.array([[-0.02, 0.06], [-3, 3], [5.0, 25.0], [-6, 6], [5.0, 25.0], [-6, 6]])
    params_prior = np.array([[0.025, 0.015], [0.8, 0.4], [13.84, 1.63], [0.0, 1.6], [12.59, 1.85], [0.0, 1.6]]) # center, width
    
    gauss_prior_params = np.hstack([param_bounds, params_prior])
    gauss_prior_f = loader.get_normalization_prior_info()

    # Set up the model
    model = BS_C(data, norm_grouping, gauss_prior_params, gauss_prior_f)

    test = np.concatenate([gauss_prior_params[:, 2], gauss_prior_f[:, 2]])

    # model.cs_theory(test[:6])

    with cProfile.Profile() as profile:
        for i in range(0, 10000):
            model.cs_theory(test[:6])


    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()

    # for i in range(0, 1000):
    #     # model.cs_theory(test[:6])
    #     model.log_likelihood(test)


if __name__ == '__main__':
    main()