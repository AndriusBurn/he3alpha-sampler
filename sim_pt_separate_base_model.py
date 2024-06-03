#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:19:51 EDT 2024

@author: Andrius Burnelis
"""
import numpy as np, warnings
import sys; sys.path.append('./')
from data_loader import DataLoader
from scipy.special import psi, gamma
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.linalg import det
import os, tqdm, pickle, scipy
from numpy import typing as npt
from typing import Tuple
from functools import wraps
import time
import cProfile, pstats

class SimBaseModel:
    def __init__(self, data : npt.ArrayLike, norm_grouping : npt.ArrayLike, param_prior : npt.ArrayLike, norm_prior : npt.ArrayLike, use_theory_cov : bool):
        """
        This class contains all the basic functions that every model should contain.
        The user must define their own cs_theory in the parameterization they choose.
        Arguments:
        ----------
        data : array
        Array that contains the data in the format [Elab, theta, cs, err]

        norm_grouping : array
        Array where all the same numbers correspond to the same normalizing parameter

        param_prior + norm_prior : array
        Array containing the prior info for the parameters in the format [low, high, mu, sigma]
        """
        # Unpack the data
        self.Elab_cs = data[:, 0]
        self.theta_cs = data[:, 1]
        self.cs_data = data[:, 2]
        self.err_cs = data[:, 3]
        self.norm_grouping = norm_grouping

        # Define initial constants and useful parameters
        self.Zb = 2 # Charge number
        self.Zt = 2 # Charge number
        self.mb = 2809.43 # MeV - 3He
        self.mt = 3728.42 # MeV - alpha
        self.mu = (self.mb * self.mt) / (self.mb + self.mt) # Reduced mass

        # Set useful constants and variables
        self.alpha = 1 / 137.036 # dimless (fine structure alpha_em)
        self.h_bar_c = 197.327 # MeV fm

        ##############################################################################
        #                       Compute the values of k, kc, etc...
        ##############################################################################
        self.kvalue_cs = self.k(self.Elab_cs)
        self.kcvalue_cs = self.kc(self.Elab_cs)
        self.etavalue_cs = self.eta(self.Elab_cs)
        self.Hvalue_cs_real = np.real(self.H(self.Elab_cs))
        self.Hvalue_cs_imag = np.imag(self.H(self.Elab_cs))
        self.Hvalue_cs = self.Hvalue_cs_real + 1.j * self.Hvalue_cs_imag
        self.C0_2value_cs = self.C0_2(self.Elab_cs)
        self.C1_2value_cs = self.C1_2(self.Elab_cs)
        self.cs_LO_values = self.cs_LO()

        # Initialize the hyperparameters
        # self.n_c = 2 # Original code
        # self.n_c = 1 # Step 2
        self.n_c = 2 # Step 3
        self.Tau_0 = 1.5
        self.nu_0 = 1.5

        self.Tau = self.Tau_0
        # self.nu = self.nu_0 + (self.cs_data.shape[0] * self.n_c) # Original code
        # self.nu = 1.5 # Step 1.
        self.nu = self.nu_0 + (self.cs_data.shape[0] * self.n_c) # Step 2.a + 3.a
        # self.nu = self.nu_0 + self.n_c # Step 2.b + 3.b


        # Store the numerator of Q
        # Get the momentum transfer
        q = 2.0 * self.kvalue_cs * np.sin((self.theta_cs * np.pi / 180) / 2.0)
        self.Q_numerator = np.array([max(q[t], self.kvalue_cs[t])
                    for t in range(len(self.theta_cs))]).real
        
        self.exp_piece = np.diag(self.err_cs**2)
        self.use_theory_cov = use_theory_cov
        self.cov_matrix = None
        self.inverse_cov_matrix = None
        self.Q_rest = None
        self.lambda_b_norm_scale = None
        self.c_squared_sum = None
        self.c_denom_1 = self.cs_LO_values * self.Q_numerator
        self.c_denom_2 = self.cs_LO_values * np.square(self.Q_numerator)

        ##############################################################################
                            # Define the binding energy of the 7Be #
        ##############################################################################
        # Binding energy of 7Be in 3/2+ and 1/2- channels
        E_plus = 1.5866 # MeV
        E_minus = 1.1575 # MeV

        # Binding momenta of the 3/2+ and 1/2- channels
        self.gamma1_plus = np.sqrt(2 * self.mu * E_plus) / self.h_bar_c
        self.gamma1_minus = np.sqrt(2 * self.mu * E_minus) / self.h_bar_c 

        # Get the kc's
        self.kc_plus = (self.Zb * self.Zt * self.alpha * (self.mu - E_plus)) / self.h_bar_c
        self.kc_minus = (self.Zb * self.Zt * self.alpha * (self.mu - E_minus)) / self.h_bar_c

        # Define the Sommerfeld parameters
        self.eta_B_plus = self.kc_plus / self.gamma1_plus
        self.eta_B_minus = self.kc_minus / self.gamma1_minus

        # Define the H's
        self.H_plus = psi(self.eta_B_plus) + (1 / (2 * self.eta_B_plus)) - np.log(self.eta_B_plus)
        self.H_minus = psi(self.eta_B_minus) + (1 / (2 * self.eta_B_minus)) - np.log(self.eta_B_minus)
        self.H_prime_plus = self.H_prime_eta(-self.eta_B_plus * (1.j))
        self.H_prime_minus = self.H_prime_eta(-self.eta_B_minus * (1.j))
        
        ##############################################################################
        #                               Set the prior params
        ##############################################################################
        # Useful information on the priors
        # All are centered around 1.0 and the sigmas are:
        # som_f_sigmas = np.array([0.064, 0.076, 0.098, 0.057, 0.045, 0.062,
        #                     0.041, 0.077, 0.063, 0.089])
        # barnard_f_sigma = np.array([0.05])
        self.param_bounds = param_prior[:, :2]
        self.prior_gauss_params = param_prior[:, 2:]
        self.f_bounds = norm_prior[:, :2]
        self.prior_gauss_f = norm_prior[:, 2:]

        self.bounds = np.concatenate([self.param_bounds, self.f_bounds])
        self.erp_dim = self.param_bounds.shape[0]
        self.norm_dim = self.f_bounds.shape[0]
        self.total_dim = self.bounds.shape[0] + 2

        # Put everything together in one prior
        self.prior_info = np.concatenate([param_prior, norm_prior])
        


    ##############################################################################
    #                                  Define Utility Functions
    ##############################################################################
    def sqrt(self, x : npt.ArrayLike) -> npt.ArrayLike:
        # Returns a complex value if negative
        return np.where(x >= 0, np.sqrt(x), 1.j * np.sqrt(np.abs(x)))
    
    def cot(self, theta_deg : npt.ArrayLike) -> npt.ArrayLike:
        # Cotangent with angle measured in degrees
        return (np.cos(theta_deg * (np.pi / 180)) /
                        np.sin(theta_deg * (np.pi / 180)))
    
    def ECM(self, Elab : npt.ArrayLike) -> npt.ArrayLike:
        # Convert energy from lab frame to CM frame
        return ((2 * self.mt) / (self.mt + self.mb +
                        self.sqrt(((self.mt + self.mb)**2) +
                        2 * self.mt * Elab))) * Elab  #MeV

    def ELAB(self, Ecm : npt.ArrayLike) -> npt.ArrayLike:
        # Convert energy from CM from to lab frame
        return ((Ecm + (2 * (self.mt + self.mb))) /
                        (2.0 * self.mt)) * Ecm  #MeV 
    
    def kc(self, Elab : npt.ArrayLike) -> npt.ArrayLike:
        # Obtain kc as a function of Elab
        return self.Zb * self.Zt * self.alpha * (
                        (self.mu + self.ECM(Elab)) / self.h_bar_c)  # fm^-1

    def k(self, Elab : npt.ArrayLike) -> npt.ArrayLike:
        # Obtain k as a function of Elab
        return (1 / self.h_bar_c) * self.sqrt(
                        ((self.mu + self.ECM(Elab))**2) - self.mu**2)
    
    def eta(self, Elab : npt.ArrayLike) -> npt.ArrayLike:
        # Obtain eta as a function of Elab
        return self.kc(Elab) / self.k(Elab)
    
    def H(self, Elab : npt.ArrayLike) -> npt.ArrayLike:
        # Obtain H as a function of Elab
        return ((psi((1.j) * self.eta(Elab))) +
                            (1 / (2.j * self.eta(Elab))) -
                            np.log(1.j * self.eta(Elab)))
    
    def h(self, Elab : npt.ArrayLike) -> npt.ArrayLike:
        # Get the real part of H
        return self.H(Elab).real
    
    def C0_2(self, Elab : npt.ArrayLike) -> npt.ArrayLike:
        # Obtain coefficient C0_2 as a function of Elab
        return (2 * np.pi * self.eta(Elab)) / (
                            np.exp(2 * np.pi * self.eta(Elab)) - 1)

    def C1_2(self, Elab : npt.ArrayLike) -> npt.ArrayLike:
        # Obtain coefficient C1_2 as a function of Elab
        return (1 / 9) * (1 + self.eta(Elab)**2) * self.C0_2(Elab)

    def C2_2(self, Elab : npt.ArrayLike) -> npt.ArrayLike:
        # Obtain coefficient C2_2 as a function of Elab
        return (1 / 100) * (4 + self.eta(Elab)**2) * self.C1_2(Elab)
    
    def C3_2(self, Elab : npt.ArrayLike) -> npt.ArrayLike:
        # Obtain coefficient C3_2 as a function of Elab
        return (1 / 441) * (9 + self.eta(Elab)**2) * self.C2_2(Elab)
    
    def H_prime_eta(self, eta : npt.ArrayLike) -> npt.ArrayLike:
        # Obtain the derivative of H with respect to eta
        return derivative(self.H_of_eta, eta, dx = 1e-10)

    def H_of_eta(self, x : npt.ArrayLike) -> npt.ArrayLike:
        # This is the functional form of H(eta)
        return (psi(1.j * x) + 1 / (2 * 1.j * x) - np.log(1.j * x))
    ##############################################################################
    #                                  End Utility Functions
    ##############################################################################

    def rutherford(self) -> npt.ArrayLike:
        """
        Returns the Rutherford cross section at the lab energies and the angles of the data.
        """
        # Convert theta to radians
        theta_rad = self.theta_cs * (np.pi / 180)

        # Get Rutherford amplitude
        f_r = (-self.etavalue_cs / (2 * self.kvalue_cs)) * (
            1 / np.square(np.sin(theta_rad / 2)))**(1 + (1.j) * self.etavalue_cs)
        
        # Convert to mb
        sigma_R = 10. * np.square(np.abs(f_r))
        return sigma_R



    def cs_LO(self) -> npt.ArrayLike:
        """
        Computes the cross section up to leading order relative to Rutherford at data.
        """
        # Convert theta to radians
        theta_rad = self.theta_cs * (np.pi / 180)

        # Define the Legendre polynomial and phase terms
        P_1 = np.cos(theta_rad)
        alpha_1 = np.arcsin(self.etavalue_cs / np.sqrt(1 + np.square(self.etavalue_cs)))
        phase_1 = np.exp((1.j) * 2 * alpha_1)

        # Write out the effective range expansions
        ERE_0 = (-self.Hvalue_cs * 2 * self.kcvalue_cs) / self.C0_2value_cs
        ERE_1_plus = ((2 * self.kcvalue_cs) / (9 * self.C1_2value_cs)) * (
            -(np.square(self.kcvalue_cs) + np.square(self.kvalue_cs)) * self.Hvalue_cs)
        ERE_1_minus = ((2 * self.kcvalue_cs) / (9 * self.C1_2value_cs)) * (
            -(np.square(self.kcvalue_cs) + np.square(self.kvalue_cs)) * self.Hvalue_cs)

        # Get the amplitudes
        f_r = (-self.etavalue_cs / (2 * self.kvalue_cs)) * (
            1 / np.square(np.sin(theta_rad / 2)))**(1 + (1.j) * self.etavalue_cs)

        f_c = f_r + (1 / ERE_0) + (
            np.square(self.kvalue_cs) * phase_1 * P_1 * (
                2 / ERE_1_plus + 1 / ERE_1_minus))

        f_i = np.square(self.kvalue_cs) * phase_1 * np.sin(theta_rad) * (
            1 / ERE_1_minus - 1 / ERE_1_plus)

        # Convert to mb
        sigma = 10. * (np.square(np.abs(f_c)) + np.square(np.abs(f_i)))
        sigma_R = 10. * np.square(np.abs(f_r))

        # Return the cross section relative to Rutherford
        sigma_ratio = sigma / sigma_R
        return sigma_ratio



    def cov_theory(self, c_bar_squared : float, Lambda_B : float) -> npt.ArrayLike:
        """
        Computes the covariance theory matrix.
        """
        # Get the LO calculation in the right shape
        y0 = np.reshape(self.cs_LO_values, (1, len(self.cs_LO_values)))
        
        # Compute the expansion parameter
        Q = self.Q_numerator / Lambda_B
        Q = np.reshape(Q, (1, len(Q)))

        # Generate the theory covariance matrix - - only 1 c_bar_squared should appear here
        cov_theory_cs = (c_bar_squared * y0 * (Q**(self.n_c + 1))).transpose() @ (
            y0 * (Q**(self.n_c + 1))) / (1 - Q.transpose() @ Q) # # Make more explicit with np.outer
        return cov_theory_cs.astype(np.double)
    
    
         
    def set_cov_matrix(self, c_bar_squared : float, Lambda_B : float):
        """
        Constructs and sets the covariance matrix. 
        This also sets the normalization of the log-likelihood.
        """
        theory_piece = self.cov_theory(c_bar_squared, Lambda_B)
        
        if self.use_theory_cov:
            self.cov_matrix = theory_piece + self.exp_piece
        else:
            self.cov_matrix = self.exp_piece
        slog_det = np.linalg.slogdet(self.cov_matrix)

        if slog_det[0] == 0:
            self.Q_rest = -np.inf
        else:
            self.Q_rest = -0.5 * (np.log(2 * np.pi) + slog_det[1])
        


    def cs_theory(self, params : npt.ArrayLike, order : int) -> npt.ArrayLike:
        """
        This is a place holder for the parameterization-dependant cs theory.
        """
        raise NotImplementedError('This must be defined in the subclass!')



    def chi_squared(self, theory : npt.ArrayLike, norm : npt.ArrayLike) -> Tuple[float, npt.ArrayLike]:
        """
        Returns the chi squared and the residual when given theory, data, and norm.
        """
        theory = np.asarray(theory)
        r = norm * theory - self.cs_data
        try:
            chi_2 = np.dot(r.T, np.linalg.solve(self.cov_matrix, r))
            return chi_2, r
        except np.linalg.LinAlgError as e:
            print('Singular covariance matrix:')
            print(self.cov_matrix)
            return np.inf
        


    def lp_flat(self, params : npt.ArrayLike, bounds : npt.ArrayLike = None) -> float:
        """
        Defines the flat uniform prior probability distribution for parameters.
        """
        params_min = min(bounds)
        params_max = max(bounds)
        volume_params = np.prod(params_max - params_min)
        if np.logical_and(params_min <= params, params <= params_max).all():
            return np.log(1 / volume_params)
        else:
            print('Improper Bounds (Log Prior Flat)')
            return -np.inf



    def gaussian(self, x : npt.ArrayLike, center : float, width : float) -> npt.ArrayLike:
        """
        Defines the form and normalization of a Gaussian.
        """
        pdf = (1 / (width * np.sqrt(2 * np.pi))) * (
            np.exp(-0.5 * np.square((x - center) / width)))
        return pdf



    def lp_gauss(self, params : npt.ArrayLike, mu : float, sigma : float, bounds : npt.ArrayLike = None) -> float:
        """
        Defines the Gaussian prior probability distribution.
        """
        params_min = min(bounds)
        params_max = max(bounds)
        if np.logical_and(params_min <= params, params <= params_max).all():
            log_prior_gauss = float(np.log(self.gaussian(params, mu, sigma)))
            return log_prior_gauss
        else:
            print('Improper Bounds (Log Prior Gaussian)')
            return -np.inf
        


    def log_prior_params(self, parameters : npt.ArrayLike) -> float:
        """
        Computes the log-prior of a given set of ERPs and normalization coefficients.
        """
        # Cast the parameters to an array
        parameters = np.array(parameters)

        # If all the parameters fall within the bounds of the prior, compute the prior
        if np.logical_and(self.prior_info[:, 0] <= parameters, parameters <= self.prior_info[:, 1]).all():
            log_p = 0.0

            # Sum over all the contributions to the prior
            for i in range(0, self.total_dim - 2):
                mu, sigma = self.prior_info[i, 2:]
                log_p += self.lp_gauss(parameters[i], mu, sigma, self.prior_info[i, :2])

            return log_p
        else:
            return -np.inf



    def get_c_squared_sum(self, Lambda_B : float, c1_tildes : npt.ArrayLike, c2_tildes : npt.ArrayLike) -> float:
        """
        Returns the sum of the squares of the c values.
        """
        # return np.sum(np.square(c1_tildes * Lambda_B) + np.square(c2_tildes * Lambda_B**2)) # Original code
        # return np.sum(np.square(c1_tildes * Lambda_B)) # Step 2.
        return np.sum(np.square(c1_tildes * Lambda_B) + np.square(c2_tildes * Lambda_B**2)) # Step 3.a + b
    


    def prior_Lambda_B(self, Lambda_B : float) -> float:
        """
        Prior for Lambda_B
        Truncated Gaussian with lower bound being defined by the maximum value in the numerator of Q, upper bound is 4.0.
        Mean: 1.0, sigma: 0.7
        """
        if Lambda_B > np.max(self.Q_numerator) and Lambda_B <= 4.0:
            return 1 / (np.sqrt(2 * np.pi) * 0.7) * np.exp(-np.square((Lambda_B - 1.0) / (2 * 0.7**2)))
        else:
            return 0
        


    def get_Tau(self, Lambda_B : float, c1_tildes : npt.ArrayLike, c2_tildes : npt.ArrayLike) -> float:
        """
        Returns the value of Tau given Lambda_B and a set of c_tildes.
        """
        return np.sqrt((self.nu_0 * self.Tau_0**2 + self.get_c_squared_sum(Lambda_B, c1_tildes, c2_tildes)) / self.nu) # Original code # 3.a
        # return np.sqrt((self.nu_0 * self.Tau_0**2 + self.get_c_squared_sum(Lambda_B, c1_tildes, c2_tildes) / self.cs_data.shape[0]) / self.nu) # Step 3.b
    


    def exp_log_unnorm_Lambda_B(self, Lambda_B : float, c1_tildes : npt.ArrayLike, c2_tildes : npt.ArrayLike) -> float:
        """
        This is the scaled-unnormalized prior on Lambda_B. This function returns the exponential of the log of the prior 
        because it needs to be normalized. We can scale it here because we are normalizing it anyways.
        """
        val = (np.log(self.prior_Lambda_B(Lambda_B)) - self.nu * np.log(self.get_Tau(Lambda_B, c1_tildes, c2_tildes)) - 
               3 * np.sum(np.log(self.Q_numerator / Lambda_B))) - self.lambda_b_norm_scale
        return np.exp(val)



    def log_prior_Lambda_B(self, Lambda_B : float, c1_tildes : npt.ArrayLike, c2_tildes : npt.ArrayLike) -> float:
        """
        Returns the normalized log prior for Lambda_B. This function integrates the scaled unnormalized prior for
        Lambda_B to obtain the normalization coefficient.
        """
        A = quad(self.exp_log_unnorm_Lambda_B, np.max(self.Q_numerator) + 0.00001, 3.999, args = (c1_tildes, c2_tildes))[0]
        if A == 0.0 or self.prior_Lambda_B(Lambda_B) == 0.0:
            return -np.inf
        else:
            return np.log(self.exp_log_unnorm_Lambda_B(Lambda_B, c1_tildes, c2_tildes)) - np.log(A)



    def log_prior_c_bar_squared(self, c_bar_squared : float, Tau : float) -> float:
        """
        The prior for c_bar^2 is the scaled inverse chi^2 distribution. This function returns the log of it. The gamma factor 
        removed for two reasons, first it is constant with respect to the parameters, and two it is not numerically stable.
        """
        return ((self.nu / 2) * np.log(self.nu * Tau**2 / 2) -
                (1 + self.nu / 2) * np.log(c_bar_squared) - (self.nu * Tau**2 / (2 * c_bar_squared)))

    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################

    def log_prior(self, parameters: npt.ArrayLike) -> float:
        """
        Computes log prior. 
        Returns log[ P(c_bar^2 | Lambda_B, theta, I) P(Lambda_B | theta, I) P(theta | I) ]
        """
        # Cast the paramters to an array
        parameters = np.array(parameters)

        # Unpack the parameters
        c_bar_squared = parameters[0]
        Lambda_B = parameters[1]
        params = parameters[2:int(2 + self.erp_dim)]
        params_f = parameters[int(2 + self.erp_dim):]

        # c_bar^2 and Lambda_B must be positive nonzero (small values threaten a singular covariance matrix)
        # and c_bar^2 needs to be not too large (15 is excessive but allows for the sampler to explore larger c_bar^2s)
        if (c_bar_squared <= 0.001) or (Lambda_B <= 0.001) or (c_bar_squared >= 15):
            return -np.inf

        ##############################################################################
        # # # # # P(theta | I)
        ##############################################################################
        params_log_prior = self.log_prior_params(np.concatenate([params, params_f]))

        # ############################################################################## STEP 0. START
        # ##############################################################################
        # ##############################################################################
        # ##############################################################################
        # # In this iteration we decouple the theta dependence in Lambda_B and the c_bar^2
        # # by forcing the prior to be a tight Gaussian around Lambda_B = 200 MeV and c_bar = 0.7
        # # # # # # P(Lambda_B | theta, I) = P(Lambda_B | I)
        # lambda_b_log_prior = self.lp_gauss(Lambda_B, 1.01354, 0.01, [0.001, 3])

        # # # # # # P(c_bar^2 | Lambda_B, theta, I) = P(c_bar^2 | I)
        # c_bar_squared_log_prior = self.lp_gauss(c_bar_squared, 0.49, 0.05, [0.001, 5])

        # ############################################################################## 
        # ##############################################################################
        # ##############################################################################
        # ############################################################################## STEP 0. END

        # ############################################################################## STEP 1. START
        # ##############################################################################
        # ##############################################################################
        # ##############################################################################
        # # # # # # P(Lambda_B | theta, I) = P(Lambda_B | I)
        # lambda_b_log_prior = np.log(self.prior_Lambda_B(Lambda_B))

        # # # # # # P(c_bar^2 | Lambda_B, theta, I)
        # c_bar_squared_log_prior = self.log_prior_c_bar_squared(c_bar_squared, 1.5)
        # ############################################################################## 
        # ##############################################################################
        # ##############################################################################
        # ############################################################################## STEP 1. END

        # ############################################################################## STEP 2. START
        # ##############################################################################
        # ##############################################################################
        # ##############################################################################
        # # # # # # P(Lambda_B | theta, I) = P(Lambda_B | I)
        # # Model evaluations for y1s and y2s
        # y1s = self.cs_theory(params, order = 1)

        # # Define the c_tildes (c_ns without the Lambda_B dependency)
        # c1_tildes = (y1s - self.cs_LO_values) / (self.c_denom_1)
        # c2_tildes = np.zeros(self.cs_LO_values.shape[0])

        # # Set the scale for the Lambda_B prior
        # vals = []
        # Lambda_Bs = np.linspace(np.max(self.Q_numerator) + 0.00001, 4, 100)
        # for Lambda_B in Lambda_Bs:
        #     vals.append(np.log(self.prior_Lambda_B(Lambda_B)) - 
        #                 self.nu * np.log(self.get_Tau(Lambda_B, c1_tildes, c2_tildes)) - 
        #                 3 * np.sum(np.log(self.Q_numerator / Lambda_B)))
        # self.lambda_b_norm_scale = np.max(np.array(vals))

        # lambda_b_log_prior = self.log_prior_Lambda_B(Lambda_B, c1_tildes, c2_tildes)

        # # # # # # P(c_bar^2 | Lambda_B, theta, I)
        # Tau = self.get_Tau(Lambda_B, c1_tildes, c2_tildes)
        # c_bar_squared_log_prior = self.log_prior_c_bar_squared(c_bar_squared, Tau)
        # ##############################################################################
        # ##############################################################################
        # ##############################################################################
        # ############################################################################## STEP 2. END

        ############################################################################## STEP 3.A + B START
        ##############################################################################
        ##############################################################################
        ##############################################################################
        # # # # # P(Lambda_B | theta, I) = P(Lambda_B | I)
        # Model evaluations for y1s and y2s
        y1s = self.cs_theory(params, order = 1)
        y2s = self.cs_theory(params, order = 2)

        # Define the c_tildes (c_ns without the Lambda_B dependency)
        c1_tildes = (y1s - self.cs_LO_values) / (self.c_denom_1)
        c2_tildes = (y2s - y1s) / self.c_denom_2

        # Set the scale for the Lambda_B prior
        vals = []
        Lambda_Bs = np.linspace(np.max(self.Q_numerator) + 0.00001, 4, 100)
        for Lambda_B in Lambda_Bs:
            vals.append(np.log(self.prior_Lambda_B(Lambda_B)) - 
                        self.nu * np.log(self.get_Tau(Lambda_B, c1_tildes, c2_tildes)) - 
                        3 * np.sum(np.log(self.Q_numerator / Lambda_B)))
        self.lambda_b_norm_scale = np.max(np.array(vals))

        lambda_b_log_prior = self.log_prior_Lambda_B(Lambda_B, c1_tildes, c2_tildes)

        # # # # # P(c_bar^2 | Lambda_B, theta, I)
        self.Tau = self.get_Tau(Lambda_B, c1_tildes, c2_tildes)
        c_bar_squared_log_prior = self.log_prior_c_bar_squared(c_bar_squared, self.Tau)
        ##############################################################################
        ##############################################################################
        ##############################################################################
        ############################################################################## STEP 3.A + B END


        # ############################################################################## ORIGINAL CODE START
        # # # # # # P(Lambda_B | theta, I)
        # ##############################################################################
        # # Model evaluations for y1s and y2s
        # y1s = self.cs_theory(params, order = 1)
        # y2s = self.cs_theory(params, order = 2)

        # # Define the c_tildes (c_ns without the Lambda_B dependency)
        # c1_tildes = (y1s - self.cs_LO_values) / (self.c_denom_1)
        # c2_tildes = (y2s - y1s) / self.c_denom_2

        # # Set the scale for the Lambda_B prior
        # vals = []
        # Lambda_Bs = np.linspace(np.max(self.Q_numerator) + 0.00001, 4, 100)
        # for Lambda_B in Lambda_Bs:
        #     vals.append(np.log(self.prior_Lambda_B(Lambda_B)) - 
        #                 self.nu * np.log(self.get_Tau(Lambda_B, c1_tildes, c2_tildes)) - 
        #                 3 * np.sum(np.log(self.Q_numerator / Lambda_B)))
        # self.lambda_b_norm_scale = np.max(np.array(vals))

        # lambda_b_log_prior = self.log_prior_Lambda_B(Lambda_B, c1_tildes, c2_tildes)
        # ##############################################################################
        # # # # # # P(c_bar^2 | Lambda_B, theta, I)
        # ##############################################################################
        # Tau = self.get_Tau(Lambda_B, c1_tildes, c2_tildes)
        # c_bar_squared_log_prior = self.log_prior_c_bar_squared(c_bar_squared, Tau) # ORIGINAL CODE END

        # Combine the sum of log pieces
        return params_log_prior + lambda_b_log_prior + c_bar_squared_log_prior



    def log_likelihood(self, parameters : npt.ArrayLike) -> float:
        """
        Returns the log-likelihood of the parameters.
        """
        # Cast the paramters to an array
        parameters = np.array(parameters)

        # Unpack the parameters
        c_bar_squared = parameters[0]
        Lambda_B = parameters[1]
        params = parameters[2:int(2 + self.erp_dim)]
        params_f = parameters[int(2 + self.erp_dim):]

        # c_bar^2 and Lambda_B must be positive nonzero (small values threaten a singular covariance matrix)
        if (c_bar_squared <= 0.001) or (Lambda_B <= max(self.Q_numerator)):
            return -np.inf
        
        # Construct the covariance matrix
        self.set_cov_matrix(c_bar_squared, Lambda_B)

        # Perform model evaluations for y2s and get normalizations
        y2s = self.cs_theory(params, order = 2)
        norm = params_f[self.norm_grouping.astype(int)]

        # Compute the chi squared
        chi2 = self.chi_squared(y2s, norm)[0]

        return -0.5 * chi2 + self.Q_rest
    


    def log_posterior(self, parameters : npt.ArrayLike) -> float:
        """
        Returns the log-posterior of a set of parameters.
        """
        return self.log_prior(parameters) + self.log_likelihood(parameters)


    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################

def main():
    E_min = 0.676 # MeV
    E_max = 2.624 # MeV
    which_data = 'som'

    loader = DataLoader(E_min, E_max, which_data)
    data = loader.get_data()
    norm_grouping = loader.get_normalization_grouping()

    # Set up the priors
    param_bounds = np.array([[-0.02, 0.06], [-3, 3], [5.0, 25.0], [-6, 6], [5.0, 25.0], [-6, 6]])
    params_prior = np.array([[0.025, 0.015], [0.8, 0.4], [13.84, 1.63], [0.0, 1.6], [12.59, 1.85], [0.0, 1.6]]) # center, width
    
    gauss_prior_params = np.hstack([param_bounds, params_prior])
    gauss_prior_f = loader.get_normalization_prior_info()

    # Set up the base model
    import sim_models
    model = sim_models.Sim_BS_C(data, norm_grouping, gauss_prior_params, gauss_prior_f, True)
    test = np.concatenate([[1.0], [0.8], gauss_prior_params[:, 2], gauss_prior_f[:, 2]])

    with cProfile.Profile() as profile:
        for i in range(0, 1000):
            model.log_posterior(test)

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()

if __name__ == "__main__":
    main()