#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:17:48 EST 2024

@author: Andrius Burnelis
"""
import numpy as np, warnings
import sys; sys.path.append('./')
from data_loader import DataLoader
from scipy.special import psi, gamma
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.linalg import det
import os, tqdm, pickle
from numpy import typing as npt


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
        self.n_c = 2
        self.Tau_0 = 1.5
        self.nu_0 = 1.5

        self.Tau = 1.0 
        self.nu = self.nu_0 + (self.cs_data.shape[0] * self.n_c)


        # Store the numerator of Q
        # Get the momentum transfer
        q = 2.0 * self.kvalue_cs * np.sin((self.theta_cs * np.pi / 180) / 2.0)
        self.Q_numerator = np.array([max(q[t], self.kvalue_cs[t])
                    for t in range(len(self.theta_cs))]).real
        
        self.use_theory_cov = use_theory_cov
        self.cov_matrix = None
        self.inverse_cov_matrix = None
        self.Q_rest = None
        self.lambda_b_norm_scale = None

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
    def sqrt(self, x):
        # Returns a complex value if negative
        return np.where(x >= 0, np.sqrt(x), 1.j * np.sqrt(np.abs(x)))
    
    def cot(self, theta_deg):
        # Cotangent with angle measured in degrees
        return (np.cos(theta_deg * (np.pi / 180)) /
                        np.sin(theta_deg * (np.pi / 180)))
    
    def ECM(self, Elab):
        # Convert energy from lab frame to CM frame
        return ((2 * self.mt) / (self.mt + self.mb +
                        self.sqrt(((self.mt + self.mb)**2) +
                        2 * self.mt * Elab))) * Elab  #MeV

    def ELAB(self, Ecm):
        # Convert energy from CM from to lab frame
        return ((Ecm + (2 * (self.mt + self.mb))) /
                        (2.0 * self.mt)) * Ecm  #MeV 
    
    def kc(self, Elab):
        # Obtain kc as a function of Elab
        return self.Zb * self.Zt * self.alpha * (
                        (self.mu + self.ECM(Elab)) / self.h_bar_c)  # fm^-1

    def k(self, Elab):
        # Obtain k as a function of Elab
        return (1 / self.h_bar_c) * self.sqrt(
                        ((self.mu + self.ECM(Elab))**2) - self.mu**2)
    
    def eta(self, Elab):
        # Obtain eta as a function of Elab
        return self.kc(Elab) / self.k(Elab)
    
    def H(self, Elab):
        # Obtain H as a function of Elab
        return ((psi((1.j) * self.eta(Elab))) +
                            (1 / (2.j * self.eta(Elab))) -
                            np.log(1.j * self.eta(Elab)))
    
    def h(self, Elab):
        # Get the real part of H
        return self.H(Elab).real
    
    def C0_2(self, Elab):
        # Obtain coefficient C0_2 as a function of Elab
        return (2 * np.pi * self.eta(Elab)) / (
                            np.exp(2 * np.pi * self.eta(Elab)) - 1)

    def C1_2(self, Elab):
        # Obtain coefficient C1_2 as a function of Elab
        return (1 / 9) * (1 + self.eta(Elab)**2) * self.C0_2(Elab)

    def C2_2(self, Elab):
        # Obtain coefficient C2_2 as a function of Elab
        return (1 / 100) * (4 + self.eta(Elab)**2) * self.C1_2(Elab)
    
    def C3_2(self, Elab):
        # Obtain coefficient C3_2 as a function of Elab
        return (1 / 441) * (9 + self.eta(Elab)**2) * self.C2_2(Elab)
    
    def H_prime_eta(self, eta):
        # Obtain the derivative of H with respect to eta
        return derivative(self.H_of_eta, eta, dx = 1e-10)

    def H_of_eta(self, x):
        # This is the functional form of H(eta)
        return (psi(1.j * x) + 1 / (2 * 1.j * x) - np.log(1.j * x))
    ##############################################################################
    #                                  End Utility Functions
    ##############################################################################

    def rutherford(self):
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



    def cs_LO(self):
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



    def cov_theory(self, c_bar_squared, Lambda_B):
        """
        Computes the covariance theory matrix.
        """
        # Get the LO calculation in the right shape
        y0 = np.reshape(self.cs_LO_values, (1, len(self.cs_LO_values)))
        
        # Compute the expansion parameter
        Q = self.Q_numerator / Lambda_B
        Q = np.reshape(Q, (1, len(Q)))

        # Generate the theory covariance matrix 
        cov_theory_cs = (c_bar_squared * y0 * (Q**(self.n_c + 1))).transpose() @ (
            c_bar_squared * y0 * (Q**(self.n_c + 1))) / (1 - Q.transpose() @ Q) # # Make more explicit with np.outer
        return cov_theory_cs.astype(np.double)
    
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################

    def set_cov_matrix(self, c_bar_squared, Lambda_B):
        theory_piece = self.cov_theory(c_bar_squared, Lambda_B)
        exp_piece = np.diag(self.err_cs**2)
        if self.use_theory_cov:
            self.cov_matrix = theory_piece + exp_piece
        else:
            self.cov_matrix = exp_piece
        self.inverse_cov_matrix = np.linalg.inv(self.cov_matrix)
        eigenvals = np.linalg.eigvalsh(self.cov_matrix)
        # print('eigenvals: {}'.format(eigenvals))

        if np.any(eigenvals == 0):
            self.Q_rest = -np.inf
        else:
            log_det = np.sum(np.log(eigenvals))
            self.Q_rest = -0.5 * (np.log(2 * np.pi) + log_det)

        # print(log_det)
        return True
        

    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################



    def cs_theory(self, params, order):
        """
        This is a place holder for the parameterization-dependant cs theory.
        """
        raise NotImplementedError('This must be defined in the subclass!')



    def chi_squared(self, theory, norm):
        """
        Returns the chi squared and the residual when given theory, data, and norm.
        """
        theory = np.asarray(theory)
        r = norm * theory - self.cs_data
        chi_2 = r.transpose() @ self.inverse_cov_matrix @ r
        return chi_2, r



    def lp_flat(self, params, bounds = None):
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



    def gaussian(self, x, center, width):
        """
        Defines the form and normalization of a Gaussian.
        """
        pdf = (1 / (width * np.sqrt(2 * np.pi))) * (
            np.exp(-0.5 * np.square((x - center) / width)))
        return pdf



    def lp_gauss(self, params, mu, sigma, bounds = None):
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
        

    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################


    def set_lambda_b_norm_scale(self):
        vals = []
        Lambda_Bs = np.linspace(np.max(self.Q_numerator) + 0.00001, 4, 100)
        for Lambda_B in Lambda_Bs:
            vals.append(np.log(self.prior_Lambda_B(Lambda_B)) - self.nu * np.log(self.Tau) - 3 * np.sum(np.log(self.Q_numerator / Lambda_B)))
        self.lambda_b_norm_scale = np.max(vals)

    def prior_Lambda_B(self, Lambda_B):
        return np.where(
            np.logical_and(np.logical_and(Lambda_B >= 0, Lambda_B <= 4.0), 
                           Lambda_B > np.max(self.Q_numerator)), 
                           1 / (np.sqrt(2 * np.pi) * 0.7) * np.exp(-np.square((Lambda_B - 1.0) / (2 * 0.7**2))), 0)

    def exp_log_unnorm_Lambda_B(self, Lambda_B):
        val = (np.log(self.prior_Lambda_B(Lambda_B)) - self.nu * np.log(self.Tau) - 3 * np.sum(np.log(self.Q_numerator / Lambda_B))) - self.lambda_b_norm_scale
        return np.exp(val)
    
    def exp_log_unnorm_Lambda_B_2(self, Lambda_B):
        val = np.log(self.prior_Lambda_B(Lambda_B)) - self.nu * np.log(self.Tau) - 3 * np.sum(np.log(self.Q_numerator / Lambda_B))
        return np.exp(val)

    def log_prior_Lambda_B(self, Lambda_B):
        A = quad(self.exp_log_unnorm_Lambda_B, np.max(self.Q_numerator) + 0.00001, 3.999)[0]
        # print('\nA: {}\n'.format(A))
        return ((np.log(self.prior_Lambda_B(Lambda_B)) - self.nu * np.log(self.Tau) - 
                 3 * np.sum(np.log(self.Q_numerator / Lambda_B))) - self.lambda_b_norm_scale - np.log(A))

    def log_prior_c_bar_squared(self, c_bar_squared):
        # Explicitly write out the log
        # return np.log(1 / c_bar_squared**(0.5 * (self.n_c + 2 + self.nu_0)) * np.exp(-(self.nu * self.Tau**2) / 2 * c_bar_squared))
        return -(0.5 * (self.n_c + 2 + self.nu_0)) * np.log(c_bar_squared) - ((self.nu * self.Tau**2) / 2 * c_bar_squared)


    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################

    def log_prior(self, parameters):
        """
        Computes the log-prior of a given set of parameters.
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



    def log_likelihood(self, parameters):
        """
        Determines the log-likelihood of a set of parameters.
        """
        # Cast the parameters to an array
        parameters = np.array(parameters)

        # Unpack the erps and the normalization parameters
        params = parameters[:self.erp_dim]
        params_f = parameters[self.erp_dim:]

        # If the parameters are within the bounds, compute the log-likelihood
        if np.logical_and(self.bounds[:, 0] <= parameters, parameters <= self.bounds[:, 1]).all():
            # Get the normalizations and theory
            norm = params_f[self.norm_grouping.astype(int)]
            theory = self.cs_theory(params, 2)

            # Compute the chi squared (data is already in chi_squared)
            chi2 = self.chi_squared(theory, norm)[0]

            # Compute and return the log-likelihood
            log_L = -0.5 * chi2 + self.Q_rest
            return log_L
        else:
            return -np.inf
        

    def log_posterior(self, parameters):
        """
        Drives the simultaneous sampling for c_bar^2, Lambda_B, ERPs, and f's.
        """
        # Cast the paramters to an array
        parameters = np.array(parameters)

        # Unpack the parameters
        c_bar_squared = parameters[0]
        Lambda_B = parameters[1]
        params = parameters[2:int(2 + self.erp_dim)]
        params_f = parameters[int(2 + self.erp_dim):]

        # print('\n\nc_bar^2: {}\nlambda_B: {}\nparams: {}\nparams_f: {}\n\n'.format(c_bar_squared, Lambda_B, params, params_f))

        # Perform model evaluations to get y1s, y2s
        y1s = self.cs_theory(params, order = 1)
        y2s = self.cs_theory(params, order = 2)

        # Compute c1s, c2s
        c1s = (y1s - self.cs_LO_values) / (self.cs_LO_values * (self.Q_numerator / Lambda_B))
        c2s = (y2s - y1s) / (self.cs_LO_values * np.square((self.Q_numerator / Lambda_B)))

        # Update the hyper parameter Tau
        self.Tau = np.sqrt((self.nu_0 * self.Tau_0**2 + np.sum(np.square(c1s) + np.square(c2s))) / self.nu)

        # Set the new covariance matrix (self.cov_matrix, self.inverse_cov_matrix, and self.Q_rest)
        self.set_cov_matrix(c_bar_squared, Lambda_B)

        # Get log prior on the ERPs + params_f
        params_log_prior = self.log_prior(np.concatenate([params, params_f]))

        # Get the log "prior" P(Lambda_B | theta, I)
        self.set_lambda_b_norm_scale()
        lambda_b_log_prior = self.log_prior_Lambda_B(Lambda_B)

        # Compute the log "prior" P(c_bar^2 | Lambda_B, theta, I)
        c_bar_squared_log_prior = self.log_prior_c_bar_squared(c_bar_squared)

        # Compute the log likelihood P(D | theta, c_bar^2, Lambda_B, I)
        LL = self.log_likelihood(np.concatenate([params, params_f]))

        # print('LL: {}\nc_bar^2: {}\nlambda_B: {}\nparams: {}\n\n'.format(LL, c_bar_squared_log_prior, lambda_b_log_prior, params_log_prior))

        # Combine the pieces for total log posterior
        log_post_pieces = np.array([LL, c_bar_squared_log_prior, lambda_b_log_prior, params_log_prior])

        # Set any possible NaN's to -inf
        log_post_pieces = np.nan_to_num(log_post_pieces, nan = -np.inf)

        return np.sum(log_post_pieces)




def main():
    E_min = 1.269 # MeV
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
    base = SimBaseModel(data, norm_grouping, gauss_prior_params, gauss_prior_f, True)
    test = np.concatenate([[20], [100], gauss_prior_params[:, 2], gauss_prior_f[:, 2]])
    base.log_posterior(test)

if __name__ == "__main__":
    main()