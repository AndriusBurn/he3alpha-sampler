#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:29:06 EDT 2024

@author: Andrius Burnelis
"""
import numpy as np, sys, emcee
sys.path.append('./')
from data_loader import DataLoader
import sim_models
import os, tqdm, warnings
import multiprocess, time
from scipy.stats import truncnorm
import multiprocessing as mp
import cProfile, pstats

warnings.simplefilter('ignore')

def main():
    ##############################################################################
    ##############################################################################
    # Code to change between runs
    ##############################################################################
    ##############################################################################
    
    # # Select the data subsets to use
    # # E_min : [0.676, 0.84 , 1.269, 1.741, 2.12 , 2.609, 2.609, 3.586, 4.332, 5.475]
    # # E_max : [0.706, 0.868, 1.292, 1.759, 2.137, 2.624, 2.624, 3.598, 4.342, 5.484]
    E_mins = np.array([0.676])
    E_maxes = np.array([2.624])
    which_datas = ['som']

    # Select the model
    parameterizations = ['sim_bs_C']

    # Parameters of sampling
    n_steps = [100000]
    n_burns = [10000]

    # Use theory cov?
    use_theory_covs = [True]

    # Always write a comment for the run (at least just '\n')!
    comments = ['Second run for results\n']

    # # # Optional # # #
    # Set a specific prior? (Default set: None)
    params_bounds = [None]
    params_priors = [None]

    ##############################################################################
    ##############################################################################
    # End code to change between runs
    ##############################################################################
    ##############################################################################


    ##############################################################################
    # # # Create the directory structure to keep track of samples

    # Get the time and format the parent directory
    current_time = time.strftime("%m_%d_%Y")
    parent_directory = './' + current_time

    # Create the parent directory with os and get the max starting number
    if not os.path.exists(parent_directory):
        os.mkdir(parent_directory)
        start_count = 0
    else:
        current_sub_dirs = os.listdir(parent_directory)
        start_count = max([int(''.join(filter(str.isdigit, x))) for x in current_sub_dirs]) + 1

    # # Generate a list of subdirectories
    subdirectories = ["run{}".format(i + start_count) for i in range(0, E_mins.shape[0])]

    # Create the subdirectories with os
    for subdirectory in subdirectories:
        if not os.path.exists(parent_directory + '/' + subdirectory):
            os.mkdir(parent_directory + '/' + subdirectory)

    # Generate the list of save names
    save_names = ["{}/{}/samples_{}.h5".format(parent_directory, subdirectories[i], i + start_count) for i in range(0, E_mins.shape[0])]
    ##############################################################################

    ##############################################################################
    # # # Do a run for each element in the user defined lists
    for i in range(0, E_maxes.shape[0]):
        # Set the run variables
        E_min = E_mins[i]
        E_max = E_maxes[i]
        which_data = which_datas[i]
        parameterization = parameterizations[i]
        use_theory_cov = use_theory_covs[i]
        n_step = n_steps[i]
        n_burn = n_burns[i]
        comment = comments[i]
        save_name = save_names[i]

        # Handle if we do not specify priors
        try:
            param_bounds = params_bounds[i]
            params_prior = params_priors[i]
        except IndexError as e:
            param_bounds = None
            params_prior = None

        # # # Load in the data
        loader = DataLoader(E_min, E_max, which_data)
        data = loader.get_data()
        norm_group = loader.get_normalization_grouping()
        gauss_prior_f = loader.get_normalization_prior_info()

        # # # Set up the READ_ME
        with open(parent_directory + '/' + subdirectories[i] + '/READ_ME.txt', '+w') as f:
            f.write("Choice of model: {}\n".format(parameterization))
            f.write("Analyzing {} data with E_min {} MeV and E_max {} MeV\n".format(which_data, E_min, E_max))
            f.write("Theory covariance: {}\n\nCustom Prior Bounds: \n{}\n\nCustom Prior Params: \n{}\n\n".format(use_theory_cov, param_bounds, params_prior))

        # # # Set the parameter bounds and initialize the model
            
        if parameterization == 'sim_bs_C':
            if param_bounds is None:
                param_bounds = np.array([[-0.02, 0.06], [-3, 3], [5.0, 25.0], [-6, 6], [5.0, 25.0], [-6, 6]])
            if params_prior is None:
                params_prior = np.array([[0.025, 0.015], [0.8, 0.4], [13.84, 1.63], [0.0, 1.6], [12.59, 1.85], [0.0, 1.6]]) # center, width
            gauss_prior_params = np.hstack([param_bounds, params_prior])
            model = sim_models.Sim_BS_C(data, norm_group, gauss_prior_params, gauss_prior_f, use_theory_cov)
        else:
            sys.stderr.write('Nothing else is implemented yet...')
            sys.exit(-1)
            
        # # # Set up the MCMC parameters, walkers, burn in, steps, etc...
        n_walkers = int(model.total_dim * 2)

        # # # Continue with the README
        with open(parent_directory + '/' + subdirectories[i] + '/READ_ME.txt', 'a') as f:
            f.write("Run with {} steps, {} burn in and {} walkers.\n".format(n_step, n_burn, n_walkers))
            f.write("Custom comment: \n{}\n".format(comment))

        ##############################################################################

        # # # Initialize the starting samples (according to the prior)
        starting_samples = []
        # Set the starting positions of c_bar_squared and Lambda_B
        starting_samples.append(np.random.chisquare(model.nu_0, size = n_walkers)) # c_bar_squared
        starting_samples.append(np.random.uniform(0.75, 3.99, size = n_walkers)) # Lambda_B

        # Set the starting positions of ERPs + normalization params
        for j in range(0, model.total_dim - 2):
            min_bound, max_bound, mu, sigma = model.prior_info[j]
            lower = (min_bound - mu) / sigma
            upper = (max_bound - mu) / sigma
            starting_samples.append(truncnorm.rvs(lower, upper, loc = mu, scale = sigma, size = n_walkers))

        # Cast to an array
        starting_samples = np.column_stack(starting_samples)

        # # # Set up the backend
        save_name = save_names[i]
        backend = emcee.backends.HDFBackend(save_name)
        backend.reset(n_walkers, model.total_dim)
          
        # Useful output statements
        sys.stdout.write('Starting run with {} data {} - {} MeV\n'.format(which_data, E_min, E_max))
        sys.stdout.write('MCMC sampling using emcee (affine invariant ensamble sampler) with {} walkers and {} steps\n'.format(n_walkers, n_step))
        N = data.shape[0]
        sys.stdout.write('The number of input data points are: {}\nThe number of parameters are: {}\n'.format(N, model.total_dim))
        # sys.stdout.write('Sampling will be split across {} cores.\n'.format(int(cpu_count - cpu_save)))
        # # # Write useful information to the README
        with open(parent_directory + '/' + subdirectories[i] + '/READ_ME.txt', 'a') as f:
            f.write('\n\nThe number of input data points are: {}\nThe number of parameters are: {}\n'.format(N, model.total_dim))
            f.write('Started burn in at {}\n'.format(time.ctime()))
        
        # # # Initialize the emcee ensemble sampler (Without multiprocessing)
        sampler = emcee.EnsembleSampler(n_walkers, model.total_dim, 
                    model.log_posterior, 
                    # moves = [(emcee.moves.DEMove(), 0.5), (emcee.moves.DESnookerMove(), 0.5)],
                    backend = backend)
        
        # # # Run the burnin and sample
        # Execute the burn in
        pos, prob, state = sampler.run_mcmc(starting_samples, n_burn, progress = True)
        sys.stdout.write('******************** Getting Samples ********************\n')
        
        with open(parent_directory + '/' + subdirectories[i] + '/READ_ME.txt', 'a') as f:
            f.write('Finished burn in - Starting run at {}\n'.format(time.ctime()))

        # Run the sampler for n_step
        run = sampler.run_mcmc(pos, n_step, progress = True)

        with open(parent_directory + '/' + subdirectories[i] + '/READ_ME.txt', 'a') as f:
            f.write('Finished run at {}\n'.format(time.ctime()))

if __name__ == "__main__":
    main()