# Useful imports
import numpy as np
import sys; sys.path.append('./')
import models
from data_loader import DataLoader
import emcee, corner, os
# from sklearn.covariance import EmpiricalCovariance
import numpy.linalg as lin
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(directory):
    # Find the README, open in context
    with open(directory + '/READ_ME.txt', 'r') as f:
        content = f.read()

        # Extract the info from the READ_ME
        E_min = float(content[int(content.find('E_min ') + 6):int(content.find('E_min ') + 6) + 5])
        E_max = float(content[int(content.find('E_max ') + 6):int(content.find('E_max ') + 6) + 5])
        n_burn = int(content[int(content.find(' steps, ') + 7):int(content.find(' burn in'))])
        n_steps = int(content[int(content.find('Run with ') + 8):int(content.find(' steps,'))])
        which_data = content[int(content.find('Analyzing ') + 9):int(content.find(' data with'))]
        parameterization = content[int(content.find('model: ') + 6):int(content.find('\n'))]
        use_theory_cov = content[int(content.find('covariance: ') + 11):int(content.find('\n'))]

    print(E_min, E_max, n_burn, n_steps, which_data, parameterization, use_theory_cov)
    sys.exit(-1)
    # Get the file name
    file_name = [i for i in os.listdir(directory) if i.endswith('.h5')][0]
    
    # # # # # Run the analysis from the notebook

    # Load the corresponding data
    loader = DataLoader(E_min, E_max, which_data)
    data = loader.get_data()
    norm_group = loader.get_normalization_grouping()
    gauss_prior_f = loader.get_normalization_prior_info()
    
    # # # Set the parameter bounds and initialize the model
    if parameterization == 'standard':
        labels = ['A0', 'r0', 'A1+', 'r1+', 'P1+', 'A1-', 'r1-', 'P1-']
        sys.stderr.write('No Longer Supported....')
        sys.exit(-1)
    elif parameterization == 'bound_state':
        labels = ['A0', 'r0', 'r1+', 'P1+', 'r1-', 'P1-']
        param_bounds = np.array([[-0.02, 0.06], [-3, 3], [-1, 1], [-6, 6], [-1, 1], [-6, 6]])
        params_prior = np.array([[0.025, 0.015], [0.8, 0.4], [0.0, 0.1], [0.0, 1.6], [0.0, 0.1], [0.0, 1.6]]) # center, width
        gauss_prior_params = np.hstack([param_bounds, params_prior])
        sys.stderr.write('Not implemented yet...')
        sys.exit(-1)
    elif parameterization == 'bs_C':
        labels = ['A0', 'r0', 'C1+^2', 'P1+', 'C1-^2', 'P1-']
        param_bounds = np.array([[-0.02, 0.06], [-3, 3], [5.0, 25.0], [-6, 6], [5.0, 25.0], [-6, 6]])
        params_prior = np.array([[0.025, 0.015], [0.8, 0.4], [13.84, 1.63], [0.0, 1.6], [12.59, 1.85], [0.0, 1.6]]) # center, width
        gauss_prior_params = np.hstack([param_bounds, params_prior])
        model = models.BS_C(data, norm_group, gauss_prior_params, gauss_prior_f, use_theory_cov)
    elif parameterization == 'init-f-wave':
        labels = ['A0', 'r0', 'C1+^2', 'P1+', 'C1-^2', 'P1-', 'r3+']
        param_bounds = np.array([[-0.02, 0.06], [-3, 3], [5.0, 25.0], [-6, 6], [5.0, 25.0], [-6, 6], [-3, 0]])
        params_prior = np.array([[0.025, 0.015], [0.8, 0.4], [13.84, 1.63], [0.0, 1.6], [12.59, 1.85], [0.0, 1.6], [-0.5, 1]]) # center, width
        gauss_prior_params = np.hstack([param_bounds, params_prior])
        model = models.F_Wave_AR(data, norm_group, gauss_prior_params, gauss_prior_f, use_theory_cov)

    # Add normalization labels
    for i in range(0, int(np.max(norm_group) + 1)):
        labels.append('f_{}'.format(i))
        
    
    # Load in the samples
    reader = emcee.backends.HDFBackend(file_name)
    samples_not_flat = reader.get_chain()
    samples_flat = reader.get_chain(flat = True)

    # Look at the variances within each chain to determine if the walker is moving enough or if it is stuck.
    within_chain_means = np.mean(samples_not_flat[n_burn:, :, :], axis = 0)

    # Create an empty array of the within chain variances
    within_chain_var = np.empty(within_chain_means.shape)

    # Run a for loop across all walkers to compute the within chain variance
    for i in range(0, within_chain_means.shape[0]):
        within_chain_var[i, :] = np.sum(np.square(within_chain_means[i, :] - samples_not_flat[n_burn:, i, :]), axis = 0) / (samples_not_flat.shape[0] // 2)

    # Get the typical within chain variance W for each parameter
    W = np.median(within_chain_var, axis = 0)


    # Now we need to loop over each chain for each parameter to see how it compares to the typical variance
    bad_indices = []
    ratios = np.empty(within_chain_means.shape)
    # Loop over each parameter
    for i in range(0, within_chain_means.shape[1]):
        # Loop over the walkers
        for j in range(0, within_chain_means.shape[0]):
            ratio = np.sum(within_chain_var[j, i] / W[i]) / within_chain_means.shape[1]
            ratios[j, i] = ratio

    # Sum along each parameter, this value should be very close to 1.0. Select out the bad indices
    total_normalized_ratios = np.sum(ratios, axis = 1)
    bad_indices = np.where(total_normalized_ratios <= 0.9)[0]
    print('Found {} bad walkers at indices:'.format(bad_indices.shape[0]))
    print(bad_indices)

    if bad_indices.shape[0] != 0:
        # Remove the bad walkers
        samples_not_flat = np.delete(samples_not_flat, bad_indices, axis = 1)

    # Thin according to the burn-in time
    thinned_samples_not_flat = samples_not_flat[n_burn:, :, :]

    # Compute the autocorrelation times for each parameter
    ac_s = reader.get_autocorr_time(discard = n_burn, tol = 0)
    ac = int(np.max(ac_s))

    # Thin according to the autocorrelation time
    thinned_samples_not_flat = thinned_samples_not_flat[::ac, :, :]

    # Flatten the samples and log-prob
    len0, len1, len2 = thinned_samples_not_flat.shape
    samples = np.reshape(thinned_samples_not_flat, (len0 * len1, len2))

    # Generate the corner plot
    corner.corner(samples[:, :model.erp_dim], labels = labels[:model.erp_dim], quantiles = [0.16, 0.5, 0.84], title_fmt = '.4f', show_titles = True)
    plt.suptitle("E_min = {} MeV, E_max = {} MeV, {} data".format(E_min, E_max, which_data))
    plt.savefig('corner_plot.png')





if __name__ == "__main__":
    directory = sys.argv[1]
    process(directory)