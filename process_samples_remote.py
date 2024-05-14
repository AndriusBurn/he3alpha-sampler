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

def process(directory, plot_cs = True):
    # Find the README, open in context
    with open(directory + '/READ_ME.txt', 'r') as f:
        content = f.read()

        # Extract the info from the READ_ME
        E_min = float(content[int(content.find('E_min ') + 6):int(content.find('E_min ') + 6) + 5])
        E_max = float(content[int(content.find('E_max ') + 6):int(content.find('E_max ') + 6) + 5])
        n_burn = int(content[int(content.find(' steps, ') + 7):int(content.find(' burn in'))])
        n_steps = int(content[int(content.find('Run with ') + 8):int(content.find(' steps,'))])
        which_data = content[int(content.find('Analyzing ') + 10):int(content.find(' data with'))]
        parameterization = content[int(content.find('model: ') + 7):int(content.find('\n'))]
        use_theory_cov = content[int(content.find('covariance: ') + 11):int(content.find('\n'))]

    # Get the file name
    file_name = directory + '/' + [i for i in os.listdir(directory) if i.endswith('.h5')][0]

    # print(E_min, E_max, n_burn, n_steps, which_data, parameterization, use_theory_cov, file_name)    
    
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
    elif parameterization == 'initial_f_wave':
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
    plt.savefig(directory + '/corner_plot.png')


    # Set up root path for the data
    root_path = './'

    # Set up the list of paths to the data
    barnard_paths = [root_path + 'barnard_data/barnard5477.txt',
                    root_path + 'barnard_data/barnard6345.txt',
                    root_path + 'barnard_data/barnard7395.txt',
                    root_path + 'barnard_data/barnard9003.txt',
                    root_path + 'barnard_data/barnard10460.txt',
                    root_path + 'barnard_data/barnard11660.txt',
                    root_path + 'barnard_data/barnard12530.txt',
                    root_path + 'barnard_data/barnard14080.txt']

    som_paths = [root_path + 'SOM/som_cm_int1.npy',
                root_path + 'SOM/som_cm_int2.npy',
                root_path + 'SOM/som_cm_int3.npy']


    data_E_max = E_max
    data_which_data = which_data


    # Load in the data, specific to the set and the E_max

    # Handle Barnard set first
    barnard_data_list = []
    for path in barnard_paths:
        barnard_data_list.append(np.loadtxt(path))

    # Handle Som set
    som_data_list = []
    l_som = [] # For Som normalization
    l_som_energies = []
    for path in som_paths:
        data_set = np.load(path, allow_pickle = True)
        som_data_list.append(data_set)
        l_som_energies.append([data_set[i][:, 0][0] for i in range(len(data_set))])
        l_som.append([len(data_set[j]) for j in range(len(data_set))])

    # Concatenate the datasets into arrays
    barnard_data = np.concatenate(barnard_data_list)
    som_temp = np.concatenate(np.concatenate(som_data_list)) 
    l_som_energies = np.array(l_som_energies)
    # 2x concatenate because it is a list of lists

    # For some reason, the Som data is formatted [E, theta, cs, err] so I swap the
    # columns to match the Barnard set followint [theta, E, cs, err]
    som_data = np.column_stack([som_temp[:, 1], som_temp[:, 0], som_temp[:, 2], som_temp[:, 3]])

    # Cap the energy at E_max
    # NOTE: The Barnard data has lowest E at 2.439 MeV and the Som data has
    # lowest E at 0.676 MeV
    if data_E_max != None:
        barnard_indices = np.where(barnard_data[:, 1] <= data_E_max)
        som_indices = np.where(som_data[:, 1] <= data_E_max)
        barnard_data = barnard_data[barnard_indices]
        som_data = som_data[som_indices]

    # Now select which data to use
    if data_which_data == 'both':
        data = np.concatenate([barnard_data, som_data])
    elif data_which_data == 'som':
        data = som_data
    elif data_which_data == 'barnard':
        data = barnard_data
    else:
        sys.stderr.write('Choose a \'which_data\': both, som, barnard...')
        sys.exit(-1)


    if plot_cs:  
        # # NOTE: Plotting only works currently for som energies

        # Set up the values for plotting
        theta_vals = np.linspace(40, 140, 100)
        param_set = samples[:, :model.erp_dim]
        normalizations = samples[:, model.erp_dim:]

        # Plot every plot_skip-th parameter (This speeds up the calculation by a bit)
        plot_skip = 25
        param_set = param_set[::plot_skip]

        # Plotting energies are at the 'nominal' energies between the lowest and highest energy of the interaction regions
        plotting_energy_pool = np.array([0.691, 0.854, 1.2803, 1.750, 2.129, 2.6163, 2.6163, 3.592, 4.337, 5.4797])

        plot_numbers = np.where(plotting_energy_pool <= E_max)[0]

        # l_som_energies = np.array([[0.676, 0.84 , 1.269, 1.741, 2.12 , 2.609, 3.586, 4.332, 5.475],
        #                            [0.691, 0.854, 1.28 , 1.75 , 2.129, 2.616, 3.592, 4.337, 5.48 ],
        #                            [0.706, 0.868, 1.292, 1.759, 2.137, 2.624, 3.598, 4.342, 5.484]])

        print('Number of curves: {}\nNumber of figures: {}'.format(param_set.shape[0], plot_numbers.shape[0]))

        fig, ax = plt.subplots(plot_numbers.shape[0], 1, figsize = (8, 1 + 5 * int(plot_numbers.shape[0])))

        # Define an array for all the cross sections
        all_cs = np.empty((plot_numbers.shape[0], param_set.shape[0], theta_vals.shape[0]))

        # If we only have one set of data to plot, ax[0] is not iterable so we have to manually set this
        if plot_numbers.shape[0] == 1:
            # Handle single plotting...
            pass
        else:
            for selection in plot_numbers:
                for i in range(0, param_set.shape[0]):
                    cs = []
                    for theta in theta_vals:
                        cross_section = model.get_cs_theory(param_set[i], theta, plotting_energy_pool[selection]) * normalizations[i, selection]
                        all_cs[np.where(plot_numbers == selection)[0][0], i, np.where(theta_vals == theta)[0][0]] = cross_section
                        cs.append(cross_section)
                    ax[selection].plot(theta_vals, cs)
                    
                # Plot the data
                for energy in l_som_energies[:, selection]:
                    indices = np.where(data[:, 1] == energy)
                    tmp = data[indices]
                    ax[selection].errorbar(tmp[:, 0], tmp[:, 2], yerr = tmp[:, 3], fmt = '.')

                
                ax[selection].set_xlabel('$\\theta$ [deg]')
                ax[selection].set_ylabel('CS [relative to Rutherford]')
                ax[selection].set_title('E = {:.4f} MeV'.format(plotting_energy_pool[selection]))

        fig.suptitle('E_min = {} MeV, E_max = {} MeV'.format(E_min, E_max))
        plt.tight_layout()
        plt.savefig(directory + '/cross_sections.png')


        # Write code here to compute the bands of cross sections
        percent = 95
        # Create an empty array for the quantiles
        band_cs = np.empty((plot_numbers.shape[0], 3, theta_vals.shape[0]))

        # Compute the quantiles
        for i in range(0, plot_numbers.shape[0]):
            band_cs[i] = np.quantile(all_cs[i], [0.5 - (percent / 200), 0.5, 0.5 + (percent / 200)], axis = 0)

        fig, ax = plt.subplots(plot_numbers.shape[0], 1, figsize = (8, 5 * int(plot_numbers.shape[0])))

        # Plot the 1 sigma band
        for i in range(0, plot_numbers.shape[0]):
            ax[i].plot(theta_vals, band_cs[i, 1, :], label = 'Median')
            ax[i].fill_between(theta_vals, band_cs[i, 0, :], band_cs[i, 2, :], alpha = 0.4, label = '{}%'.format(percent))
            ax[i].set_xlabel('$\\theta$ [deg]')
            ax[i].set_ylabel('CS [relative to Rutherford]')
            ax[i].set_title('E = {:.4f} MeV'.format(plotting_energy_pool[i]))
            ax[i].legend()

        # Plot the data
        for j in plot_numbers:
            for energy in l_som_energies[:, j]:
                indices = np.where(data[:, 1] == energy)
                tmp = data[indices]
                ax[j].errorbar(tmp[:, 0], tmp[:, 2], yerr = tmp[:, 3], fmt = '.')

        plt.suptitle('Results from E_max: {} MeV'.format(E_max))
        plt.tight_layout()
        plt.savefig(directory + '/cs_bands.png')

if __name__ == "__main__":
    directory = sys.argv[1]
    plot_cs = sys.argv[2]
    process(directory, plot_cs)
