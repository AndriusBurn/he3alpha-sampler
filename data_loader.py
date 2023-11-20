#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:12:06 EDT 2023

@author: Andrius Burnelis
"""
import numpy as np
import sys


class DataLoader(object):
    """
    This class loads and selects the subset of the data we want to work with.
    It returns the data in the 'get_data()' method.
    It also handles the normalization information in the 'get_normalization_info()' method.

    NOTE: To load the data correctly, we cannot separate the data between the interaction regions.
    E_min : [0.676, 0.84 , 1.269, 1.741, 2.12 , 2.609, 2.609, 3.586, 4.332, 5.475]
    E_max : [0.706, 0.868, 1.292, 1.759, 2.137, 2.624, 2.624, 3.598, 4.342, 5.484]
    """
    def __init__(self, E_min, E_max, which_data):
        # Store into self
        self.E_min = E_min
        self.E_max = E_max
        self.which_data = which_data

        # Load each of the sets [E, theta, cs, err]
        som_data = np.load('./data/som.npy')
        barnard_data = np.load('./data/barnard.npy')
        
        # Select data larger than E_min
        barnard_upper_indices = np.where(barnard_data[:, 0] >= E_min)
        som_upper_indices = np.where(som_data[:, 0] >= E_min)

        # Select data smaller than E_max
        barnard_lower_indices = np.where(barnard_data[:, 0] <= E_max)
        som_lower_indices = np.where(som_data[:, 0] <= E_max)

        # Take the intersection of these two to select the data by energy
        barnard_indices = np.intersect1d(barnard_lower_indices, barnard_upper_indices)
        som_indices = np.intersect1d(som_lower_indices, som_upper_indices)

        # Select the data based on energy
        self.barnard_data = barnard_data[barnard_indices]
        self.som_data = som_data[som_indices]

        # Now select which data
        if which_data == 'both':
            self.data = np.concatenate([self.barnard_data, self.som_data])
        elif which_data == 'som':
            self.data = self.som_data
        elif which_data == 'barnard':
            self.data = self.barnard_data
        else:
            sys.stderr.write('Choose a \'which_data\': both, som, barnard...')
            sys.exit(-1)


    def get_data(self):
        """
        Simply returns the data in the [E, theta, cs, err] format
        """
        return self.data
    

    def get_normalization_grouping(self):
        """
        This returns an array that is used to group the data together. Indices of like
        values in this array, should have the same normalization coefficient attached to 
        them in the log-likelihood.
        """
        # Define the array of data points for each interaction region and energy
        l_som = np.array([[ 5, 11, 16, 16, 18, 18, 18, 18, 18, 18],
                          [ 5,  9, 14, 14, 16, 16, 16, 16, 16, 16],
                          [ 7,  9, 15, 16, 18, 18, 18, 18, 19, 19]])
        
        l_som_energies = np.array([0.706, 0.868, 1.292, 1.759, 2.137, 2.624, 2.624, 3.598, 4.342, 5.484])

        # Select the energy range we want
        l_som_lower_indices = np.where(l_som_energies <= self.E_max)
        l_som_upper_indices = np.where(l_som_energies >= self.E_min)

        # Pull the total indices for the energy range we want
        indices = np.intersect1d(l_som_lower_indices, l_som_upper_indices)
        self.indices = indices

        # Select based on the energy
        l_som = l_som[:, indices]
        l_som_energies = l_som_energies[indices]

        # Collapse all the interaction regions to one
        l_som = np.sum(l_som, axis = 0)

        # Create an empty array of normalization matchings
        grouping = np.zeros(self.data.shape[0])

        # Generate the array that groups the Som data
        count = 0
        for i in range(0, l_som.shape[0]):
            for j in range(0, l_som[i]):
                grouping[count] = i
                count += 1

        # Group for all the Barnard data
        grouping[count:] = l_som.shape[0]
        return grouping
    

    def get_normalization_prior_info(self):
        # All the priors are gaussian centered at 1.0 with a different sigmas
        som_f_sigmas = np.array([0.064, 0.076, 0.098, 0.057, 0.045, 0.062, 0.041, 0.077, 0.063, 0.089])
        barnard_f_sigma = np.array([0.05])

        # Select the sigmas (Assumes we already grouped the norms)
        if self.which_data == 'som':
            sigmas = som_f_sigmas[self.indices]
        if self.which_data == 'barnard':
            sigmas = barnard_f_sigma
        if self.which_data == 'both':
            sigmas = np.concatenate([som_f_sigmas[self.indices], barnard_f_sigma])
        
        # Specify the means
        mus = np.ones(sigmas.shape)

        # Get the means and sigmas together + generate the bounds
        gauss_prior_fs = np.vstack([mus, sigmas]).T
        bounds = np.array([[0, 2] for i in sigmas])

        # Combine all together
        norm_prior_info = np.hstack([bounds, gauss_prior_fs])
        return norm_prior_info


def main():
    # TEST
    #     E_min : [0.676, 0.84 , 1.269, 1.741, 2.12 , 2.609, 2.609, 3.586, 4.332, 5.475]
    #     E_max : [0.706, 0.868, 1.292, 1.759, 2.137, 2.624, 2.624, 3.598, 4.342, 5.484]
    E_min = 1.269 # MeV
    E_max = 2.642
    which_data = 'som'

    loader = DataLoader(E_min, E_max, which_data)

    data = loader.get_data()
    norm_group = loader.get_normalization_grouping()
    norm_prior_info = loader.get_normalization_prior_info()

    print(data)
    print(norm_group)
    print(norm_prior_info)

if __name__ == '__main__':
    main()