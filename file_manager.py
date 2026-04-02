# Code by Landry Marquis
#
# If you use this code, please cite the paper: 
# Marquis L. et al., Fusion of JWST data: Demonstrating practical feasibility,
# Astronomy & Astrophysics, volume 708, 2026

import numpy as np
import scipy.sparse
import os
import inspect
from astropy.io import fits


def retrieve_name(var, dup):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][dup]


def save_preprocessed_vectors(vectors, nb_sparse, fusion_name):

    if not os.path.exists('Vectors_Save/' + fusion_name):
        os.makedirs('Vectors_Save/' + fusion_name)

    duplicate = [0] * (nb_sparse + 1)
    for i in range(nb_sparse + 1, len(vectors)):
        count = 0
        for x in vectors[nb_sparse:i]:
            if vectors[i] is x:
                count += 1
        duplicate.append(count)

    names = []
    for i in range(len(vectors)):
        name = retrieve_name(vectors[i], duplicate[i])
        names.append(name)

    for i in range(len(names)):
        if i < nb_sparse:
            scipy.sparse.save_npz('Vectors_Save/' + fusion_name + '/' + names[i], vectors[i])
        else:
            np.save('Vectors_Save/' + fusion_name + '/' + names[i], vectors[i])

    np.save('Vectors_Save/' + fusion_name + '/names', names)


def load_preprocessed_vectors(fusion_name):

    vectors_file_names = os.listdir('Vectors_Save/' + fusion_name + '/')
    names = np.load('Vectors_Save/' + fusion_name + '/names.npy')
    vectors = []

    for name in names:
        if name + '.npy' in vectors_file_names:
            vectors.append(np.load('Vectors_Save/' + fusion_name + '/' + name + '.npy', allow_pickle = True))
        else:
            vectors.append(scipy.sparse.load_npz('Vectors_Save/' + fusion_name + '/' + name + '.npz'))

    return vectors


def save_as_fits(image, fits_name):

    hdu = fits.PrimaryHDU(data = image)
    hdu.writeto(fits_name, overwrite = True)


def read_throughput(path):

    with open(path, 'r') as file:
        content = file.read()
        temp = np.array((content.split('\n')[1:]))
        throughput = np.zeros((2, len(temp)-1))
        for i in range(len(temp)-1):
            throughput[:,i] = temp[i].split(" ")

    return throughput
