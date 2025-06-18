import os
from argparse import ArgumentParser
from pathlib import Path
import h5py


keys_list = []
def get_hdf5_keys(hdf5_f, root_key=''):
    global keys_list
    keys_list.append(root_key)
    for key in hdf5_f.keys():
        if isinstance(hdf5_f[key], h5py.Group):
            get_hdf5_keys(hdf5_f[key], root_key + '/' + key)
        elif isinstance(hdf5_f[key], h5py.Dataset):
            keys_list.append(f'{root_key}/{key}:\t{hdf5_f[key].shape}\t{hdf5_f[key].dtype}')
            

def extract_hdf5_data(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        get_hdf5_keys(f)
        for key in keys_list:
            print(key)


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract keys from an HDF5 file")
    parser.add_argument("--hdf5_file", type=Path, help="Path to the HDF5 file")
    
    args = parser.parse_args()
    
    extract_hdf5_data(args.hdf5_file)