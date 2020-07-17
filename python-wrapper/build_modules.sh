#!/bin/bash

# script to build the python module and to install it
# to be launched from python-wrapper directory

# change path with your own
path_module="/usr/local/lib/python3.6/dist-packages/libdll-1.0.egg-info"

path_all_file="./src/all.cpp"

if [ -f "${path_module}" ]; then
  rm -r "${path_module}"
fi

if [ -f "${path_all_file}" ]; then
  rm -r "${path_all_file}"
fi

# change path if you want to localize the library from elsewhere
sudo python setup.py install "../dll/release/lib"
