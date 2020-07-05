rm -r /usr/local/lib/python3.6/dist-packages/lib_dll_mnist-1.0.egg-info
rm -r src/all.cpp
sudo python setup.py install "../dll/release/lib"
