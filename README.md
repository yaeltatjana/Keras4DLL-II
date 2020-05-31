# Projet de Bachelor - Keras4DLL-II

**Ce projet est la suite du projet de semestre  [Keras4DLL-I](https://gitlab.forge.hefr.ch/yael.iseli/keras4dll-i)**

Pour cloner le projet avec les *submodules* :
```
git clone --recursive -j8 https://gitlab.forge.hefr.ch/tb_19-20_iseli/keras4dll-ii.git
```

Pour lancer l'exemple simple avec DLL :
```
cd dll
make dll_mnist_simple_example
cp -r mnist ./release/bin
./release/bin/dll_mnist_simple_example
```

Pour lancer le *wrapper* :
```
cd dll
make create_shared_library 
cd ../python-wrapper
python setup.py install
cp -r ../dll/mnist/ .
python test-dll-wrapper.py 
```

