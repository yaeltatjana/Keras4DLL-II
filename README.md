# Projet de Bachelor - Keras4DLL-II

**Ce projet est la suite du projet de semestre  [Keras4DLL-I](https://gitlab.forge.hefr.ch/yael.iseli/keras4dll-i)**

Pour cloner le projet avec les *submodules* :
```
git clone --recursive -j8 https://github.com/yaeltatjana/Keras4DLL-II.git
```

Les dernières modifications sont actuellement sur les branches *dev*, tant pour [Keras4DLL-II](https://github.com/yaeltatjana/Keras4DLL-II/tree/dev) que pour [DLL](https://github.com/yaeltatjana/dll/tree/dev). Il faut donc changer de branche:
```
git checkout dev
cd dll
git checkout dev
```

Pour compiler la librairie DLL pour le *wrapper* :
```
cd dll
make release/lib/libdll_mnist_mylib.so
```

Pour compiler le *wrapper* (la librairie DLL doit être compilée d'abord) :
```
cd python-wrapper
python setup.py install "path-to-so-lib"
cp -r ../dll/mnist/ .
python test-dll-wrapper.py 
```

Pour tester le *wrapper* :
```
cp -r dll/mnist/ dll
cd python-wrapper
python test-dll-wrapper.py 
```

