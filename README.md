# Projet de Bachelor - Keras4DLL-II

Le *deep learning*, cet apprentissage automatique inspiré des neurones des animaux, est aujourd'hui en plein essor. Il est fortement utilisé dans la plupart des domaines et simplifie notre vie quotidienne. Les réseaux de neurones issus des recherches sont de plus en plus complexes et profonds. Le défi d'aujourd'hui est donc d'accélérer les calculs car les temps d'exécution sont très longs. La *Deep learning Library* (DLL) est une librairie C++ de *deep learning* réputée pour ses performances au niveau de l'exécution. Les *frameworks* les plus connus étant en Python, nous développons un *wrapper* Python pour DLL afin d'améliorer l'accessibilité à la librairie pour les chercheurs. Ces derniers sont rarement des développeurs et il est donc compliqué pour eux d'utiliser DLL. Le défi de ce travail est de porter DLL dans le monde Python tout en conservant ses bonnes performances d'exécution.


**Ce projet est la suite du projet de semestre  [Keras4DLL-I](https://gitlab.forge.hefr.ch/yael.iseli/keras4dll-i)**

## Cloner le projet

Pour cloner le projet avec les *submodules* :
```
git clone --recursive https://github.com/yaeltatjana/Keras4DLL-II.git
```

## Manuel d'installation des dépendances

Pour chaque ensemble de commandes, nous considérons que nous sommes dans le répertoire du projet **Keras4DLL-II**.

Installations requises pour la compilation du projet:
```
pip install numpy         # fonctions utilisées dans les tests fonctionnels
pip install Cython        # compilation du wrapper
pip install unittest      # lancement des tests fonctionnels
```

Il est important d'installer les *header files* de DLL sur la machine:
```
cd dll
make install_headers
```

## *Shared library* de DLL
Pour chaque ensemble de commandes, nous considérons que nous sommes dans le répertoire du projet **Keras4DLL-II**.

### Compilation

Pour compiler la librairie DLL pour le *wrapper* :
```
cd dll
make wrapper_lib
```

/!\ La compilation peut s'avérer assez longue.

### Benchmark

```
cd dll
make wrapper_perf
sudo ./benchmark.sh
```

/!\ La compilation peut s'avérer assez longue.

### Tests fonctionnels

```
cd dll
make wrapper_test
sudo ./test.sh
```

/!\ La compilation peut s'avérer assez longue.

## *Wrapper*
Pour chaque ensemble de commandes, nous considérons que nous sommes dans le répertoire du projet **Keras4DLL-II**.

### Compilation
Pour compiler le *wrapper* (la librairie DLL doit être compilée en premier) :
```
cd python-wrapper
./build_modules.sh
```

Pour spécifier un emplacement de la *shared library* autre que celui issu de la compilation du côté DLL, il est possible d'utiliser la commande suivante :
```
cd python-wrapper
sudo python setup.py install "path/to/sharedlib/repository"
```

### Utilisation
Exemple d'utilisation du module *dll* :

```python
import dll as dll

if __name__ == "__main__":
    # MNIST dataset
    mnist = dll.PyMnistReader()
    mnist.display()

    # Network with 2 dense layers: relu -> softmax
    net = dll.PyDenseDenseNet([28 * 28, 16], [16, 10])
    net.display()
    net.set_initial_momentum(0.8)

    net.fine_tune(mnist, 5)
    net.evaluate(mnist)
```

Pour lancer le script (accessible depuis le répertoire **python-wrapper**):
```
cd python-wrapper
python example.py
```

### Benchmark
Le script python prend en paramètre le nom des *benchmarks* à lancer:

```
cd python-wrapper
python test.py mnist dd
```
Les tests existants sont les suivants:
* mnist
* text
* dd
* ddd
* lenet
* alexnet
* vggnet16

### Tests fonctionnels
```
cd python-wrapper
sudo ./test.sh
```
