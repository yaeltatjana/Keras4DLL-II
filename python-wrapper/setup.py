from distutils.core import setup, Extension
from Cython.Distutils import build_ext

# Setup configurations for compiling wrapper and python module
setup(
    name="lib-dll-mnist",
    version="1.0",
    ext_modules=[
        Extension(
            "wrapperlibmnist",
            ["wrapper.pyx"],
            language="c++",
            libraries=['dll_mnist_mylib'],
            library_dirs=['../dll/release/lib'],
            runtime_library_dirs=['../dll/release/lib'],
        ),

        Extension(
            "mnist_reader",
            ["mnist_reader.pyx"],
            language="c++",
            libraries=['dll_mnist_mylib'],
            library_dirs=['../dll/release/lib'],
            runtime_library_dirs=['../dll/release/lib'],
        )
    ],
    cmdclass={'build_ext': build_ext, },
)
