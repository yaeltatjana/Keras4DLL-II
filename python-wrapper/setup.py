try:
    from distutils.core import setup, Extension
    from Cython.Distutils import build_ext
except:
    print("You don't have Cython installed")
    sys.exit(1)


# Setup configurations for compiling wrapper and python module
setup(
    name="lib-dll-mnist",
    version="1.0",
    ext_modules=[
        # Extension(
        #     "wrapperlibmnist",
        #     ["wrapper.pyx"],
        #     language="c++",
        #     libraries=['dll_mnist_mylib'],
        #     library_dirs=['../dll/release/lib'],
        #     runtime_library_dirs=['../dll/release/lib'],
        # ),
        # Extension(
        #     "mnist_reader",
        #     ["mnist_reader.pyx"],
        #     language="c++",
        #     libraries=['dll_mnist_mylib'],
        #     library_dirs=['../dll/release/lib'],
        #     runtime_library_dirs=['../dll/release/lib'],
        # ),
        Extension(
            name='dll',
            sources=['src/all.pyx'], #"src/MnistReader/mnist_reader.pyx", "src/DDNet/dense_dense_net.pyx" #"mnist_reader.pyx", "dense_dense_net.pyx"
            language="c++",
            libraries=['dll_mnist_mylib'],
            library_dirs=['../dll/release/lib'],
            runtime_library_dirs=['../dll/release/lib'],
        )
    ],
    cmdclass={'build_ext': build_ext, },
)
