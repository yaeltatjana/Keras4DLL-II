try:
    from distutils.core import setup, Extension
    from Cython.Distutils import build_ext
    import sys
except:
    print("You don't have Cython installed")
    sys.exit(1)

path = ''

if len(sys.argv) < 2:
    print('Stop, need library path')
    sys.exit(1)
else:
    path = sys.argv[2]
    sys.argv.remove(path)  # remove so that cython can read properly its arguments


# Setup configurations for compiling wrapper and python module
setup(
    name="lib-dll",
    version="1.0",
    ext_modules=[
        Extension(
            name='dll',
            sources=['src/all.pyx'],
            language="c++",
            libraries=['dll'],
            library_dirs=[path],
            runtime_library_dirs=[path],
        )
    ],
    cmdclass={'build_ext': build_ext}
)
