from setuptools import setup, find_packages

setup(
    name='DART_dynamic_models',                 # Name of your package
    version='0.1.0',                   # Version number
    description='This package provides easy access to DART dynamic models and relative utility functions, e.g. for plotting.',  # Short description
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Lorenzo Lyons',
    author_email='l.lyons@tudelft.nl',
    url='https://github.com/Lorenzo-Lyons/DART', 
    packages=find_packages(),          # Automatically find sub-packages
    install_requires=[                 # Dependencies (if any)
        # 'requests', 
        # 'numpy',
    ],
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    package_data={
        'DART_dynamic_models': ['SVGP_saved_parameters/*'],  # Include all files in the data folder
        'DART_dynamic_models': ['actuator_dynamics_saved_parameters/*'],  # Include all files in the data folder
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    #python_requires='>=3.6.9',           # Specify minimum Python version
)