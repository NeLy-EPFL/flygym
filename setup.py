from setuptools import setup, find_packages


setup(
    name='flygym',
    version='0.0.1',
    author='Neuroengineering Laboratory, EPFL',
    author_email='sibo.wang@epfl.ch',
    description='Gym environments for NeuroMechFly in various physics simulators',
    packages=find_packages(),
    package_data={'flygym': ['data/*']},
    include_package_data=True,
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
    ],
    install_requires=[
        'gymnasium',
        'numpy',
        'scipy',
        'pyyaml',
        'jupyter',
        'mediapy',
        'imageio',
        'imageio[pyav]',
        'imageio[ffmpeg]',
        'tqdm'
    ],
    extras_require={
        'mujoco': ['mujoco', 'dm_control'],
        'pybullet': ['pybullet'],
        'doc': ['sphinx', 'furo', 'numpydoc'],
        'test': ['pytest', 'ruff']
    },
)
