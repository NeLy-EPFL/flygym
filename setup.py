from setuptools import setup, find_packages


setup(
    name='flygym',
    version='0.0.1',
    author='Sibo Wang',
    author_email='sibo.wang@epfl.ch',
    description='Gym environments for NeuroMechFly in various physics simulators',
    packages=find_packages(),
    package_data={'flygym': ['data/*']},
    include_package_data=True,
    python_requires='>=3.7, <3.9',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
    ],
    install_requires=[
        'gymnasium',
        "numpy",
        "pyyaml",
        "mediapy",
        "imageio",
        "imageio[pyav]",
        "imageio[ffmpeg]"
    ],
    extras_require={
        'mujoco': ['mujoco', 'dm_control'],
        'pybullet': ['pybullet'],
        'docs': ['sphinx', 'furo', 'numpydoc']
    },
)