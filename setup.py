from setuptools import setup, find_packages


setup(
    name="flygym",
    version="1.2.0",
    author="Neuroengineering Laboratory, EPFL",
    author_email="sibo.wang@epfl.ch",
    description=(
        "Implementation of NeuroMechFly v2, framework for simulating embodied "
        "sensorimotor control in adult Drosophila"
    ),
    packages=find_packages(),
    package_data={"flygym": ["data/*", "config.yaml"]},
    include_package_data=True,
    python_requires=">=3.9,<3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    install_requires=[
        "gymnasium",
        "numpy<2",
        "scipy",
        "pyyaml",
        "jupyter",
        "mediapy",
        "imageio",
        "imageio[pyav]",
        "imageio[ffmpeg]",
        "tqdm",
        "mujoco==3.2.3",
        "dm_control==1.0.23",
        "numba",
        "opencv-python",
    ],
    extras_require={
        "dev": [
            "sphinx",
            "sphinxcontrib.googleanalytics",
            "furo",
            "numpydoc",
            "pytest",
            "ruff",
            "black==23.3.0",
            "black[jupyter]",
            "shapely",
            "rasterio",
            "requests",
        ],
        "examples": [
            "joblib",
            "networkx",
            "lightning",
            "tensorboardX",
            "pandas",
            "scikit-learn",
            "seaborn",
            "torch<=2.5.0",
            "phiflow",
            "h5py",
            "toolz",  # remove when it's added to flyvis's requirements.txt (flyvis #2)
        ],
    },
    url="https://neuromechfly.org/",
    long_description=open("README.md", encoding="UTF-8").read(),
    long_description_content_type="text/markdown",
)
