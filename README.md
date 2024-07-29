## Simulating embodied sensorimotor control with NeuroMechFly v2

![](https://github.com/NeLy-EPFL/_media/blob/main/flygym/banner_large.jpg?raw=true)

[**Documentation**](https://neuromechfly.org/) | [**Preprint**](https://www.biorxiv.org/content/10.1101/2023.09.18.556649) | [**Discussion Board**](https://github.com/NeLy-EPFL/flygym/discussions)

![Python: 3.9–3.12](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)
[![License: MPL 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/NeLy-EPFL/flygym/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/flygym.svg)](https://badge.fury.io/py/flygym)
![Repo Size](https://img.shields.io/github/repo-size/NeLy-EPFL/flygym)

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/nelyepfl/flygym)
![Docker image size](https://img.shields.io/docker/image-size/nelyepfl/flygym/latest)



![overview_video](https://github.com/NeLy-EPFL/_media/blob/main/flygym/overview_video.gif?raw=true)

This repository contains the source code for FlyGym, the Python library for NeuroMechFly v2, a digital twin of the adult fruit fly *Drosophila* melanogaster that can see, smell, walk over challenging terrain, and interact with the environment (see our [NeuroMechFly v2 paper](https://www.biorxiv.org/content/10.1101/2023.09.18.556649)).

NeuroMechFly consists of the following components:
- **Biomechanical model:** The biomechanical model is based on a micro-CT scan of a real adult female fly (see our original NeuroMechFly publication). We have adjusted several body segments (in particular in the antennae) to better reflect the biological reality.
- **Vision:** The fly has compound eyes consisting of individual units called ommatidia arranged on a hexagonal lattice. We have simulated the visual inputs on the fly’s retinas.
- **Olfaction:** The fly has odor receptors in the antennae and the maxillary palps. We have simulated the odor inputs experienced by the fly by computing the odor/chemical intensity at these locations.
- **Hierarchical control:** The fly’s Central Nervous System consists of the brain and the Ventral Nerve Cord (VNC), a hierarchy analogous to our brain-spinal cord organization. The user can build a two-part model — one handling brain-level sensory integration and decision making and one handling VNC-level motor control — with an interface between the two consisting of descending (brain-to-VNC) and ascending (VNC-to-brain) representations.
- **Leg adhesion:** Insects have evolved specialized adhesive structures at the tips of the legs that enable locomotion vertical walls and overhanging ceilings. We have simulated these structures in our model. The mechanism by which the fly lifts the legs during locomotion despite adhesive forces is not well understood; to abstract this, adhesion can be turned on/off during leg stance/swing.
- **Mechanosensory feedback:** The user has access to the joint angles, forces, and contact forces experienced by the fly.

NeuroMechFly formulates the control of the simulated fly as a partially observable Markov Decision Process (MDP) and implements the Gym interface. This allows the user to use a wide range of reinforcement learning algorithms to train the fly to perform tasks. The standardized interface also allows the user to easily implement their own premotor computation and/or sensory preprocessing processes.

This package is developed at the [Neuroengineering Laboratory](https://www.epfl.ch/labs/ramdya-lab/), EPFL.

## Installation and dependencies
In brief:
```bash
pip install "flygym"
# or pip install "flygym[examples]" to install additional dependencies needed for examples
```

Alternatively, we provide a [Docker image](https://hub.docker.com/r/nelyepfl/flygym). See [our website](https://neuromechfly.org/installation.html) for details, especially if you plan to install FlyGym in the developer mode (i.e. if you plan to make changes to the code). Dependencies are specified in [`setup.py`](https://github.com/NeLy-EPFL/flygym/blob/main/setup.py) and will be installed automatically upon installation using pip. Installation should take no more than a few minutes. The PyPI version of the current release of FlyGym is indicated on the shield at the top of this page. No special, paid software is required to use FlyGym.

## Demos
See [our website](https://neuromechfly.org/tutorials/index.html) for tutorials, including expected outputs. For code blocks that take more than a few seconds to run, the running time (on a 2020 MacBook Pro with M1 processor running macOS 13.5.2) is indicated, typically in the form of a progress bar.

## Reproducing results in the paper
We are constantly working on expanding the package and improving its usability; therefore the package is subject to change. To reproduce the exact results demonstrated in our preprint, use [FlyGym 0.1.0](https://github.com/NeLy-EPFL/flygym/releases/tag/v0.1.0) and analysis code [here](https://github.com/NeLy-EPFL/nmf2-paper).


## Citation
If you use NeuroMechFly in your work, please cite the following papers:
```
@article{WangChen2023,
  author = {Sibo Wang-Chen and Victor Alfred Stimpfling and Pembe Gizem \"{O}zdil and Louise Genoud and Femke Hurtak and Pavan Ramdya},
  title = {NeuroMechFly 2.0, a framework for simulating embodied sensorimotor control in adult Drosophila},
  year = {2023},
  doi = {10.1101/2023.09.18.556649},
  URL = {https://www.biorxiv.org/content/early/2023/09/18/2023.09.18.556649},
  journal = {bioRxiv}
}

@article{LobatoRios2022,
  doi = {10.1038/s41592-022-01466-7},
  url = {https://doi.org/10.1038/s41592-022-01466-7},
  year = {2022},
  month = may,
  volume = {19},
  number = {5},
  pages = {620--627},
  author = {Victor Lobato-Rios and Shravan Tata Ramalingasetty and Pembe Gizem \"{O}zdil and Jonathan Arreguit and Auke Jan Ijspeert and Pavan Ramdya},
  title = {{NeuroMechFly}, a neuromechanical model of adult {Drosophila} melanogaster},
  journal = {Nature Methods}
}
```