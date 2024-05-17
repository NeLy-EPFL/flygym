import numpy as np
from flygym.examples.vision.realistic_vision import RetinaMapper


def test_retina_mapper():
    a = np.random.RandomState(0).rand(2, 2, 721)
    rm = RetinaMapper()
    assert (a == rm.flyvis_to_flygym(rm.flygym_to_flyvis(a))).all()
    assert (a == rm.flygym_to_flyvis(rm.flyvis_to_flygym(a))).all()
