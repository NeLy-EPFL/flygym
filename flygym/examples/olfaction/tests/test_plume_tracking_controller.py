import numpy as np
from collections import Counter
from flygym.examples.olfaction import PlumeNavigationController, WalkingState


def test_controller():
    controller = PlumeNavigationController(dt=1e-3, random_seed=0)
    state_hist_no_encounter = []
    state_hist_all_encounter = []
    for i in range(100000):
        state, _, _ = controller.decide_state(
            encounter_flag=False, fly_heading=np.array([-1, 1])  # positive y points up
        )
        state_hist_no_encounter.append(state)
    counter_no_encounter = Counter(state_hist_no_encounter)

    controller = PlumeNavigationController(dt=1e-3, random_seed=0)
    for i in range(100000):
        state, _, _ = controller.decide_state(
            encounter_flag=True, fly_heading=np.array([-1, 1])
        )
        state_hist_all_encounter.append(state)
    counter_all_encounter = Counter(state_hist_all_encounter)

    assert (
        counter_all_encounter[WalkingState.FORWARD]
        > counter_no_encounter[WalkingState.FORWARD] * 1.5
    )
    assert (
        counter_all_encounter[WalkingState.TURN_LEFT]
        > counter_all_encounter[WalkingState.TURN_RIGHT] * 2
    )
    assert (
        0.75
        < counter_no_encounter[WalkingState.TURN_LEFT]
        / counter_no_encounter[WalkingState.TURN_RIGHT]
        < 1.25
    )
