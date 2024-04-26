import numpy as np


class DefaultEEConfig:
    def __init__(self):
        pass

    @property
    def home_ee_pos(self):
        return np.array([0.41, 0, 0.5], dtype=np.float32)

    @property
    def home_joint_pos(self):
        return np.pi * np.array([0.0, -0.15, 0.0, -0.75, 0.0, 0.60, 0], dtype=np.float32)

    @property
    def ee_range_low(self):
        # 0.01 -> 1cm in real world with polymetis
        return [0.0, -0.5, 0.15, -np.pi, -np.pi, -np.pi]

    @property
    def ee_range_high(self):
        return [0.9, 0.5, 0.9, np.pi, np.pi, np.pi]

    def clip(self, pos: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # default task sets no limits :)
        return pos, rot

    def ee_in_good_range(self, pos: np.ndarray, quat: np.ndarray, verbose) -> bool:
        # default task sets no limits :)
        # NOTE: when clipping rotation, we should operate in absolute value
        return True

    def reset(self, controller) -> None:
        """perform some movement before calling go-home in reset

        this is useful for some safety reasons
        for example, we may want to first move up then call go_home
        which may involve rotating the gripper
        """
        return


def get_ee_config(task):
    if task == "default":
        return DefaultEEConfig()

    assert False, f"unknown task {task}"
