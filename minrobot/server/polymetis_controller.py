from dataclasses import dataclass
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import time
import polymetis

from .task_ee_config import get_ee_config, DefaultEEConfig


class ActionSpace:
    def __init__(self, low: list[float], high: list[float], normalize: bool):
        self.raw_low = np.array(low, dtype=np.float32)
        self.raw_high = np.array(high, dtype=np.float32)
        self.normalize = normalize

        if self.normalize:
            self.low = np.ones_like(self.raw_low) * -1
            self.high = np.ones_like(self.raw_high) * 1
        else:
            self.low = self.raw_low
            self.high = self.raw_high

    def _assert_in_range(self, action: np.ndarray):
        correct = (action <= self.high).all() and (action >= self.low).all()
        if correct:
            return

        for i in range(self.low.size):
            check = action[i] >= self.low[i] and action[i] <= self.high[i]
            print(f"{self.low[i]:.2f} <= {action[i]:.2f} <= {self.high[i]:.2f}? {check}")
        assert False

    def maybe_denormalize(self, action: np.ndarray, clip):
        self._assert_in_range(action)

        if self.normalize:
            action = self.raw_low + (0.5 * (action + 1.0) * (self.raw_high - self.raw_low))

        if clip:
            action = np.clip(action, self.raw_low, self.raw_high)
        return action


@dataclass
class PolyMetisControllerConfig:
    controller_type: str = "CARTESIAN_DELTA"
    max_delta: float = 0.05
    task: str = "default"
    norm_action: int = 1

    def __post_init__(self):
        assert self.controller_type in {"CARTESIAN_DELTA", "CARTESIAN_IMPEDANCE"}


class PolyMetisController:
    """Controller run on server.

    all methods should return python native types for easy integration with 0rpc
    """

    def __init__(self, cfg: PolyMetisControllerConfig) -> None:
        self.robot_ip = "localhost"
        self.cfg = cfg

        self._ee_config = get_ee_config(cfg.task)
        self._action_space = self._get_action_space(cfg, self._ee_config)

        self._robot = polymetis.RobotInterface(ip_address=self.robot_ip, enforce_version=False)
        self._gripper = polymetis.GripperInterface(p_address=self.robot_ip)

        self._robot.set_home_pose(torch.from_numpy(self._ee_config.home_joint_pos))
        self._robot.go_home(blocking=True)
        ee_pos, _ = self._robot.get_ee_pose()
        print(f"init ee pos: {ee_pos}, desired init ee pos: {self._ee_config.home_ee_pos}")

        if hasattr(self._gripper, "metadata") and hasattr(self._gripper.metadata, "max_width"):
            # Should grab this from robotiq2f
            self._max_gripper_width = self._gripper.metadata.max_width
        else:
            self._max_gripper_width = 0.08  # default, from FrankaHand Value

        self.desired_gripper_qpos = 0

    def hello(self):
        info = [
            "PolyMetisController:",
            f"\t controller_type: {self.cfg.controller_type}",
            f"\t task: {self.cfg.task}",
            f"\t internal action range: {self._action_space.raw_low} -> {self._action_space.raw_high}",
            f"\t client action range: {self._action_space.low} -> {self._action_space.high}",
            f"\t normalized action space {bool(self.cfg.norm_action)}"
        ]
        return "\n".join(info)

    def _get_action_space(self, cfg: PolyMetisControllerConfig, ee_config: DefaultEEConfig):
        if cfg.controller_type == "CARTESIAN_DELTA":
            high = [cfg.max_delta] * 3 + [cfg.max_delta * 4] * 3
            low = [-x for x in high]
        elif self.cfg.controller_type == "CARTESIAN_IMPEDANCE":
            low = ee_config.ee_range_low
            high = ee_config.ee_range_high
        else:
            raise ValueError("Invalid Controller type provided")

        # Add the gripper action space
        low.append(0.0)
        high.append(1.0)
        return ActionSpace(low, high, bool(cfg.norm_action))

    def update_gripper(self, gripper_action: float, blocking=False) -> None:
        # We always run the gripper in absolute position
        gripper_action = max(min(gripper_action, 1), 0)
        width = self._max_gripper_width * (1 - gripper_action)

        self.desired_gripper_qpos = gripper_action

        self._gripper.goto(
            width=width,
            speed=0.1,
            force=0.01,
            blocking=blocking,
        )

    def update(self, action_: list[float]) -> None:
        """
        Updates the robot controller with the action
        """
        assert len(action_) == 7, f"wrong action dim: {len(action_)}"

        if not self._robot.is_running_policy():
            print("restarting cartesian impedance controller")
            self._robot.start_cartesian_impedance()
            time.sleep(1)

        action = np.array(action_)
        action = self._action_space.maybe_denormalize(action, clip=True)

        robot_action: np.ndarray = action[:-1]
        gripper_action: float = action[-1]

        if self.cfg.controller_type == "CARTESIAN_DELTA":
            ee_pos, ee_quat = self._robot.get_ee_pose()
            delta_pos, delta_ori = np.split(robot_action, [3])

            # compute new pos and new quat
            new_pos = ee_pos.numpy() + delta_pos
            # TODO: this can be made much faster using purpose build methods instead of scipy.
            old_rot = Rotation.from_quat(ee_quat.numpy())
            delta_rot = Rotation.from_euler("xyz", delta_ori)
            new_rot = (delta_rot * old_rot).as_euler("xyz")

            # clip
            new_pos, new_rot = self._ee_config.clip(new_pos, new_rot)
            new_quat = Rotation.from_euler("xyz", new_rot).as_quat()  # type: ignore
            self._robot.update_desired_ee_pose(torch.from_numpy(new_pos).float(), torch.from_numpy(new_quat).float())
        elif self.cfg.controller_type == "CARTESIAN_IMPEDANCE":
            pos, rot = np.split(robot_action, [3])
            pos, rot = self._ee_config.clip(pos, rot)

            ori = Rotation.from_euler("xyz", rot).as_quat()  # type: ignore
            self._robot.update_desired_ee_pose(torch.from_numpy(pos).float(), torch.from_numpy(ori).float())
        else:
            raise ValueError("Invalid Controller type provided")

        # Update the gripper
        self.update_gripper(gripper_action, blocking=False)

    def _go_home(self, home):
        """home is in joint position"""
        self._robot.set_home_pose(torch.from_numpy(home))
        self._robot.go_home(blocking=True)

        ee_pos, _ = self._robot.get_ee_pose()
        ee_pos = ee_pos.numpy()
        if np.abs(ee_pos - self._ee_config.home_ee_pos).max() > 0.02:
            print("going home, 2nd try")
            # go home again
            self._robot.set_home_pose(torch.from_numpy(home))
            self._robot.go_home(blocking=True)

    def reset(self, randomize: bool) -> None:
        print("reset env")

        self.update_gripper(0, blocking=True)  # open the gripper
        self._ee_config.reset(self)

        if self._robot.is_running_policy():
            self._robot.terminate_current_policy()

        home = self._ee_config.home_joint_pos
        if randomize:
            # TODO: adjust this noise
            high = 0.1 * np.ones_like(home)
            noise = np.random.uniform(low=-high, high=high)
            print("home noise:", noise)
            home = home + noise

        self._go_home(home)

        assert not self._robot.is_running_policy()
        self._robot.start_cartesian_impedance()
        time.sleep(1)

    def get_state(self) -> dict[str, list[float]]:
        """
        Returns the robot state dictionary.
        For VR support MUST include [ee_pos, ee_quat]
        """
        ee_pos, ee_quat = self._robot.get_ee_pose()
        gripper_state = self._gripper.get_state()
        gripper_pos = 1 - (gripper_state.width / self._max_gripper_width)

        state = {
            "robot0_eef_pos": ee_pos.tolist(),
            "robot0_eef_quat": ee_quat.tolist(),
            "robot0_gripper_qpos": [gripper_pos],
            "robot0_desired_gripper_qpos": [self.desired_gripper_qpos],
        }

        return state
