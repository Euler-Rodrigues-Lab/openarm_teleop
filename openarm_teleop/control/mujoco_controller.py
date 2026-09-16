"""
MuJoCo Controller for OpenArm robot.
Handles MuJoCo interfacing, joint torque control, and gravity compensation.
Takes joint position goals and applies them to the simulation.

Author: Chuizheng Kong (Refactored)
"""

import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from typing import Optional, Tuple


class OpenArmMuJoCoController:
    """
    MuJoCo Controller for OpenArm v1 robot.
    Applies joint positions.
    """

    def __init__(self, mujoco_model, mujoco_data, kp=150.0, kd=None, debug=False):
        """
        Initialize MuJoCo controller for OpenArm.

        Args:
            mujoco_model: MuJoCo model
            mujoco_data: MuJoCo data
            kp: Proportional gain for joint control
            kd: Derivative gain for joint control (auto-computed if None)
            debug: Enable debug output
        """
        self.model = mujoco_model
        self.data = mujoco_data
        self.debug = debug

        # Mocap body state (optional, enabled when a mocap root exists in XML).
        self.mocap_enabled = False
        self.mocap_index = None
        self.mocap_body_name = None

        # Get joint indices in MuJoCo model
        self._setup_joint_indices()


        # Goal joint angles
        self.q_goal_right = np.zeros(7)
        self.q_goal_left = np.zeros(7)
        self.q_goal_right_hand = None
        self.q_goal_left_hand = None

        # Initialize with current joint positions
        self._update_current_positions()


        # Build actuator cache for position control
        self._build_actuator_cache()

        if self.debug:
            print("OpenArm MuJoCo Controller initialized")

    # --- Mocap Body Management ---
    def setup_mocap_body(self, mocap_body_name="base_mocap_mover"):
        self.mocap_body_name = mocap_body_name
        self.mocap_enabled = False
        self.mocap_index = None

        mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
        if mocap_body_id >= 0:
            for i in range(self.model.nmocap):
                if self.model.body_mocapid[mocap_body_id] == i:
                    self.mocap_index = i
                    self.mocap_enabled = True
                    break

            if not self.mocap_enabled and self.model.nmocap > 0:
                self.mocap_index = 0
                self.mocap_enabled = True

        if self.debug:
            print(f"Mocap body setup: {self.mocap_enabled}, index: {self.mocap_index}")

    def update_mocap_body(self, position, R_world_body):
        if not self.mocap_enabled or self.mocap_index is None:
            return

        try:
            self.data.mocap_pos[self.mocap_index] = np.asarray(position, dtype=float).reshape(3)

            quat_xyzw = Rotation.from_matrix(np.asarray(R_world_body, dtype=float).reshape(3, 3)).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            self.data.mocap_quat[self.mocap_index] = quat_wxyz
        except Exception as e:
            if self.debug:
                print(f"Error updating mocap: {e}")

    def _build_actuator_cache(self):
        """Cache actuator IDs for position control."""
        self.right_arm_actuator_ids = []
        self.left_arm_actuator_ids = []
        self.right_hand_actuator_ids = []
        self.left_hand_actuator_ids = []
        self.other_actuator_ids = []

        controlled_actuators = set()

        # Helper to find actuator for joint
        def get_actuator_for_joint(joint_id):
            for i in range(self.model.nu):
                # Check for joint actuator
                if self.model.actuator_trnid[i, 0] == joint_id and self.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                    return i
            return None

        # Helper to find actuator for tendon
        def get_actuator_for_tendon(tendon_name):
            tendon_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
            if tendon_id == -1: return None
            
            for i in range(self.model.nu):
                if self.model.actuator_trnid[i, 0] == tendon_id and self.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
                    return i
            return None

        for joint_id in self.right_arm_joint_ids:
            aid = get_actuator_for_joint(joint_id)
            if aid is not None:
                self.right_arm_actuator_ids.append(aid)
                controlled_actuators.add(aid)

        for joint_id in self.left_arm_joint_ids:
            aid = get_actuator_for_joint(joint_id)
            if aid is not None:
                self.left_arm_actuator_ids.append(aid)
                controlled_actuators.add(aid)

        # Right Hand (Tendon Actuated)
        aid = get_actuator_for_tendon("split_right")
        if aid is not None:
            self.right_hand_actuator_ids.append(aid)
            controlled_actuators.add(aid)

        # Left Hand (Tendon Actuated)
        aid = get_actuator_for_tendon("split_left")
        if aid is not None:
            self.left_hand_actuator_ids.append(aid)
            controlled_actuators.add(aid)

        for i in range(self.model.nu):
            if i not in controlled_actuators:
                self.other_actuator_ids.append(i)

    def _setup_joint_indices(self):
        """Setup joint indices and names for both arms."""
        # Right arm joint names in MuJoCo
        self.right_arm_joint_names = [
            "openarm_right_joint1",
            "openarm_right_joint2",
            "openarm_right_joint3",
            "openarm_right_joint4",
            "openarm_right_joint5",
            "openarm_right_joint6",
            "openarm_right_joint7",
        ]

        # Left arm joint names in MuJoCo
        self.left_arm_joint_names = [
            "openarm_left_joint1",
            "openarm_left_joint2",
            "openarm_left_joint3",
            "openarm_left_joint4",
            "openarm_left_joint5",
            "openarm_left_joint6",
            "openarm_left_joint7",
        ]

        # Torso joint names (None for OpenArm)
        self.torso_joint_names = []

        # Head joint names (None for OpenArm)
        self.head_joint_names = []

        # Right hand joint names
        self.right_hand_joint_names = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]

        # Left hand joint names
        self.left_hand_joint_names = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]

        # Get joint qpos addresses using the correct approach
        self.right_arm_qpos_addrs = []
        self.left_arm_qpos_addrs = []
        self.torso_qpos_addrs = []
        self.head_qpos_addrs = []
        self.right_hand_qpos_addrs = []
        self.left_hand_qpos_addrs = []

        # Build joint name to ID mapping
        joint_name2id = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                joint_name2id[joint_name] = i

        # Get qpos addresses
        for joint_name in self.right_arm_joint_names:
            if joint_name in joint_name2id:
                self.right_arm_qpos_addrs.append(self.model.jnt_qposadr[joint_name2id[joint_name]])

        for joint_name in self.left_arm_joint_names:
            if joint_name in joint_name2id:
                self.left_arm_qpos_addrs.append(self.model.jnt_qposadr[joint_name2id[joint_name]])

        for joint_name in self.torso_joint_names:
            if joint_name in joint_name2id:
                self.torso_qpos_addrs.append(self.model.jnt_qposadr[joint_name2id[joint_name]])

        for joint_name in self.head_joint_names:
            if joint_name in joint_name2id:
                self.head_qpos_addrs.append(self.model.jnt_qposadr[joint_name2id[joint_name]])

        for joint_name in self.right_hand_joint_names:
            if joint_name in joint_name2id:
                self.right_hand_qpos_addrs.append(self.model.jnt_qposadr[joint_name2id[joint_name]])

        for joint_name in self.left_hand_joint_names:
            if joint_name in joint_name2id:
                self.left_hand_qpos_addrs.append(self.model.jnt_qposadr[joint_name2id[joint_name]])

        # Get qvel addresses
        self.right_arm_qvel_addrs = []
        self.left_arm_qvel_addrs = []
        self.torso_qvel_addrs = []
        self.head_qvel_addrs = []
        self.right_hand_qvel_addrs = []
        self.left_hand_qvel_addrs = []

        for joint_name in self.right_arm_joint_names:
            if joint_name in joint_name2id:
                self.right_arm_qvel_addrs.append(self.model.jnt_dofadr[joint_name2id[joint_name]])

        for joint_name in self.left_arm_joint_names:
            if joint_name in joint_name2id:
                self.left_arm_qvel_addrs.append(self.model.jnt_dofadr[joint_name2id[joint_name]])

        for joint_name in self.torso_joint_names:
            if joint_name in joint_name2id:
                self.torso_qvel_addrs.append(self.model.jnt_dofadr[joint_name2id[joint_name]])

        for joint_name in self.head_joint_names:
            if joint_name in joint_name2id:
                self.head_qvel_addrs.append(self.model.jnt_dofadr[joint_name2id[joint_name]])

        for joint_name in self.right_hand_joint_names:
            if joint_name in joint_name2id:
                self.right_hand_qvel_addrs.append(self.model.jnt_dofadr[joint_name2id[joint_name]])

        for joint_name in self.left_hand_joint_names:
            if joint_name in joint_name2id:
                self.left_hand_qvel_addrs.append(self.model.jnt_dofadr[joint_name2id[joint_name]])

        # Keep the old joint_ids for compatibility with torque application
        self.right_arm_joint_ids = [
            joint_name2id[name] for name in self.right_arm_joint_names if name in joint_name2id
        ]
        self.left_arm_joint_ids = [
            joint_name2id[name] for name in self.left_arm_joint_names if name in joint_name2id
        ]
        self.torso_joint_ids = [
            joint_name2id[name] for name in self.torso_joint_names if name in joint_name2id
        ]
        self.head_joint_ids = [
            joint_name2id[name] for name in self.head_joint_names if name in joint_name2id
        ]
        self.right_hand_joint_ids = [
            joint_name2id[name] for name in self.right_hand_joint_names if name in joint_name2id
        ]
        self.left_hand_joint_ids = [
            joint_name2id[name] for name in self.left_hand_joint_names if name in joint_name2id
        ]

    def _update_current_positions(self):
        """Update current joint positions from MuJoCo data."""
        self.q_current_right = np.array(
            [self.data.qpos[addr] for addr in self.right_arm_qpos_addrs]
        )
        self.q_current_left = np.array([self.data.qpos[addr] for addr in self.left_arm_qpos_addrs])
        self.q_current_torso = np.array([self.data.qpos[addr] for addr in self.torso_qpos_addrs])
        self.q_current_head = np.array([self.data.qpos[addr] for addr in self.head_qpos_addrs])

        self.qd_current_right = np.array(
            [self.data.qvel[addr] for addr in self.right_arm_qvel_addrs]
        )
        self.qd_current_left = np.array([self.data.qvel[addr] for addr in self.left_arm_qvel_addrs])
        self.qd_current_torso = np.array([self.data.qvel[addr] for addr in self.torso_qvel_addrs])
        self.qd_current_head = np.array([self.data.qvel[addr] for addr in self.head_qvel_addrs])

    def set_joint_goals(self, goals):
        """
        Set target joint angles from a dictionary.

        Args:
            goals: Dictionary containing 'q_goal_right', 'q_goal_left'.
        """
        if goals.get("q_goal_right") is not None:
            self.q_goal_right = goals["q_goal_right"]
        if goals.get("q_goal_left") is not None:
            self.q_goal_left = goals["q_goal_left"]
        
        # Openarm gripper range is 0.0 to 0.044
        if goals.get("left_gripper_val") is not None:
            val = goals["left_gripper_val"] - 0.01 # Offset to ensure fingers fully close
            pos = np.clip(val, 0.0, 0.044)
            self.q_goal_left_hand = np.array([pos])

        if goals.get("right_gripper_val") is not None:
            val = goals["right_gripper_val"] - 0.01
            pos = np.clip(val, 0.0, 0.044)
            self.q_goal_right_hand = np.array([pos])

    @staticmethod
    def real_angle(q_goal, q_curr):
        # Safety function to wrap the agle to [-pi, pi], avoid the sign flipping
        q_diff = q_goal - q_curr
        q_diff_wrapped = (q_diff + np.pi) % (2 * np.pi) - np.pi
        q_goal_real = q_curr + q_diff_wrapped

        return q_goal_real

    def update(self, engaged=True):
        """
        Compute and apply torques based on current goals.
        Note: Dynamic control is not fully implemented for OpenArm yet. Using position control is recommended.
        """
        self.update_position_control()

    def update_position_control(self):
        """
        Apply joint position goals directly to actuators (for position actuators).
        """
        self._update_current_positions()

        # Right arm
        if self.q_goal_right is not None:
            q_goal_right_real = self.real_angle(self.q_goal_right, self.q_current_right)
            for i, aid in enumerate(self.right_arm_actuator_ids):
                if i < len(q_goal_right_real):
                    self.data.ctrl[aid] = q_goal_right_real[i]

        # Left arm
        if self.q_goal_left is not None:
            q_goal_left_real = self.real_angle(self.q_goal_left, self.q_current_left)
            for i, aid in enumerate(self.left_arm_actuator_ids):
                if i < len(q_goal_left_real):
                    self.data.ctrl[aid] = q_goal_left_real[i]

        # Right Hand (2 fingers)
        if self.q_goal_right_hand is not None:
            for i, aid in enumerate(self.right_hand_actuator_ids):
                 # Both fingers move same amount
                 if i < len(self.q_goal_right_hand):
                    self.data.ctrl[aid] = self.q_goal_right_hand[i]

        # Left Hand (2 fingers)
        if self.q_goal_left_hand is not None:
            for i, aid in enumerate(self.left_hand_actuator_ids):
                if i < len(self.q_goal_left_hand):
                    self.data.ctrl[aid] = self.q_goal_left_hand[i]


    def update_sim_from_hardware(self, hardware_state):
        """
        Update MuJoCo simulation state directly from hardware readings.
        This bypasses physics simulation and sets joint positions/velocities directly.

        Args:
            hardware_state: Dictionary containing hardware joint states with keys:
                'q_right': right arm joint positions (7 values)
                'qd_right': right arm joint velocities (7 values)
                'q_left': left arm joint positions (7 values)
                'qd_left': left arm joint velocities (7 values)
                'q_torso': torso joint positions (6 values)
                'qd_torso': torso joint velocities (6 values)
                'q_head': head joint positions (2 values)
                'qd_head': head joint velocities (2 values)
                'q_right_hand': right hand joint positions (1 value)
                'qd_right_hand': right hand joint velocities (1 value)
                'q_left_hand': left hand joint positions (1 value)
                'qd_left_hand': left hand joint velocities (1 value)
        """
        # Update right arm positions and velocities
        if "q_right" in hardware_state and len(hardware_state["q_right"]) == len(
            self.right_arm_qpos_addrs
        ):
            for i, addr in enumerate(self.right_arm_qpos_addrs):
                self.data.qpos[addr] = hardware_state["q_right"][i]

        if "qd_right" in hardware_state and len(hardware_state["qd_right"]) == len(
            self.right_arm_qvel_addrs
        ):
            for i, addr in enumerate(self.right_arm_qvel_addrs):
                self.data.qvel[addr] = hardware_state["qd_right"][i]

        # Update left arm positions and velocities
        if "q_left" in hardware_state and len(hardware_state["q_left"]) == len(
            self.left_arm_qpos_addrs
        ):
            for i, addr in enumerate(self.left_arm_qpos_addrs):
                self.data.qpos[addr] = hardware_state["q_left"][i]

        if "qd_left" in hardware_state and len(hardware_state["qd_left"]) == len(
            self.left_arm_qvel_addrs
        ):
            for i, addr in enumerate(self.left_arm_qvel_addrs):
                self.data.qvel[addr] = hardware_state["qd_left"][i]


        # Update right hand positions and velocities
        if "q_right_hand" in hardware_state and len(hardware_state["q_right_hand"]) == len(
            self.right_hand_qpos_addrs
        ):
            for i, addr in enumerate(self.right_hand_qpos_addrs):
                self.data.qpos[addr] = hardware_state["q_right_hand"][i]

        if "qd_right_hand" in hardware_state and len(hardware_state["qd_right_hand"]) == len(
            self.right_hand_qvel_addrs
        ):
            for i, addr in enumerate(self.right_hand_qvel_addrs):
                self.data.qvel[addr] = hardware_state["qd_right_hand"][i]

        # Update left hand positions and velocities
        if "q_left_hand" in hardware_state and len(hardware_state["q_left_hand"]) == len(
            self.left_hand_qpos_addrs
        ):
            for i, addr in enumerate(self.left_hand_qpos_addrs):
                self.data.qpos[addr] = hardware_state["q_left_hand"][i]

        if "qd_left_hand" in hardware_state and len(hardware_state["qd_left_hand"]) == len(
            self.left_hand_qvel_addrs
        ):
            for i, addr in enumerate(self.left_hand_qvel_addrs):
                self.data.qvel[addr] = hardware_state["qd_left_hand"][i]

        # Update current position tracking variables for consistency
        self._update_current_positions()

    def get_current_joint_angles(self):
        """
        Get current joint angles for both arms and torso.
        """
        self._update_current_positions()
        return (
            self.q_current_right.copy(),
            self.q_current_left.copy(),
        )

    def get_sew_transform(self):
        """
        Returns a function that transforms points from SEW frame to World frame.
        """
        # Find OpenArm base body ID
        base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_body_link0")
        
        if base_body_id != -1:
            # Get body position in world frame
            pos_body_world = self.data.xpos[base_body_id]
            
            # Get body orientation in world frame
            rot_body_world = self.data.xmat[base_body_id].reshape(3, 3)
            
            # Offset from body origin to SEW origin (in body frame)
            offset = np.array([0, 0, 0.698])
            
            # SEW Origin in World Frame
            sew_origin_world = pos_body_world + rot_body_world @ offset
            sew_rot_world = rot_body_world
        else:
            # Fallback if body not found
            sew_origin_world = np.zeros(3)
            sew_rot_world = np.eye(3)

        def to_world(p_local):
            return sew_origin_world + sew_rot_world @ p_local
            
        return to_world
