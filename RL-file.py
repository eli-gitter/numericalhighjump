# rl_jump_train.py
# -----------------
# Train a policy with PPO to maximize jump height in your MuJoCo model.
# 1) Edit MODEL_XML_PATH below to point to your bio-model.xml
# 2) pip install gymnasium[mujoco] stable-baselines3
# 3) python rl_jump_train.py

import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

# ====== EDIT THIS ======
MODEL_XML_PATH = r"/Users/eliajanes/Documents/GettingStarted/bio-model.xml"
# =======================


class JumpEnv(gym.Env):
    """
    Gymnasium environment wrapping a MuJoCo model to learn to jump as high as possible.

    Action:    3D continuous (hip, knee, ankle) normalized to [-1, 1], scaled to actuator ctrl ranges
    Obs:       [root_z, root_rot, hip, knee, ankle, z_vel, rot_vel, hip_vel, knee_vel, ankle_vel, contact(0/1)]
    Reward:    Shaped each step (upward velocity, height) + terminal bonus for peak height
    Episode:   Ends shortly after landing post-flight, or at a step/time cap
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path: str,
        max_steps: int = 800,          # ~8s if dt=0.01 and frame_skip=1
        frame_skip: int = 5,           # number of mj_step calls per env.step
        landing_grace: int = 15,       # steps to wait after first post-flight contact
        seed: int | None = 42,
        domain_randomization: bool = True,
    ):
        super().__init__()
        assert os.path.exists(model_path), f"XML not found: {model_path}"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.RandomState(seed)
        self.domain_randomization = domain_randomization

        # Resolve joint indices
        m = self.model
        self.joint_id = {
            'root_x'  : mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'root_x'),
            'root_z'  : mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'root_z'),
            'root_rot': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'root_rot'),
            'hip'     : mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'hip'),
            'knee'    : mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'knee'),
            'ankle'   : mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'ankle'),
        }
        self.qpos_adr = {k: m.jnt_qposadr[v] for k, v in self.joint_id.items()}
        self.qvel_adr = {k: m.jnt_dofadr[v]  for k, v in self.joint_id.items()}
        self.dof_idx  = {k: m.jnt_dofadr[v]  for k, v in self.joint_id.items()}

        # Resolve actuators
        self.actuator_idx = {
            'hip'  : mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_ctrl'),
            'knee' : mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_ctrl'),
            'ankle': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'ankle_ctrl'),
        }
        # Control ranges
        self.ctrl_range = {}
        for name, aidx in self.actuator_idx.items():
            if aidx >= 0 and m.actuator_ctrllimited[aidx]:
                lo, hi = m.actuator_ctrlrange[aidx]
            else:
                lo, hi = -1.0, 1.0  # fallback
            self.ctrl_range[name] = (float(lo), float(hi))
        print("[DBG] ctrl ranges:", self.ctrl_range)

        # Optional contact sensors
        self._contact_sensor_slices = []
        for sname in ['foot_contact_L', 'foot_contact_R', 'contact_L', 'contact_R',
                      'foot_L_force', 'foot_R_force', 'foot_force']:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            if sid != -1:
                adr = m.sensor_adr[sid]
                dim = m.sensor_dim[sid]
                self._contact_sensor_slices.append((adr, dim))

        # Spaces
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.landing_grace = landing_grace

        # 14 total elements: 10 state vars + 1 contact flag + 3 actuator forces
        obs_high = np.array([np.inf] * 13 + [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # action space unchanged
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)


        # Episode state
        self.steps = 0
        self.max_height = 0.0
        self.seen_flight = False
        self.post_landing_count = 0

        self._reset_state()

    # ---- utilities ----
    def _scale_and_set_ctrl(self, a):
        # a in [-1,1] -> actuator range
        for i, name in enumerate(['hip', 'knee', 'ankle']):
            lo, hi = self.ctrl_range[name]
            val = lo + 0.5*(float(a[i]) + 1.0)*(hi - lo)
            self.data.ctrl[self.actuator_idx[name]] = float(np.clip(val, lo, hi))

    def _has_ground_contact(self) -> bool:
        d = self.data
        if self._contact_sensor_slices:
            for adr, dim in self._contact_sensor_slices:
                if np.max(d.sensordata[adr:adr+dim]) > 0.1:
                    return True
            return False
        return d.ncon > 0

    def _obs(self):
        d = self.data

        # Base kinematic state
        obs = [
            d.qpos[self.qpos_adr['root_z']],     # vertical position
            d.qpos[self.qpos_adr['root_rot']],   # body rotation
            d.qpos[self.qpos_adr['hip']],        # joint angles
            d.qpos[self.qpos_adr['knee']],
            d.qpos[self.qpos_adr['ankle']],
            d.qvel[self.qvel_adr['root_z']],     # vertical velocity
            d.qvel[self.qvel_adr['root_rot']],   # angular velocity
            d.qvel[self.qvel_adr['hip']],        # joint angular velocities
            d.qvel[self.qvel_adr['knee']],
            d.qvel[self.qvel_adr['ankle']],
            1.0 if self._has_ground_contact() else 0.0,  # contact flag
        ]

        # Add actuator forces (torques) to observation — gives the agent a sense of effort
        hip_f   = float(d.qfrc_actuator[self.dof_idx['hip']])
        knee_f  = float(d.qfrc_actuator[self.dof_idx['knee']])
        ankle_f = float(d.qfrc_actuator[self.dof_idx['ankle']])
        obs.extend([hip_f, knee_f, ankle_f])

        return np.array(obs, dtype=np.float32)


    def _reset_state(self):
        mujoco.mj_resetData(self.model, self.data)

        # Base ICs (aligned with your provided script), with optional small randomization
        x_pos = -1.0
        z_pos = 0.2
        rot   = -0.5
        hip   = -0.8
        knee  = 1.0
        ankle = -0.4
        x_vel = 0.0
        z_vel = 0.0
        r_vel = 0.0

        if self.domain_randomization:
            z_pos += self.rng.uniform(-0.05, 0.05)
            rot   += self.rng.uniform(-0.05, 0.05)
            hip   += self.rng.uniform(-0.05, 0.05)
            knee  += self.rng.uniform(-0.05, 0.05)
            ankle += self.rng.uniform(-0.05, 0.05)
            x_vel += self.rng.uniform(-0.2, 0.2)
            z_vel += self.rng.uniform(-0.1, 0.1)
            r_vel += self.rng.uniform(-0.2, 0.2)

        d = self.data
        d.qpos[self.qpos_adr['root_x']]   = x_pos
        d.qpos[self.qpos_adr['root_z']]   = z_pos
        d.qpos[self.qpos_adr['root_rot']] = rot
        d.qpos[self.qpos_adr['hip']]      = hip
        d.qpos[self.qpos_adr['knee']]     = knee
        d.qpos[self.qpos_adr['ankle']]    = ankle

        d.qvel[self.qvel_adr['root_x']]   = x_vel
        d.qvel[self.qvel_adr['root_z']]   = z_vel
        d.qvel[self.qvel_adr['root_rot']] = r_vel
        d.qvel[self.qvel_adr['hip']]      = 0.0
        d.qvel[self.qvel_adr['knee']]     = 0.0
        d.qvel[self.qvel_adr['ankle']]    = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        self.max_height = float(d.qpos[self.qpos_adr['root_z']])
        self.seen_flight = False
        self.post_landing_count = 0

    # ---- Gym API ----
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        # --- Apply action ---
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._scale_and_set_ctrl(a)

        # --- Integrate physics ---
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        d = self.data
        self.steps += 1
        root_z = float(d.qpos[self.qpos_adr['root_z']])
        vz = float(d.qvel[self.qvel_adr['root_z']])
        contact = self._has_ground_contact()

        # --- Track max height ---
        self.max_height = max(self.max_height, root_z)

        # --- Termination conditions ---
        terminated = False
        truncated = False
        if not contact:
            self.seen_flight = True
        if self.seen_flight and contact:
            self.post_landing_count += 1
            if self.post_landing_count >= self.landing_grace:
                terminated = True
        if self.steps >= self.max_steps:
            truncated = True

        # ==============================================================
        # PHASE-BASED REWARD
        # ==============================================================

        reward = 0.0
        # joint actuator forces
        hip_f   = abs(d.qfrc_actuator[self.dof_idx['hip']])
        knee_f  = abs(d.qfrc_actuator[self.dof_idx['knee']])
        ankle_f = abs(d.qfrc_actuator[self.dof_idx['ankle']])
        total_force = hip_f + knee_f + ankle_f

        if contact:
            # Phase 1: Grounded (loading + push-off)
            # Reward producing upward force while on the ground
            reward += 0.0002 * total_force
            # Encourage upward velocity while still touching the ground
            reward += 0.005 * max(vz, 0.0)
        else:
            # Phase 2: Flight
            # Reward reaching and maintaining height
            reward += 0.05 * max(root_z - 0.2, 0.0)
            # Small reward for being stable in air (low angular velocity)
            reward -= 0.001 * abs(d.qvel[self.qvel_adr['root_rot']])

        # Small penalty for wasted control effort
        reward -= 1e-4 * np.sum(a * a)

        # Big terminal bonus for actual achieved height
        if terminated or truncated:
            reward += 100.0 * max(self.max_height - 0.2, 0.0)

        # ==============================================================
        obs = self._obs()
        info = {"max_height": self.max_height}
        return obs, float(reward), terminated, truncated, info



def make_env():
    return JumpEnv(
        model_path=MODEL_XML_PATH,
        max_steps=1200,
        frame_skip=4,
        landing_grace=20,
        domain_randomization=True,
    )


from stable_baselines3 import PPO

def train_and_save(total_timesteps: int = 500_000, out_path: str = "policies/ppo_jump_height"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    env = DummyVecEnv([lambda: make_env()])

    # Check if a previous model exists so we can keep improving it
    if os.path.exists(out_path + ".zip"):
        print("[INFO] Loading existing model for continued training...")
        model = PPO.load(out_path, env=env)
        # Continue training from previous state (don't reset timesteps)
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    else:
        print("[INFO] Starting new model training...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=4096,
            batch_size=512,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.3,
            ent_coef=0.01,         # encourage exploration
            learning_rate=1e-4,
            vf_coef=0.5,
        )
        model.learn(total_timesteps=total_timesteps)

    # Save after training (new or continued)
    model.save(out_path)
    print(f"[OK] Saved/updated policy to {out_path}.zip")

    # Quick evaluation (headless)
    vec_env = env
    obs = vec_env.reset()
    max_seen = 0.0
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        if isinstance(infos, list) and len(infos) > 0 and "max_height" in infos[0]:
            max_seen = max(max_seen, float(infos[0]["max_height"]))
        if dones[0]:
            break
    print(f"[EVAL] Peak root_z observed in eval: {max_seen:.3f} m")



if __name__ == "__main__":
    if not os.path.exists(MODEL_XML_PATH):
        raise FileNotFoundError(f"MuJoCo XML not found at: {MODEL_XML_PATH}")
    train_and_save()
