## mujoco 정리파일 (-ing)



```python
#input 값
#return 값 -line7 from humanoid.py
return self.__get__obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)
```

Input

```python
self.model = mujoco_py.load_model_from_path(fullpath)
self.sim = mujoco_py.MjSim(self.model)
self.data = self.sim.data
#mujoco_py에서 sim 가져옵니다.
```

```python
class mujoco_py.MjSim(model, data=None, nsubsteps=1, udd_callback=None)
```

[mujoco_py.MjSim API해당문서](https://openai.github.io/mujoco-py/build/html/reference.html#mujoco_py.MjSim)

MjSim represents a running simulation including its state.

Similar to Gym’s `MujocoEnv`, it internally wraps a `PyMjModel` and a `PyMjData`.

[PyMjData  해당문서](https://openai.github.io/mujoco-py/build/html/reference.html#pymjdata-time-dependent-data)

1. gym
2. Robotics



```python

self.init_qpos = self.sim.data.qpos.ravel().copy() # 다차원 배열을 1차원 배열로
self.init_qvel = self.sim.data.qvel.ravel().copy()
observation, _reward, done, _info = self.step(np.zeros(self.model.nu)) 
assert not done
self.obs_dim = observation.size



bounds = self.model.actuator_ctrlrange.copy()
low = bounds[:, 0]
high = bounds[:, 1]
self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

high = np.inf*np.ones(self.obs_dim)
low = -high
self.observation_space = spaces.Box(low, high, dtype=np.float32)

self.seed() # random위해 seed 생성

def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
```

data에 들어있는 것은

- `act`

- `act_dot`

- `active_contacts_efc_pos`

- `actuator_force`

- `actuator_length`

- `actuator_moment`

- `actuator_velocity`

- `body_jacp`

- `body_jacr`

- `body_xmat`

- `body_xpos`

- `body_xquat`

- `body_xvelp`

- `body_xvelr`

- `cacc`

- `cam_xmat`

- `cam_xpos`

- `cdof`

- `cdof_dot`

- `cfrc_ext`

- `cfrc_int`

- `cinert`

- `contact`

- `crb`

- `ctrl`

- `cvel`

- `efc_AR`

- `efc_AR_colind`

- `efc_AR_rowadr`

- `efc_AR_rownnz`

- `efc_D`

- `efc_J`

- `efc_JT`

- `efc_JT_colind`

- `efc_JT_rowadr`

- `efc_JT_rownnz`

- `efc_J_colind`

- `efc_J_rowadr`

- `efc_J_rownnz`

- `efc_R`

- `efc_aref`

- `efc_b`

- `efc_diagApprox`

- `efc_force`

- `efc_frictionloss`

- `efc_id`

- `efc_margin`

- `efc_solimp`

- `efc_solref`

- `efc_state`

- `efc_type`

- `efc_vel`

- `energy`

- `geom_jacp`

- `geom_jacr`

- `geom_xmat`

- `geom_xpos`

- `geom_xvelp`

- `geom_xvelr`

- `light_xdir`

- `light_xpos`

- `maxuse_con`

- `maxuse_efc`

- `maxuse_stack`

- `mocap_pos`

- `mocap_quat`

- `nbuffer`

- `ncon`

- `ne`

- `nefc`

- `nf`

- `nstack`

- `pstack`

- `qLD`

- `qLDiagInv`

- `qLDiagSqrtInv`

- `qM`

- `qacc`

- `qacc_unc`

- `qacc_warmstart`

- `qfrc_actuator`

- `qfrc_applied`

- `qfrc_bias`

- `qfrc_constraint`

- `qfrc_inverse`

- `qfrc_passive`

- `qfrc_unc`

- `qpos`

- `qvel`

- `sensordata`

- `set_joint_qpos`

- `set_joint_qvel`

- `set_mocap_pos`

- `set_mocap_quat`

- `site_jacp`

- `site_jacr`

- `site_xmat`

- `site_xpos`

- `site_xvelp`

- `site_xvelr`

- `solver`

- `solver_fwdinv`

- `solver_iter`

- `solver_nnz`

- `subtree_angmom`

- `subtree_com`

- `subtree_linvel`

- `ten_length`

- `ten_moment`

- `ten_velocity`

- `ten_wrapadr`

- `ten_wrapnum`

- `time`

- `timer`

- `userdata`

- `warning`

- `wrap_obj`

- `wrap_xpos`

- `xanchor`

- `xaxis`

- `xfrc_applied`

- `ximat`

- `xipos`



## Mujoco env 환경

```ptyhon
observation, _reward, done, _info = self.step(np.zeros(self.model.nu)) 
# observation은 
```

```python
>>> print(sim.data.qpos)
[0.  0.  1.4 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
>>> print(sim.data.qvel)
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.  0. 0. 0.]
>>> print(len(sim.data.qpos))
28
>>> print(len(sim.data.qvel))
27
```

```python
#아래 코드는 mujoco에서 제공하는 샘프로 코드 humanoid임... gym과는 다름... ㅠ ㅠ 
import mujoco_py
import os
mj_path, _ = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

sim.step()
print(sim.data.qpos)
# [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
#   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
#   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
#  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
#  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
#  -2.22862221e-05]


```



##  Gym 환경

### Humanoid 환경

```python
>>> import gym
>>> import numpy as np
>>> env = gym.make('Humanoid-v2')
>>> env.reset()
array([ 1.40919187e+00,  9.99939372e-01, -7.71515557e-03,  6.19486342e-04,
       -7.83231679e-03,  7.32990013e-03,  1.10215862e-03,  9.22152196e-03,
       -3.53687229e-03, -9.47177185e-03,  2.24031908e-03, -7.53064783e-03,
       -9.78305413e-03, -7.90378196e-03, -2.35972876e-03,  7.21348110e-03,
       -9.63757539e-03,  9.81365608e-03, -5.37928060e-03, -5.45530816e-03,
        4.62269139e-03, -2.30846754e-03, -6.07179910e-03, -7.80136491e-03,
       -4.62899115e-03,  3.13089568e-03, -4.99864454e-03, -9.77169365e-03,
        7.79317196e-03, -7.51343512e-03, -8.70429433e-03,  9.04579738e-03,
        8.86997241e-03,  1.35371253e-04,  2.62731035e-03, -5.96273831e-04,
       -4.16089135e-03, -1.42343764e-03,  9.92559119e-03,  7.25291998e-03,
        2.39257416e-03,  3.21586491e-03,  8.03763845e-03, -8.14553818e-03,
       -1.73986608e-03,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.23048629e+00,
        2.22138142e+00,  3.60175818e-02,  5.08580438e-04,  6.92649338e-02,
       -2.28067340e-02, -1.45719862e-01,  4.22829666e-02,  4.15173691e+00,
        8.32207894e+00,  8.75941005e-02,  8.52586837e-02,  8.65669965e-03,
       -9.48430053e-07,  1.03256258e-02, -1.73086837e-04, -5.19991100e-02,
        1.15694768e-03,  4.04227184e-01,  2.03575204e+00,  4.42095777e-02,
        4.22583228e-02,  4.96572605e-02, -3.53365805e-04,  8.86839205e-03,
        2.62871584e-04, -2.65017636e-01, -5.25535349e-03,  1.95838275e-01,
        5.85278711e+00,  2.47825773e-01,  2.07245699e-01,  5.44878712e-02,
       -1.11894233e-02, -1.98573257e-02, -7.68599391e-02, -1.14165924e-01,
       -4.43381354e-01, -7.95774731e-01,  4.52555626e+00,  8.75130894e-01,
        8.52177371e-01,  2.94415167e-02, -6.39981824e-03, -3.70873002e-02,
       -1.42505935e-01, -6.61732318e-02, -2.54556309e-01, -1.47121696e+00,
        2.63249442e+00,  1.03880128e+00,  1.02280378e+00,  2.23579875e-02,
       -4.52699073e-03, -3.48314474e-02, -1.32260552e-01, -4.58998850e-02,
       -1.74289172e-01, -1.34101095e+00,  1.76714587e+00,  2.46552906e-01,
        2.08662074e-01,  5.07748816e-02,  9.76573369e-03, -1.78664934e-02,
        7.40836067e-02, -1.03463383e-01,  4.26646504e-01, -8.01292363e-01,
        4.52555626e+00,  8.75600634e-01,  8.54897329e-01,  2.56875352e-02,
        4.42418949e-03, -2.69687268e-02,  1.33681550e-01, -4.88395732e-02,
        2.38539846e-01, -1.47432285e+00,  2.63249442e+00,  1.03947655e+00,
        1.02516668e+00,  1.91349593e-02,  2.50097375e-03, -2.08164387e-02,
        1.22639095e-01, -2.73892105e-02,  1.61362279e-01, -1.34307207e+00,
        1.76714587e+00,  4.21369777e-01,  3.29555083e-01,  1.13780212e-01,
        2.88445051e-02, -3.97973027e-02,  1.70819412e-01,  9.88779871e-02,
       -3.94461574e-01,  7.07453341e-01,  1.59405984e+00,  3.26212158e-01,
        3.46453814e-01,  1.67310656e-01,  7.44276068e-02, -1.52998263e-01,
        1.27011610e-01,  3.27153627e-01, -2.87772277e-01,  5.45640552e-01,
        1.19834313e+00,  4.11293226e-01,  3.17421761e-01,  1.18596914e-01,
       -3.15872654e-02, -4.27927247e-02, -1.70411037e-01,  1.08758061e-01,
        4.02529474e-01,  6.91480909e-01,  1.59405984e+00,  3.12298760e-01,
        3.36511289e-01,  1.70509615e-01, -7.59998737e-02, -1.51845891e-01,
       -1.23013870e-01,  3.33878141e-01,  2.87198022e-01,  5.30312746e-01,
        1.19834313e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.03897265e-03,
       -5.19718377e-03, -9.69684886e-03, -3.73163767e-03, -6.55453620e-03,
       -4.56384745e-03,  2.95585284e-03, -1.25888048e-02, -1.78848202e-03,
       -1.77108184e-03, -6.37364953e-03, -4.37417405e-03, -5.74800641e-03,
       -1.25170983e-02, -1.83779786e-03, -1.78063829e-03, -7.53750238e-03,
       -4.37980576e-03,  3.25020748e-03, -1.23691521e-02,  7.08154301e-03,
       -2.68338727e-03, -7.35882667e-03, -3.47203757e-03,  3.20353603e-03,
       -1.49959237e-02,  7.10708463e-03, -3.70730737e-03, -7.34005333e-03,
       -3.41231334e-03,  3.20353603e-03, -1.49959237e-02,  7.10708463e-03,
       -3.70730737e-03, -7.34005333e-03, -3.41231334e-03, -5.17590491e-03,
       -1.39599905e-02,  2.32139831e-03, -1.38037683e-03, -7.44075880e-03,
       -4.40130021e-03, -5.17902649e-03, -2.38855204e-02,  2.28666895e-03,
       -5.25514174e-03, -7.44027562e-03, -4.19111384e-03, -5.17902649e-03,
       -2.38855204e-02,  2.28666895e-03, -5.25514174e-03, -7.44027562e-03,
       -4.19111384e-03,  8.97333714e-03, -3.93557516e-03, -5.05778417e-03,
       -5.15276479e-03, -3.37980035e-03, -3.60929896e-03,  8.92324685e-03,
       -6.15537201e-03, -2.73146090e-03, -5.19875253e-03, -3.77332862e-03,
       -3.98579708e-03,  9.42994859e-03, -1.44019552e-02, -1.20165546e-02,
        6.10358184e-04, -3.28620450e-03, -5.57029198e-03,  9.45643751e-03,
       -1.31477215e-02, -1.08110131e-02,  6.15788761e-04, -3.48114641e-03,
       -5.36759563e-03,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
>>> len(env.reset())
376
```

--> [humanoid](https://github.com/openai/gym/blob/master/gym/envs/mujoco/assets/humanoid.xml) xml에 index가 있습니다.

### HalfCheetah 환경

```python
>>> env4 = gym.make('HalfCheetah-v2')
>>> env4.reset()
array([-0.01354274,  0.09147245,  0.01138785,  0.04450717,  0.06320186,
       -0.02112325, -0.01080774,  0.09767808,  0.05346551,  0.07971385,
       -0.09114062,  0.00974805,  0.02388392,  0.09041859,  0.16462125,
       -0.00306638, -0.21058064])
>>> len(env4.reset())
17

```

```python
    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        raise NotImplementedError
```

https://github.com/openai/gym/blob/master/gym/core.py

```html
<compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
```

라디안 단위의 앵글 사용, 직각좌표계 사용

```html
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
```

default 값은 - 



```python
>>> env2.step(1)
(array([ 0.06564904, -0.0929163 ,  0.24385769,  0.29318351,  0.33469796,
        0.30338688,  0.22153364,  0.25048561,  1.45700757, -1.06580746,
       -2.81455293,  9.48865455,  7.03090576,  9.46484916,  8.72155794,
        8.36623684,  5.21930428]), 0.9867045633683121, False, {'reward_run': 1.0867045633683121, 'reward_ctrl': -0.1})
>>> env2.step(2)
(array([-0.08439401, -0.20258574,  0.65312413,  0.65847164,  0.63724025,
        0.72622202,  0.8565884 ,  0.5151336 , -0.13310376, -1.31438361,
        0.65543312, -3.44283725,  0.34697721, -2.47633232, -0.88667629,
        0.82038081, -0.12822246]), -0.32865025571052386, False, {'reward_run': 0.07134974428947616, 'reward_ctrl': -0.4})
>>> len(env.step(1)[0])
17
## Init policy할 때 쓰이는
# Initialize policy
	if args.policy_name == "TD3": policy = TD3.TD3(state_dim, action_dim, max_action)
	elif args.policy_name == "OurDDPG": policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
	elif args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action)
##

>>> env.observation_space
Dict(achieved_goal:Box(3,), desired_goal:Box(3,), observation:Box(10,)) #box 오브젝트 형식
>>> env2.action_space
Box(6,)
>>> env2.action_space.high
array([1., 1., 1., 1., 1., 1.], dtype=float32)

## random action 취할 시 이용하는 .sample()
>>> env2.action_space.sample()
array([ 0.09762701,  0.43037874,  0.20552675,  0.08976637, -0.1526904 ,
        0.29178822], dtype=float32)
```



```python
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError
```





```python
# TD3에서 쓰는 HalfCheetah env
# openai/gym/gym/envs/mujoco/half_cheetah.py
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
```

그래서 한번 해봤습니다..

```python
>>> def evaluate_policy(policy, eval_episodes=10):
...     avg_reward = 0.
...     for _ in range(eval_episodes):
...             obs = env.reset()
...             done = False
...             while not done:
...                     action = policy.select_action(np.array(obs))
...                     obs, reward, done, _ = env.step(action)
...                     avg_reward += reward
...     avg_reward /= eval_episodes
...     print ("---------------------------------------")
...     print ("Evaluation over: ", eval_episodes , "episodes: ", avg_reward)
...     print ("---------------------------------------")
...     return avg_reward
...
>>> seed = 0
>>> env = gym.make('HalfCheetah-v2')
>>> env.seed(seed)
[0]
>>> torch.manual_seed(seed)
<torch._C.Generator object at 0x10b600ad0>
>>> np.random.seed(seed)
>>> state_dim = env.observation_space.shape[0]
>>> action_dim = env.action_space.shape[0]
>>> max_action = float(env.action_space.high[0])
>>> print(state_dim)
17
>>> print(action_dim)
6
>>> print(max_action)
1.0
>>> print(env.observation_space)
Box(17,)

>>> type(env.observation_space)
<class 'gym.spaces.box.Box'>

>>> type(env.action_space)
<class 'gym.spaces.box.Box'>

>>> print(env.action_space)
Box(6,)

>>> print(env.action_space.high)
[1. 1. 1. 1. 1. 1.]

>>> print(replay_buffer)
<utils.ReplayBuffer object at 0x10ece3cf8>

>>> policy = TD3.TD3(state_dim, action_dim, max_action)

>>> evaluations = [evaluate_policy(policy)]
---------------------------------------
Evaluation over:  10 episodes:  -1.418249014019275
---------------------------------------
```

그래서 box가 궁금했습니다..

```python
import numpy as np

import gym
from gym import logger

class Box(gym.Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.
    Example usage:
    self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
    """
    def __init__(self, low=None, high=None, shape=None, dtype=None):
        """
        Two kinds of valid input:
            Box(low=-1.0, high=1.0, shape=(3,4)) # low and high are scalars, and shape is provided
            Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            shape = low.shape
        else:
            assert np.isscalar(low) and np.isscalar(high)
            low = low + np.zeros(shape)
            high = high + np.zeros(shape)
        if dtype is None:  # Autodetect type
            if (high == 255).all():
                dtype = np.uint8
            else:
                dtype = np.float32
            logger.warn("gym.spaces.Box autodetected dtype as {}. Please provide explicit dtype.".format(dtype))
        self.low = low.astype(dtype)
        self.high = high.astype(dtype)
        gym.Space.__init__(self, shape, dtype)

    def sample(self):
        return gym.spaces.np_random.uniform(low=self.low, high=self.high + (0 if self.dtype.kind == 'f' else 1), size=self.low.shape).astype(self.dtype)

    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "Box" + str(self.shape)

    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)
```



### Hopper 환경

```python
>>> env5 = gym.make('Hopper-v2')
>>> env5.reset()
array([ 1.24865273e+00,  2.84226244e-03, -3.36845412e-03, -1.87042456e-03,
       -4.85806257e-03, -3.21682134e-03, -4.93706795e-03, -6.09194722e-04,
        4.74014768e-03,  2.52174078e-03, -3.99487536e-03])
>>> len(env5.reset())
11
```



## ROBOTICS 

### FetchReach 환경

```python
>>> env2 = gym.make('FetchReach-v0')
>>> len(env2.reset())
3
>>> env2.reset()
{'observation': array([ 1.34184371e+00,  7.49100477e-01,  5.34717228e-01,  1.89027457e-04,	7.77191143e-05,  3.43749435e-06, -1.26100357e-08, -9.04671898e-08,	4.55387076e-06, -2.13287826e-06]), 'achieved_goal': array([1.34184371, 0.74910048, 0.53471723]), 'desired_goal': array([1.34356719, 0.68918438, 0.65263931])}

>>> len(env.reset()['observation'])
10
>>> len(env.reset()['achieved_goal'])
3
>>> len(env.reset()['desired_goal'])
3
>>> type(env.reset()['desired_goal'])
<class 'numpy.ndarray'>
>>> type(env.reset()['achieved_goal'])
<class 'numpy.ndarray'>
>>> type(env.reset()['observation'])
<class 'numpy.ndarray'>
```

observation을 알기위해서는 her에서 어떤 식으로 데이터를 가공했는지 알 수 있는~/her/experiment/data_generation/fetch_data_generation.py 를 봐야 합니다. [link](https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py)

```python

"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []

def main():
    env = gym.make('FetchPickAndPlace-v0')
    numItr = 100
    initStateSpace = "random"
    env.reset()
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)

        
def goToGoal(env, lastObs):

    goal = lastObs['desired_goal']
    objectPos = lastObs['observation'][3:6]
    gripperPos = lastObs['observation'][:3]
    gripperState = lastObs['observation'][9:11]
    object_rel_pos = lastObs['observation'][6:9]
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object
    
    timeStep = 0 #count the total number of timesteps
    episodeObs.append(lastObs)

    while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
        env.render()
        action = [0, 0, 0, 0]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]*6

        action[len(action)-1] = 0.05 #open

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        gripperPos = obsDataNew['observation'][:3]
        gripperState = obsDataNew['observation'][9:11]
        object_rel_pos = obsDataNew['observation'][6:9]

    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i]*6

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        gripperPos = obsDataNew['observation'][:3]
        gripperState = obsDataNew['observation'][9:11]
        object_rel_pos = obsDataNew['observation'][6:9]


#여기서 ['observation'] [:11]의 정의가 나옵니다.
#[0..10] 의 값은 각각
#gripperPos(3개),objectPos(3개),object_rel_pos(3개),girpperState(2개) 입니다.
```

그렇다면, mujoco에서 env를 어떻게 정의했는지 봅시다



```xml
<?xml version="1.0" encoding="utf-8"?>
<mujoco>
	<compiler angle="radian" coordinate="local" meshdir="../stls/fetch" texturedir="../textures"></compiler>
	<option timestep="0.002">
		<flag warmstart="enable"></flag>
	</option>

	<include file="shared.xml"></include>
	
	<worldbody>
		<geom name="floor0" pos="0.8 0.75 0" size="0.85 0.7 1" type="plane" condim="3" material="floor_mat"></geom>
		<body name="floor0" pos="0.8 0.75 0">
			<site name="target0" pos="0 0 0.5" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere"></site>
		</body>

		<include file="robot.xml"></include>
		<!-- robot.xml도 봐야겠네요. -->
		<body pos="1.3 0.75 0.2" name="table0">
			<geom size="0.25 0.35 0.2" type="box" mass="2000" material="table_mat"></geom>
		</body>

		<light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1" name="light0"></light>
	</worldbody>
	
	<actuator></actuator>
</mujoco>
```

robot.xml파일에서 table0을 정의했으니 robot.xml파일을 뜯어봅시다!

```xml
<mujoco>
	<body mocap="true" name="robot0:mocap" pos="0 0 0">
		<geom conaffinity="0" contype="0" pos="0 0 0" rgba="0 0.5 0 0.7" size="0.005 0.005 0.005" type="box"></geom>
		<geom conaffinity="0" contype="0" pos="0 0 0" rgba="0 0.5 0 0.1" size="1 0.005 0.005" type="box"></geom>
		<geom conaffinity="0" contype="0" pos="0 0 0" rgba="0 0.5 0 0.1" size="0.005 1 0.001" type="box"></geom>
		<geom conaffinity="0" contype="0" pos="0 0 0" rgba="0 0.5 0 0.1" size="0.005 0.005 1" type="box"></geom>
	</body>
	<body childclass="robot0:fetch" name="robot0:base_link" pos="0.2869 0.2641 0">
		<joint armature="0.0001" axis="1 0 0" damping="1e+11" name="robot0:slide0" pos="0 0 0" type="slide"></joint>
		<joint armature="0.0001" axis="0 1 0" damping="1e+11" name="robot0:slide1" pos="0 0 0" type="slide"></joint>
		<joint armature="0.0001" axis="0 0 1" damping="1e+11" name="robot0:slide2" pos="0 0 0" type="slide"></joint>
		<inertial diaginertia="1.2869 1.2236 0.9868" mass="70.1294" pos="-0.0036 0 0.0014" quat="0.7605 -0.0133 -0.0061 0.6491"></inertial>
		<geom mesh="robot0:base_link" name="robot0:base_link" material="robot0:base_mat" class="robot0:grey"></geom>
		<body name="robot0:torso_lift_link" pos="-0.0869 0 0.3774">
			<inertial diaginertia="0.3365 0.3354 0.0943" mass="10.7796" pos="-0.0013 -0.0009 0.2935" quat="0.9993 -0.0006 0.0336 0.0185"></inertial>
			<joint axis="0 0 1" damping="1e+07" name="robot0:torso_lift_joint" range="0.0386 0.3861" type="slide"></joint>
			<geom mesh="robot0:torso_lift_link" name="robot0:torso_lift_link" material="robot0:torso_mat"></geom>
			<body name="robot0:head_pan_link" pos="0.0531 0 0.603">
				<inertial diaginertia="0.0185 0.0128 0.0095" mass="2.2556" pos="0.0321 0.0161 0.039" quat="0.5148 0.5451 -0.453 0.4823"></inertial>
				<joint axis="0 0 1" name="robot0:head_pan_joint" range="-1.57 1.57"></joint>
				<geom mesh="robot0:head_pan_link" name="robot0:head_pan_link" material="robot0:head_mat" class="robot0:grey"></geom>
				<body name="robot0:head_tilt_link" pos="0.1425 0 0.058">
					<inertial diaginertia="0.0063 0.0059 0.0014" mass="0.9087" pos="0.0081 0.0025 0.0113" quat="0.6458 0.66 -0.274 0.2689"></inertial>
					<joint axis="0 1 0" damping="1000" name="robot0:head_tilt_joint" range="-0.76 1.45" ref="0.06"></joint>
					<geom mesh="robot0:head_tilt_link" name="robot0:head_tilt_link" material="robot0:head_mat" class="robot0:blue"></geom>
					<body name="robot0:head_camera_link" pos="0.055 0 0.0225">
						<inertial diaginertia="0 0 0" mass="0" pos="0.055 0 0.0225"></inertial>
						<body name="robot0:head_camera_rgb_frame" pos="0 0.02 0">
							<inertial diaginertia="0 0 0" mass="0" pos="0 0.02 0"></inertial>
							<body name="robot0:head_camera_rgb_optical_frame" pos="0 0 0" quat="0.5 -0.5 0.5 -0.5">
								<inertial diaginertia="0 0 0" mass="0" pos="0 0 0" quat="0.5 -0.5 0.5 -0.5"></inertial>
								<camera euler="3.1415 0 0" fovy="50" name="head_camera_rgb" pos="0 0 0"></camera>
							</body>
						</body>
						<body name="robot0:head_camera_depth_frame" pos="0 0.045 0">
							<inertial diaginertia="0 0 0" mass="0" pos="0 0.045 0"></inertial>
							<body name="robot0:head_camera_depth_optical_frame" pos="0 0 0" quat="0.5 -0.5 0.5 -0.5">
								<inertial diaginertia="0 0 0" mass="0" pos="0 0 0" quat="0.5 -0.5 0.5 -0.5"></inertial>
							</body>
						</body>
					</body>
				</body>
			</body>
			<body name="robot0:shoulder_pan_link" pos="0.1195 0 0.3486">
				<inertial diaginertia="0.009 0.0086 0.0041" mass="2.5587" pos="0.0927 -0.0056 0.0564" quat="-0.1364 0.7624 -0.1562 0.613"></inertial>
				<joint axis="0 0 1" name="robot0:shoulder_pan_joint" range="-1.6056 1.6056"></joint>
				<geom mesh="robot0:shoulder_pan_link" name="robot0:shoulder_pan_link" material="robot0:arm_mat"></geom>
				<body name="robot0:shoulder_lift_link" pos="0.117 0 0.06">
					<inertial diaginertia="0.0116 0.0112 0.0023" mass="2.6615" pos="0.1432 0.0072 -0.0001" quat="0.4382 0.4382 0.555 0.555"></inertial>
					<joint axis="0 1 0" name="robot0:shoulder_lift_joint" range="-1.221 1.518"></joint>
					<geom mesh="robot0:shoulder_lift_link" name="robot0:shoulder_lift_link" material="robot0:arm_mat" class="robot0:blue"></geom>
					<body name="robot0:upperarm_roll_link" pos="0.219 0 0">
						<inertial diaginertia="0.0047 0.0045 0.0019" mass="2.3311" pos="0.1165 0.0014 0" quat="-0.0136 0.707 0.0136 0.707"></inertial>
						<joint axis="1 0 0" limited="false" name="robot0:upperarm_roll_joint"></joint>
						<geom mesh="robot0:upperarm_roll_link" name="robot0:upperarm_roll_link" material="robot0:arm_mat"></geom>
						<body name="robot0:elbow_flex_link" pos="0.133 0 0">
							<inertial diaginertia="0.0086 0.0084 0.002" mass="2.1299" pos="0.1279 0.0073 0" quat="0.4332 0.4332 0.5589 0.5589"></inertial>
							<joint axis="0 1 0" name="robot0:elbow_flex_joint" range="-2.251 2.251"></joint>
							<geom mesh="robot0:elbow_flex_link" name="robot0:elbow_flex_link" material="robot0:arm_mat" class="robot0:blue"></geom>
							<body name="robot0:forearm_roll_link" pos="0.197 0 0">
								<inertial diaginertia="0.0035 0.0031 0.0015" mass="1.6563" pos="0.1097 -0.0266 0" quat="-0.0715 0.7035 0.0715 0.7035"></inertial>
								<joint armature="2.7538" axis="1 0 0" damping="3.5247" frictionloss="0" limited="false" name="robot0:forearm_roll_joint" stiffness="10"></joint>
								<geom mesh="robot0:forearm_roll_link" name="robot0:forearm_roll_link" material="robot0:arm_mat"></geom>
								<body name="robot0:wrist_flex_link" pos="0.1245 0 0">
									<inertial diaginertia="0.0042 0.0042 0.0018" mass="1.725" pos="0.0882 0.0009 -0.0001" quat="0.4895 0.4895 0.5103 0.5103"></inertial>
									<joint axis="0 1 0" name="robot0:wrist_flex_joint" range="-2.16 2.16"></joint>
									<geom mesh="robot0:wrist_flex_link" name="robot0:wrist_flex_link" material="robot0:arm_mat" class="robot0:blue"></geom>
									<body name="robot0:wrist_roll_link" pos="0.1385 0 0">
										<inertial diaginertia="0.0001 0.0001 0.0001" mass="0.1354" pos="0.0095 0.0004 -0.0002"></inertial>
										<joint axis="1 0 0" limited="false" name="robot0:wrist_roll_joint"></joint>
										<geom mesh="robot0:wrist_roll_link" name="robot0:wrist_roll_link" material="robot0:arm_mat"></geom>
										<body euler="0 0 0" name="robot0:gripper_link" pos="0.1664 0 0">
											<inertial diaginertia="0.0024 0.0019 0.0013" mass="1.5175" pos="-0.09 -0.0001 -0.0017" quat="0 0.7071 0 0.7071"></inertial>
											<geom mesh="robot0:gripper_link" name="robot0:gripper_link" material="robot0:gripper_mat"></geom>
											<body name="robot0:gipper_camera_link" pos="0.055 0 0.0225">
												<body name="robot0:gripper_camera_rgb_frame" pos="0 0.02 0">
													<body name="robot0:gripper_camera_rgb_optical_frame" pos="0 0 0" quat="0.5 -0.5 0.5 -0.5">
														<camera euler="3.1415 0 0" fovy="50" name="gripper_camera_rgb" pos="0 0 0"></camera>
													</body>
												</body>
												<body name="robot0:gripper_camera_depth_frame" pos="0 0.045 0">
													<body name="robot0:gripper_camera_depth_optical_frame" pos="0 0 0" quat="0.5 -0.5 0.5 -0.5"></body>
												</body>
											</body>

											<body childclass="robot0:fetchGripper" name="robot0:r_gripper_finger_link" pos="0 0.0159 0">
												<inertial diaginertia="0.1 0.1 0.1" mass="4" pos="-0.01 0 0"></inertial>
												<joint axis="0 1 0" name="robot0:r_gripper_finger_joint" range="0 0.05"></joint>
												<geom pos="0 -0.008 0" size="0.0385 0.007 0.0135" type="box" name="robot0:r_gripper_finger_link" material="robot0:gripper_finger_mat" condim="4" friction="1 0.05 0.01"></geom>
											</body>
											<body childclass="robot0:fetchGripper" name="robot0:l_gripper_finger_link" pos="0 -0.0159 0">
												<inertial diaginertia="0.1 0.1 0.1" mass="4" pos="-0.01 0 0"></inertial>
												<joint axis="0 -1 0" name="robot0:l_gripper_finger_joint" range="0 0.05"></joint>
												<geom pos="0 0.008 0" size="0.0385 0.007 0.0135" type="box" name="robot0:l_gripper_finger_link" material="robot0:gripper_finger_mat" condim="4" friction="1 0.05 0.01"></geom>
											</body>
											<site name="robot0:grip" pos="0.02 0 0" rgba="0 0 0 0" size="0.02 0.02 0.02"></site>
										</body>
									</body>
								</body>
							</body>
						</body>
					</body>
				</body>
			</body>
		</body>
		<body name="robot0:estop_link" pos="-0.1246 0.2389 0.3113" quat="0.7071 0.7071 0 0">
			<inertial diaginertia="0 0 0" mass="0.002" pos="0.0024 -0.0033 0.0067" quat="0.3774 -0.1814 0.1375 0.8977"></inertial>
			<geom mesh="robot0:estop_link" rgba="0.8 0 0 1" name="robot0:estop_link"></geom>
		</body>
		<body name="robot0:laser_link" pos="0.235 0 0.2878" quat="0 1 0 0">
			<inertial diaginertia="0 0 0" mass="0.0083" pos="-0.0306 0.0007 0.0552" quat="0.5878 0.5378 -0.4578 0.3945"></inertial>
			<geom mesh="robot0:laser_link" rgba="0.7922 0.8196 0.9333 1" name="robot0:laser_link"></geom>
			<camera euler="1.55 -1.55 3.14" fovy="25" name="lidar" pos="0 0 0.02"></camera>
		</body>
		<body name="robot0:torso_fixed_link" pos="-0.0869 0 0.3774">
			<inertial diaginertia="0.3865 0.3394 0.1009" mass="13.2775" pos="-0.0722 0.0057 0.2656" quat="0.9995 0.0249 0.0177 0.011"></inertial>
			<geom mesh="robot0:torso_fixed_link" name="robot0:torso_fixed_link" class="robot0:blue"></geom>
		</body>
		<body name="robot0:external_camera_body_0" pos="0 0 0">
			<camera euler="0 0.75 1.57" fovy="43.3" name="external_camera_0" pos="1.3 0 1.2"></camera>
		</body>
	</body>
</mujoco>
```

