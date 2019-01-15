from collections import deque

from numpy import linalg as LA
import numpy as np
import pickle
from mujoco_py import MujocoException

from baselines.her.util import convert_episode_to_batch_major, store_args


class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

        ############################################ hrl multi agent ###################################################
        self.initial_high_goal_gt_tilda = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_high_goal_gt = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        self.high_level_train_step = 10
        self.discount = 0.99
        self.total_timestep = 0
        ################################################################################################################
    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []

        ####################### hrl #############################

        Rt_high_sum = np.zeros((self.rollout_batch_size, 1), np.float32)
        total_timestep = 1
        high_goal_gt = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        #high_goal_gt_tilda = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        high_old_obj_st = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)

        u_temp = np.empty((self.rollout_batch_size, self.dims['u']), np.float32)

        low_nn_at = np.zeros((self.high_level_train_step*self.rollout_batch_size, self.dims['u']),
                                  np.float32).reshape(self.rollout_batch_size, self.high_level_train_step, self.dims['u'])
        low_nn_st = np.zeros((self.high_level_train_step*self.rollout_batch_size, self.dims['o']),
                                  np.float32).reshape(self.rollout_batch_size, self.high_level_train_step, self.dims['o'])
        intrinsic_reward = np.zeros((self.rollout_batch_size, 1), np.float32)

        high_goal_gt[:] = self.initial_high_goal_gt
        #high_goal_gt_tilda[:] = self.initial_high_goal_gt_tilda

        ##########################################################

        for t in range(self.T):
            #print_point
            #print("cont t : ", t)
            #print("cont total_timestep : ", total_timestep)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            reward_new = np.zeros(self.rollout_batch_size)
            done_new = np.zeros(self.rollout_batch_size)

            # compute new states and observations
            for i in range(self.rollout_batch_size):
                #print_point
                #print(" i : ", i)

                policy_output = self.policy.get_low_actions(
                    # o, ag, self.g,
                    o[i], ag[i], high_goal_gt[i],
                    compute_Q=self.compute_Q,
                    noise_eps=self.noise_eps if not self.exploit else 0.,
                    random_eps=self.random_eps if not self.exploit else 0.,
                    use_target_net=self.use_target_net)
                if self.compute_Q:
                    # u, Q = policy_output
                    u = policy_output
                    ## print_point
                    #print(" self.compute_Q u : ", u)
                    Q = self.policy.Get_Q_value(o[i], high_goal_gt[i], u)
                    Qs.append(Q)
                else:
                    u = policy_output
                    ## print_point
                    #print(" self.compute_Q else u : ", u)

                if u.ndim == 1:
                    # The non-batched case should still have a reasonable shape.
                    u = u.reshape(1, -1)

                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    # curr_o_new, _, _, info = self.envs[i].step(u[i])
                    ##################################### hrl ###############################
                    #curr_o_new, reward, done, info = self.envs[i].step(u[i])  # jangikim
                    #print("u.reshape(4,)", u.reshape(4,))
                    curr_o_new, reward, done, info = self.envs[i].step(u.reshape(4,))  # jangikim
                    #########################################################################
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    #jangikim
                    reward_new[i] = reward

                    ## print_point
                    #print(" curr_o_new [0] : ".format(i), curr_o_new)

                    #done_new[i] = done
                    #if success[i] == 1 or done==1:

                    if success[i] == 1:
                    #    done_new[i] = 1
                        print("done_new[{0}] : ".format(i), 1)
                    #else:
                    #    done_new[i] = 0

                    #done_new[i] = 0 if t + 1 == self.T else float(done)
                    done_new[i] = 0 if total_timestep == self.T else float(done)

                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()

                except MujocoException as e:
                    return self.generate_rollouts()

                low_nn_at[i][t % self.high_level_train_step] = u
                low_nn_st[i][t % self.high_level_train_step] = o_new[i]
                Rt_high_sum[i] += reward_new[i]

                if total_timestep % self.high_level_train_step == 0:


                    high_goal_gt[i] = self.policy.get_high_goal_gt(o[i], ag[i], self.g[i],
                                                                   compute_Q=self.compute_Q,
                                                                   noise_eps=self.noise_eps if not self.exploit else 0.,
                                                                   random_eps=self.random_eps if not self.exploit else 0.,
                                                                   use_target_net=self.use_target_net)
                    '''
                    high_goal_gt_tilda[i] = self.policy.get_high_goal_gt_tilda(high_old_obj_st[i], ag[i], self.g[i],
                                                                           o_new[i],
                                                                           low_nn_st[i],
                                                                           low_nn_at[i])
                    '''
                    self.policy.update_meta_controller(self.g[i],
                                                       Rt_high_sum[i] * 0.1,
                                                       done_new[i],
                                                       low_nn_st[i],
                                                       low_nn_at[i],
                                                       int((self.total_timestep + 1) / self.high_level_train_step),
                                                       ag[i])

                    high_old_obj_st[i] = o_new[i]
                    low_nn_at[i] = np.zeros((self.high_level_train_step, self.dims['u']), np.float32)
                    low_nn_st[i] = np.zeros((self.high_level_train_step, self.dims['o']), np.float32)
                    Rt_high_sum[i] = 0
                else:
                    high_goal_gt[i] = o[i] + high_goal_gt[i] - o_new[i]

                u_temp[i] = u
                #temp_test = (t % self.high_level_train_step)
                intrinsic_reward[i] = -LA.norm(o[i] + high_goal_gt[i] - o_new[i])

                self.policy.update_controller(o[i], o_new[i], high_goal_gt[i], u, intrinsic_reward[i],
                                              done_new[i],
                                              total_timestep)

            total_timestep += 1
            self.total_timestep += 1
            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            #acts.append(u.copy())
            acts.append(u_temp.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new

        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o
        ########################## hrl #########################
        self.initial_high_goal_gt[:] = high_goal_gt
        #self.initial_high_goal_gt_tilda[:] = high_goal_gt_tilda
        ########################################################
        episode = dict(o=obs,
                   u=acts,
                   g=goals,
                   ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size


        return convert_episode_to_batch_major(episode)




    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)
