import numpy as np

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, sub_goal, next_state, next_sub_goal, action, reward, done)
class ReplayBuffer(object):
	def __init__(self, c_step, max_size=2e5):
		self.step_storage = []
		self.episode_storage = []
		self.tmp_storage = [] # c_step 길이만큼만 받아서 episode_storage에 뭉텅이로 전달해주고 지워짐
		self.max_size = max_size
		self.c_step = c_step
		self.step_ptr = 0
		self.episode_ptr = 0

	def step_add(self, data):
		self.tmp_storage.append(data)
		if len(self.step_storage) == self.max_size:
			self.step_storage[int(self.step_ptr)] = data
			self.step_ptr = (self.step_ptr + 1) % self.max_size
		else:
			self.step_storage.append(data)




	def episode_add(self):
		if len(self.episode_storage) == (self.max_size/10) :
			self.episode_storage[int(self.episode_ptr)] = self.tmp_storage
			self.episode_ptr = (self.episode_ptr + 1) % (self.max_size/10)
		else :
			self.episode_storage.append(self.tmp_storage)
		self.tmp_storage = []


	def step_sample(self, batch_size):
		ind = np.random.randint(0, len(self.step_storage), size=batch_size)
		x, env_g, g, y, env_n_g, n_g, u, r, d, l_r = [], [], [], [], [], [], [], [], [], []

		for i in ind: 
			X, ENV_G, G, Y, ENV_N_G, N_G, U, R, D, L_R = self.step_storage[i]
			x.append(np.array(X, copy=False))
			env_g.append(np.array(ENV_G, copy=False))
			g.append(np.array(G, copy=False))
			y.append(np.array(Y, copy=False))
			env_n_g.append(np.array(ENV_N_G, copy=False))
			n_g.append(np.array(N_G, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))
			l_r.append(np.array(L_R, copy=False))

		return np.array(x), np.array(env_g), np.array(g), np.array(y), np.array(env_n_g), np.array(n_g), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), np.array(l_r).reshape(-1, 1)

	def episode_sample(self, batch_size):
		ind = np.random.randint(0, len(self.episode_storage), size=batch_size)
		t = []
		for i in ind :
			t.append(np.array(self.episode_storage[i], copy=False))

		return np.array(t)

class Normal_ReplayBuffer(object):
	def __init__(self, max_size=1e6):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in ind:
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

