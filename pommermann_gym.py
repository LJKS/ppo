import Pommermann
import ray
import tensorflow as tf


"Pommermann Vectorized Environment set, wich can contain a set of different opponents, which are sampled by some curriculum designer"
class Pommermann_Vec_Env:
    def __init__(self, size, num_enemies, num_sampled_per_enemy, curriculum_designer, model_load_name, num_envs, input_size, layers, outputs, name):

        assert size%num_enemies == 0
        self.observation_size = create_pommermann_state_vec_from_dict(Pommermann.make().reset()).shape
        self.model_load_name = name
        self.curriculum_designer = 3#TODO curriculum_designer(num_enemies)
        #opponents are iteration numbers to load from
        self.sampled_opponent_iterations = self.curriculum_designer.sample_opponents()
        self.num_sub_envs = num_enemies
        self.num_envs_per_sub_env = size/num_enemies
        self.sub_envs = [Pommermann_Env_Batch(num_envs_per_sub_env, input_size, layers, outputs, name, file_name, self.sampled_opponent_iterations[i]) for i in range(num_enemies)]

        test_num_enemies, _ = self.reset().shape
        assert test_num_enemies == num_enemies

    def finish_iteration(self):
        average_rewards_in_envs = [env_batch.compute_average_reward() for env_batch in self.sub_envs]
        self.curriculum_designer.update(average_rewards)
        self.opponents = self.curriculum_designer.sample_opponents()

    @ray.method(num_return_values=3)
    def step(self, action):
        state_list = []
        reward_list = []
        done_list = []
        batch_wise_actions = np.split(action, self.num_sub_envs)
        batch_wise_result_ids = [self.sub_envs[i].step(batch_wise_actions[i]).remote() for i in range(self.num_sub_envs)]
        batch_wise_results = ray.get(batch_wise_result_ids)
        observations = np.concatenate([batch_wise_results[i][0] for i in range(self.num_sub_envs)])
        rewards = np.concatenate([batch_wise_results[i][0] for i in range(self.num_sub_envs)])
        dones = np.concatenate([batch_wise_results[i][0] for i in range(self.num_sub_envs)])
        return observations, rewards, dones

    def reset(self):
        observation_ids = [env.reset.remote() for env in self.sub_envs]
        observation_list = ray.get(observation_ids)
        return np.concatenate(observations)


"represents a batch of Pommermann Envs, which share the same opponent, modeled by a single network"
@ray.remote
class Pommermann_Env_Batch:
    def __init__(self, num_envs, input_size, layers, outputs, name, file_name, iteration):
        self.envs = [Pommermann_Env() for _ in range(size)]
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.sampled_opponent_network = Network(input_size, layers, outputs, name, self.graph)
        self.sampled_opponent_network.restore(self.session, file_name, iteration)

        #these can be used to update the curriculum_designer
        self.num_done = 0
        self.reward_aggregator = 0

        with self.graph.as_default():
            self.sampled_opponent_action_distribution = tfp.distributions.Categorical(probs=self.sampled_opponent_network.output_list[0])
            self.sampled_opponent_action_sample = self.sampled_opponent_action_distribution.sample()

    def sample_actions_from_sampled_opponent(self, observations):
        feed_dict = {self.sampled_opponent_network.input_placeholder:observations}
        sampled_actions_from_sampled_opponent = self.session.run(self.sampled_opponent_action_sample, feed_dict=feed_dict)
        return sampled_actions_from_sampled_opponent

    def step(self, actions):
        #actions is np.array of size [num_envs]
        sampled_opponent_observations = np.stack([create_pommermann_state_vec_from_dict(environment.state[environment.sampled_opponent_agent_position]) for environment in self.envs])
        sampled_opponent_actions = self.sample_actions_from_sampled_opponent(sampled_opponent_observations)
        state_list = []
        reward_list = []
        done_list = []
        for i, env in enumerate(self.envs):
            env_state, env_reward, env_done = env.step(actions[i], sampled_opponent_actions[i])
            state_list.append(env_state)
            reward_list.append(env_reward)
            done_list.append(env_done)
        env_batch_state = np.stack(state_list)
        env_batch_reward = np.stack(reward_list)
        env_batch_done = np.stack(done_list)

        self.reward_aggregator = self.reward_aggregator + np.sum(env_batch_reward)
        self.num_done = self.num_done + np.sum(env_batch_done)

        return env_batch_state, env_batch_reward, env_batch_done


    #TODO compute average reward per step or per done?
    def compute_average_reward(self):
        return self.reward_aggregator

    def reset(self):
        env_batch_state = np.stack([env.reset() for env in self.envs])
        self.reward_aggregator = 0
        self.num_done = 0
        return env_batch_state



"""A single Pommermann Env, which computes the dynamics of one environment
the environment is self-resetting, i.e. it resets when done and directly returns the first observation of the next sequence"""
class Pommermann_Env:
    def __init__(self):
        permutation = shuffle([0,1])

        self.ppo_model_agent_position = permutation[0]
        self.ppo_model_agent = Pommer_Agent(ppo_model_agent_position)

        self.sampled_opponent_agent_positions = permutation[1]
        self.sampled_opponent_agent = Pommer_Agent(sampled_opponent_agent_position)

        self.env = Pommermann.make(sorted([ppo_model_agents, sampled_opponent_agents]), key = lambda agent: agent.env_position)

        self.state = self.env.reset()



    def step(self, ppo_model_action, sampled_opponent_action):
        self.ppo_model_agent.set_action(ppo_model_action)
        self.sampled_opponent_agent.set_action(sampled_opponent_action)
        feed_action = self.env.act(state)
        state, reward, done, info = self.env.step(feed_action)
        if done:
            self.state = self.env.reset()
        else:
            self.state = state
        vec_state = create_pommermann_state_vec_from_dict(self.state[self.ppo_model_agent_position])

        return vec_state[ppo_model_agent], reward[ppo_model_agent], done, info

    def reset(self):
        permutation = shuffle([0,1])

        self.ppo_model_agent_position = permutation[0]
        self.ppo_model_agent = Pommer_Agent(ppo_model_agent_position)

        self.sampled_opponent_agent_positions = permutation[1]
        self.sampled_opponent_agent = Pommer_Agent(sampled_opponent_agent_position)

        self.state = self.env.reset()

        return create_pommermann_state_vec_from_dict(self.state[self.ppo_model_agent_position])




"Utility to control a single agent in the environment"
class Pommer_Agent:
    def __init__(self, env_position):
        #Position of the agent in the input of the env
        self.agent_position = env_position
        #Action the agent is supposed to dump
        self.action = None

    def act(self):
        return self.action

    def set_action(self, action):
        self.action = action

class Self_play_curriculum:
    def __init__(self, num_samples_per_iteration):
        self.iteration = 0
        self.num_samples = num_samples_per_iteration

    def update(self, rewards):
        self.iteration = self.iteration + 1

    def sample_opponents(self):
        return[self.iteration for i in range(self.num_samples)]

def create_pommermann_state_vec_from_dict(env_dict):
    keys = ['board', 'bomb_blast_strength', 'bomb_life', 'position', 'ammo']
    action_vector = np.concatenate([action_dict[key].flatten for key in keys])
    return action_vector
