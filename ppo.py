import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import time
#HYPERPARAMETERS
INITIALIZER = tf.glorot_uniform_initializer
CLIPPINGPARAMETER = .2
ACTOR_LEARN_RATE = 0.0003
CRITIC_LEARN_RATE = 0.0001
OPTIMIZATION_BATCH_SIZE = 1000
CREATION_BATCH_SIZE = 16
CREATION_STEPS = 4000
GAMMA = 0.97
ADVANTAGE_FUNCTION = 'Advantages'
ACTION_ACTIVATION = tf.nn.tanh
ITERATIONS = 1000
EPOCHS = 5
ENTROPY_LOSS_FACTOR = .01
KL_EARLY_STOPPING = 0.05
#ADVANTAGE_FUNCTION = 'Generalized_Advantages'

##NOTES
#This uses tanh for actions now
#Implements PPO
#Printing but not using KL divergence for early stopping right tensorflow
#Now normalizing advantages
#TODO: Why not use early stopping? Seems very very viable
#TODO: Maybe use Normalizing Wrapper for SubprocVecEnv
#TODO
class Network:
    def __init__(self, input_size=None, layers=None, outputs=None, name=None):
        """
        Args:
            input_size (int): Input size used for the input placeholder
            layers (list of dicts): List of layers, where each layers is a dict with keys:
                'size' : number of units in layer
                'activation': activation function for layer
            outputs: (list of dicts): List of output layers, where each layer is a dict with keys:
                'size' : number of units in layer
                'activation': activation function for respective layer
            name (str) : String naming the network
        """
        self.name = name
        self.save_list = []
        self.layer_list = []
        self.output_list = []

        #TODO
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, input_size), name='input_placeholder')

        #Create layers
        for i,layer in enumerate(layers):
            layer_input = self.input_placeholder if i==0 else self.layer_list[-1]
            layer_input_size = input_size if i==0 else layers[i-1]['size']
            layer_name = name+'_layer_'+str(i)
            self.layer_list.append(self._fully_connected_layer(layer_input, layer_input_size, layer['size'], layer['activation'], layer_name))

        #Create outputs
        for i, output in enumerate(outputs):
            output_name = name+'_output_'+str(i)
            self.output_list.append(self._fully_connected_layer(self.layer_list[-1], layers[-1]['size'], output['size'], output['activation'], output_name))



    def _fully_connected_layer(self, input, input_size, size, activation, name):

        """
        Takes batch of observations, applies a fully connected layer to it
        input (tensor): Input tensor
        input_size (tensor): size of input tensor at it's last dimension
        size (int) : number of units in layer
        name(str) : name of layer
        """
        bias_name = name + '_bias'
        weights_name = name + '_weights'

        bias = tf.get_variable(bias_name, shape=size, initializer=INITIALIZER)
        weights = tf.get_variable(weights_name, shape=[input_size, size], initializer=INITIALIZER)

        self.save_list.append(bias)
        self.save_list.append(weights)

        drive = tf. matmul(input, weights) + bias

        return drive if activation==None else activation(drive)

class PPO_model:
    def __init__(self, actor_description, critic_description, env_name, save_file=None):
        """
        Args:
        actor_description (list of dicts): List of layers, where each layers is a dict with keys:
            'size' : number of units in layer
            'activation function': activation function for layer
        critic_description (list of dicts): List of layers, where each layers is a dict with keys:
            'size' : number of units in layer
            'activation function': activation function for layer
        """
        self.env_name = env_name

        self.environments = SubprocVecEnv([self.env_function for env in range(CREATION_BATCH_SIZE)])
        #self.environments.reset()
        self.input_size = len(self.env_function().observation_space.high)
        self.output_size = len(self.env_function().action_space.high)

        self.session = tf.Session()
        self.actor = Network(self.input_size, actor_description, [{'size': self.output_size,'activation':ACTION_ACTIVATION}, {'size': self.output_size, 'activation': tf.nn.softplus}], 'actor')
        self.critic = Network(self.input_size, critic_description, [{'size':1, 'activation':None}], 'critic')

        self.save_list = self.actor.save_list + self.critic.save_list

        self.action_distribution = tfp.distributions.Normal(self.actor.output_list[0], self.actor.output_list[1])
        self.value_prediction = self.critic.output_list[0]
        #placeholders for training
        self.advantage_placeholder = tf.placeholder(tf.float32, shape=(None), name='advantage_placeholder')
        self.old_action_placeholder = tf.placeholder(tf.float32, shape=(None, self.output_size), name='old_action_placeholder')
        self.old_log_prob_placeholder = tf.placeholder(tf.float32, shape=(None), name='old_log_prob_placeholder')
        self.value_target_placeholder = tf.placeholder(tf.float32, shape=(None), name='value_target_placeholder')



        self.logger_dict = {'value_loss' : self.value_loss(), 'objective_function':self.clipped_ppo_objective(), 'entropy' : self.approx_entropy_loss()}
        self.logger_keys = list(self.logger_dict.keys())
        self.logger_fetches = [self.logger_dict[key] for key in self.logger_keys]
        self.iteration_logger = []
        self.epoch_logger = [[]]
        self.minibatch_logger = [[[]]]
        self.reward_logger = []

        self.actor_optimizer = tf.train.AdamOptimizer(ACTOR_LEARN_RATE)
        self.critic_optimizer = tf.train.AdamOptimizer(CRITIC_LEARN_RATE)
        self.retrieve_list = self.logger_fetches + self.train_step() + self.approximate_kl_divergence()

        if save_file==None:
            self.session.run(tf.global_variables_initializer())
        else:
            self.restore(save_file)


    def train(self):
        data = self.create_data()
        self.train_on_samples(data)

    def create_data(self):

        observation = self.environments.reset()
        action_fetch, log_prob_fetch = self.sample_action()

        fetch_list = [action_fetch, log_prob_fetch, self.value_prediction]

        observation_list = []
        action_list = []
        log_prob_list = []
        value_estimate_list = []
        reward_list = []
        done_list = []

        for step in range(CREATION_STEPS):
            observation_list.append(observation)
            feed_dict = self.feed_dictionary(observation)
            action, log_prob, value_estimate = self.session.run(fetch_list, feed_dict=feed_dict)
            observation, reward, done, info = self.env_step(action)
            action_list.append(action)
            log_prob_list.append(log_prob)
            value_estimate_list.append(value_estimate)
            reward_list.append(reward)
            done_list.append(done)

        done_array = np.stack(done_list)
        observation_array = np.stack(observation_list)
        action_array = np.stack(action_list)
        log_prob_array = np.stack(log_prob_list)
        value_estimate_array = np.stack(value_estimate_list)
        reward_array = np.stack(reward_list)
        value_target_array = self.compute_values(reward_array, done_array)
        advantage_array = None
        if ADVANTAGE_FUNCTION == 'Advantages':
            advantage_array = self.compute_advantages(value_target_array, value_estimate_array)
        elif ADVANTAGE_FUNCTION == 'Generalized_Advantages':
            advantage_array = self.compute_generalized_advantage_estimations(reward_array, value_target_array, value_estimate_array, dones)

        #Write down how good we are now on avaerage
        self.reward_logger.append(np.mean(reward_array))

        finished_samples = self.compute_finished_samples(done_array)
        num_finished_samples = np.sum(finished_samples)
        sample_number = CREATION_STEPS*CREATION_BATCH_SIZE

        print(str(num_finished_samples) + ' out of ' + str(sample_number) + 'usable')

        observation_array = np.reshape(observation_array, (sample_number, self.input_size))
        action_array = np.reshape(action_array, (sample_number, self.output_size))
        log_prob_array = np.reshape(log_prob_array, (sample_number))
        value_estimate_array = np.reshape(value_estimate_array, (sample_number))
        value_target_array = np.reshape(value_target_array, (sample_number))
        advantage_array = np.reshape(advantage_array, (sample_number))
        finished_samples = np.reshape(finished_samples, (sample_number))

        observation_array = observation_array[finished_samples]
        action_array = action_array[finished_samples]
        log_prob_array = log_prob_array[finished_samples]
        value_estimate_array = value_estimate_array[finished_samples]
        value_target_array = value_target_array[finished_samples]
        advantage_array = advantage_array[finished_samples]

        data = {'observations':observation_array, 'advantages':advantage_array, 'old_actions':action_array, 'old_log_probs':log_prob_array, 'target_values':value_target_array, 'data_length':num_finished_samples}
        return data

    def train_on_samples(self, data):
        """
        data (dict) : Has keys: 'advantages', 'old_actions', 'old_log_probs', 'target_values'
        """
        #Maybe split actor and critic training completely?
        approx_kl = 0
        for epoch in range(EPOCHS):
            mini_batch_list = self.generate_minibatches(data)
            for mini_batch in mini_batch_list:
                fetches = self.session.run(self.retrieve_list, feed_dict = mini_batch)
                #This ignores the last three outputs, they are the train steps and finally approx_kl and should not be logged
                approx_kl = fetches[-1]
                #print('approx_kl' + str(fetches[-1]))
                #print('approx_kl' + str(fetches[-1]))
                logging_fetches = fetches[0:len(self.logger_keys)]
                self.update_minibatch_logger(logging_fetches)
                if approx_kl > KL_EARLY_STOPPING:
                    break
            self.update_epoch_logger()
            if approx_kl > KL_EARLY_STOPPING:
                break
        self.update_iteration_logger()


    def generate_minibatches(self, data):
        """
        Args:
            data (dict) : Has keys: 'observations', 'advantages', 'old_actions', 'old_log_probs', 'target_values', 'data_length', each containing:
                data['observations'] (ndarray): shape(data_length, observation_size)
                data['advantages'] (ndarray): shape(data_length)
                data['old_actions'] (ndarray): shape(data_length, action_size)
                data['old_log_probs'] (ndarray): shape(data_length)
                data['target_values'] (ndarray): shape(data_length)
                data['data_length'] (int) : len of total data sequence

        Returns:
            minibatch_list with feed_dicts, each of length of OPTIMIZATION_BATCH_SIZE
        """
        #Does not have data_length for obvious reasons, observations are handled seperately, as the data structure prefers this
        data_key_list = ['advantages','old_actions','old_log_probs','target_values']
        placeholder_list = [self.advantage_placeholder, self.old_action_placeholder, self.old_log_prob_placeholder, self.value_target_placeholder]

        #Shuffle data
        random_permutation = np.random.permutation(range(data['data_length']))
        for key in data_key_list:
            data[key] = data[key][random_permutation]

        mini_batch_list = []
        indices = range(0,data['data_length'], OPTIMIZATION_BATCH_SIZE)
        for start_index in indices:
            end_index = start_index + OPTIMIZATION_BATCH_SIZE
            #Make sure indexing works
            minibatch_dict = self.feed_dictionary(data['observations'][start_index:end_index])
            for data_key, feed_key in zip(data_key_list, placeholder_list):
                minibatch_dict[feed_key] = data[data_key][start_index:end_index]
            mini_batch_list.append(minibatch_dict)
        return mini_batch_list

    def sample_action(self):
        """
        action of size [batch_size, action_size]
        log_probs are logarithm probabilities, size[batch_size]
        """
        actions = self.action_distribution.sample()
        log_probs = self.action_distribution.log_prob(actions)
        log_probs = tf.reduce_sum(log_probs, -1)
        return actions, log_probs

    def value_loss(self):
        value_f_loss = tf.square(self.value_prediction - self.value_target_placeholder)
        loss = tf.reduce_mean(value_f_loss)
        return loss


    def clipped_ppo_objective(self):
        prob_ratio = tf.exp(tf.reduce_sum(self.action_distribution.log_prob(self.old_action_placeholder), axis=-1)-self.old_log_prob_placeholder)
        clipped_prop_ratio = tf.clip_by_value(prob_ratio, 1-CLIPPINGPARAMETER, 1+CLIPPINGPARAMETER)
        clipped_objective = tf.minimum(prob_ratio*self.advantage_placeholder, clipped_prop_ratio*self.advantage_placeholder)
        loss = -tf.reduce_mean(clipped_objective)
        return loss

    def approx_entropy_loss(self):
        entropy_loss = tf.reduce_mean(self.action_distribution.log_prob(self.old_action_placeholder))
        return entropy_loss

    def policy_loss(self):
        return self.clipped_ppo_objective() + ENTROPY_LOSS_FACTOR*self.approx_entropy_loss()

    def approximate_kl_divergence(self):
        #Computes the approximate kl div between old and new log probs for early stopping
        return [tf.reduce_mean(self.old_log_prob_placeholder-tf.reduce_sum(self.action_distribution.log_prob(self.old_action_placeholder), axis=-1))]

    def feed_dictionary(self, observation, old_action=None, old_log_prob=None, value_target=None):
        feed_dict = {self.actor.input_placeholder : observation, self.critic.input_placeholder : observation}
        if old_action!=None:
            feed_dict[self.old_action_placeholder]=old_action
        if old_log_prob!=None:
            feed_dict[self.old_log_prob_placeholder]=old_log_prob
        if value_target!=None:
            feed_dict[self.value_target_placeholder]=value_target
        return feed_dict

    def env_step(self, action):
        self.environments.step_async(action)
        return self.environments.step_wait()

    def train_step(self):
        #TODO
        return [self.actor_optimizer.minimize(self.policy_loss()), self.critic_optimizer.minimize(self.value_loss())]

    def update_minibatch_logger(self, log_entries):
        log_dict = {}
        for entry, key in zip(log_entries, self.logger_keys):
            log_dict[key] = entry
        self.minibatch_logger[-1][-1].append(log_dict)

    def update_epoch_logger(self):
        last_entries = self.minibatch_logger[-1][-1]
        log_dict = {}
        for key in self.logger_keys:
            log_dict[key] = np.mean([dict[key] for dict in last_entries])
        self.epoch_logger[-1].append(log_dict)
        self.minibatch_logger[-1].append([])

    def update_iteration_logger(self):
        last_entries = self.epoch_logger[-1]
        log_dict = {}
        for key in self.logger_keys:
            log_dict[key] = np.mean([dict[key] for dict in last_entries])
        self.iteration_logger.append(log_dict)
        self.epoch_logger.append([])
        self.minibatch_logger.append([[]])

    def env_function(self):
        #env_string = self.env_name
        return gym.make(self.env_name)

    def compute_values(self, rewards, dones):

        """
        rewards (ndarray): rewards of shape (CREATION_STEPS, CREATION_BATCH_SIZE)
        dones (reward_array): dones of shape (CREATION_STEPS, CREATION_BATCH_SIZE)
        """
        assert rewards.shape == dones.shape
        assert rewards.shape[0] == dones.shape[0]
        assert rewards.shape[0] == CREATION_STEPS

        rewards = np.flip(rewards,axis=0)
        dones = np.flip(dones, axis=0)
        values = np.zeros(rewards.shape)

        accumulated_reward = np.zeros((CREATION_BATCH_SIZE))

        for i in range(CREATION_STEPS):
            #TODO check this again: is resetting at i correct, or rather like i +- 1?
            accumulated_reward[dones[i,:]] = 0
            accumulated_reward = accumulated_reward*GAMMA
            values[i,:] = rewards[i,:] + accumulated_reward
            accumulated_reward = values[i,:]
        values = np.flip(values,axis=0)
        return values

    def compute_advantages(self, values, value_estimations):
        advantages = values - np.squeeze(value_estimations)
        #Normalize advantages
        advantages = advantages - np.mean(advantages)
        advantages = advantages/np.std(advantages)
        return advantages

#TODO Generalized advantage Estimation
    """
    def compute_generalized_advantage_estimations(self, rewards, , values, value_estimations, dones):
        assert rewards.shape == value_estimations.shape
        assert rewards.shape == dones.shape
        deltas = np.zeros(rewards.shape)
        rewards = np.flip(rewards,axis=0)
        value_estimations = np.flip(rewards, axis=0)
        dones = np.flip(dones.flip(dones,axis=0))

        for i in range(CREATION_STEPS):
    """
    def save(self, file_name):
        stub = None
    #TODO
    def restore(self, file_name):
        stub = None

    def compute_finished_samples(self, dones):
        finished_samples = np.ones(dones.shape)==1
        run_done = dones[-1,:]
        for i in range(CREATION_STEPS):
            if np.all(run_done):
                break
            run_done = run_done + dones[-(i+1),:]
            finished_samples[-(i+1)] = run_done
        return finished_samples


def main():
    network_description = [{'size':32, 'activation':tf.nn.relu}, {'size':64, 'activation':tf.nn.relu},{'size':64, 'activation':tf.nn.relu}]
    trainer = PPO_model(network_description, network_description, 'LunarLanderContinuous-v2')
    for i in range(ITERATIONS):
        trainer.train()
        print('results: ')
        #print(trainer.epoch_logger)
        print(trainer.reward_logger)
        print('logger: ')
        print(trainer.minibatch_logger[-2])
if __name__ == '__main__':
    main()
