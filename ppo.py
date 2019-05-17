import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
import time
np.set_printoptions(threshold=999999)
#HYPERPARAMETERS
INITIALIZER = tf.glorot_uniform_initializer
CLIPPINGPARAMETER = 0.2
ACTOR_LEARN_RATE = 0.0003
CRITIC_LEARN_RATE = 0.0005
OPTIMIZATION_BATCH_SIZE = 256
CREATION_BATCH_SIZE = 8
CREATION_STEPS = 2000
GAMMA = 0.97
GAE_LAMBDA = 0.95
ADVANTAGE_FUNCTION = 'Generalized_Advantages'
ACTION_ACTIVATION = tf.nn.tanh
ITERATIONS = 1000
EPOCHS = 5
ENTROPY_LOSS_FACTOR = 0.005
KL_EARLY_STOPPING = 0.05
#ADVANTAGE_FUNCTION = 'Generalized_Advantages'

##NOTES
#This uses tanh for actions now
#Implements PPO
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

        #input placeholder
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
        Used for Network models, implements a single fully connected layer
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

        drive = tf.matmul(input, weights) + bias

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
        #String used by gym to create environment
        self.env_name = env_name

        #core is a parallelized model of the environments, environments is inside a normalizing wrapper, that uses an updating average
        self.env_core = SubprocVecEnv([self.env_function for env in range(CREATION_BATCH_SIZE)])
        self.environments = VecNormalize(self.env_core)
        # If not want to use the normalizing wrapper:
        #self.environments = self.env_core
        self.environments.reset()

        #Input size and output size of network
        self.input_size = len(self.env_function().observation_space.high)
        self.output_size = len(self.env_function().action_space.high)

        #The tensorflow model
        self.session = tf.Session()
        self.actor = Network(self.input_size, actor_description, [{'size': self.output_size,'activation':ACTION_ACTIVATION}, {'size': self.output_size, 'activation': tf.nn.sigmoid}], 'actor')
        self.critic = Network(self.input_size, critic_description, [{'size':1, 'activation':None}], 'critic')

        #The list of variables, which should be saved
        self.save_list = self.actor.save_list + self.critic.save_list

        # THis has to be normal for the entropy to be correctly computed! If you change this, also change entropy computation
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
        value_estimate_array = np.squeeze(np.stack(value_estimate_list))
        reward_array = np.stack(reward_list)
        value_target_array = self.compute_values(reward_array, done_array)
        advantage_array = None
        if ADVANTAGE_FUNCTION == 'Advantages':
            advantage_array = self.compute_advantages(value_estimate_array, reward_array, done_array)
        elif ADVANTAGE_FUNCTION == 'Generalized_Advantages':
            advantage_array = self.compute_generalized_advantage_estimations(value_estimate_array, reward_array, done_array)

        #Write down how good we are now on avaerage
        self.reward_logger.append(np.mean(reward_array))

        finished_samples = self.compute_finished_samples(done_array)
        num_finished_samples = np.sum(finished_samples)
        sample_number = CREATION_STEPS*CREATION_BATCH_SIZE

        print(str(num_finished_samples) + ' out of ' + str(sample_number) + 'usable')
        print(str(num_finished_samples/np.sum(done_array)) + ' steps per run')
        #print(list(zip(value_estimate_array[0:200,1].tolist(), value_target_array[0:200,1].tolist())))
        observation_array = np.reshape(observation_array, (sample_number, self.input_size))
        action_array = np.reshape(action_array, (sample_number, self.output_size))
        log_prob_array = np.reshape(log_prob_array, (sample_number))
        value_estimate_array = np.reshape(value_estimate_array, (sample_number))
        value_target_array = np.reshape(value_target_array, (sample_number))
        advantage_array = np.reshape(advantage_array, (sample_number))
        finished_samples = np.reshape(finished_samples, (sample_number))
        #print(list(zip(value_target_array[0:500].tolist(), np.reshape(done_array, (sample_number))[0:500].tolist(), np.reshape(reward_array,(sample_number)).tolist())))
        observation_array = observation_array[finished_samples]
        action_array = action_array[finished_samples]
        log_prob_array = log_prob_array[finished_samples]
        value_estimate_array = value_estimate_array[finished_samples]
        value_target_array = value_target_array[finished_samples]
        advantage_array = advantage_array[finished_samples]
        print('Targets: Mean: ' + str(np.mean(value_target_array)) + ' Std: ' + str(np.std(value_target_array)))
        print('Estimates: Mean: ' + str(np.mean(value_estimate_array)) + ' Std: ' + str(np.std(value_estimate_array)))
        data = {'observations':observation_array, 'advantages':advantage_array, 'old_actions':action_array, 'old_log_probs':log_prob_array, 'target_values':value_target_array, 'data_length':num_finished_samples}
        return data

    def train_on_samples(self, data):
        """
        data (dict) : Has keys: 'advantages', 'old_actions', 'old_log_probs', 'target_values'
        """

        # approx kl is used for early stopping: Once difference between old and new policy get too large, break the training, start new iteration
        approx_kl = 0

        #Training Loop
        for epoch in range(EPOCHS):
            #data is split into minibatches
            mini_batch_list = self.generate_minibatches(data)
            for mini_batch in mini_batch_list:
                #Optimize network
                fetches = self.session.run(self.retrieve_list, feed_dict = mini_batch)
                #Approx KL is used for early stopping, get latest KL
                approx_kl = fetches[-1]
                #Get entries for logging, which can be used for visualization and/or debugging
                logging_fetches = fetches[0:len(self.logger_keys)]
                self.update_minibatch_logger(logging_fetches)
                #Early stopping if KL is too big
                print(approx_kl)
                if approx_kl > KL_EARLY_STOPPING:
                    break
            #Update Logger
            self.update_epoch_logger()
            #Early stopping
            if approx_kl > KL_EARLY_STOPPING:
                break
        #Update Logger
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
        #Does not have data_length , as this has to be handled alone
        data_key_list = ['advantages','old_actions','old_log_probs','target_values']
        #List of placeholders, that need feeding
        placeholder_list = [self.advantage_placeholder, self.old_action_placeholder, self.old_log_prob_placeholder, self.value_target_placeholder]

        #Shuffle data
        random_permutation = np.random.permutation(range(data['data_length']))
        for key in data_key_list:
            data[key] = data[key][random_permutation]

        #Slice data into minibatches
        mini_batch_list = []
        indices = range(0,data['data_length'], OPTIMIZATION_BATCH_SIZE)
        for start_index in indices:
            end_index = start_index + OPTIMIZATION_BATCH_SIZE
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
        #Sample actions
        actions = self.action_distribution.sample()
        #Get (logarithmic) probabilites of actions
        log_probs = self.action_distribution.log_prob(actions)
        #Sum over individual action components (--> Multiplication in log space is addition!)
        log_probs = tf.reduce_sum(log_probs, -1)
        return actions, log_probs

    def value_loss(self):
        #Compute Mean square error for critic training
        value_f_loss = tf.square(self.value_prediction - self.value_target_placeholder)
        loss = tf.reduce_mean(value_f_loss)
        return loss


    def clipped_ppo_objective(self):
        #Clipped objective function from PPO Paper
        #Probability ratio of old and new parameters
        prob_ratio = tf.exp(tf.reduce_sum(self.action_distribution.log_prob(self.old_action_placeholder), axis=-1)-self.old_log_prob_placeholder)
        clipped_prop_ratio = tf.clip_by_value(prob_ratio, 1-CLIPPINGPARAMETER, 1+CLIPPINGPARAMETER)
        clipped_objective = tf.minimum(prob_ratio*self.advantage_placeholder, clipped_prop_ratio*self.advantage_placeholder)
        loss = -tf.reduce_mean(clipped_objective)
        return loss

    def approx_entropy_loss(self):
        entropy = tf.reduce_mean(self.action_distribution.entropy())
        #entropy = - tf.reduce_mean(self.action_distribution.log_prob(self.old_action_placeholder))
        entropy_loss = -entropy
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

    def compute_advantages(self, value_estimations, rewards, dones):
        value_estimations = np.squeeze(value_estimations)
        value_state_t =value_estimations
        value_state_t_plus_one = np.append(value_estimations, np.zeros((1,CREATION_BATCH_SIZE)), axis=0)[1:]
        value_state_t_plus_one[dones] = 0
        advantages = rewards + GAMMA*value_state_t_plus_one - value_state_t
        #Normalize advantages
        #advantages = advantages - np.mean(advantages)
        advantages = advantages/np.std(advantages)
        return advantages

#TODO Generalized advantage Estimation

    def compute_generalized_advantage_estimations(self, value_estimations, rewards, dones):
        value_estimations = np.squeeze(value_estimations)
        #print(value_estimations.shape)
        assert rewards.shape == value_estimations.shape
        assert rewards.shape == dones.shape
        value_estimations = np.squeeze(value_estimations)
        value_state_t =value_estimations
        value_state_t_plus_one = np.append(value_estimations, np.zeros((1,CREATION_BATCH_SIZE)), axis=0)[1:]
        value_state_t_plus_one[dones] = 0
        deltas = rewards + GAMMA*value_state_t_plus_one - value_state_t

        advantages = np.zeros((CREATION_STEPS, CREATION_BATCH_SIZE))
        gamma_lambda = GAMMA*GAE_LAMBDA
        accumulated_advantage = np.zeros((1,CREATION_BATCH_SIZE))
        for j in range(CREATION_STEPS):
            i = (j+1)*-1
            accumulated_advantage = gamma_lambda * accumulated_advantage
            accumulated_advantage[np.expand_dims(dones[i,:], axis=0)] = 0
            accumulated_advantage = accumulated_advantage + deltas[i,:]
            advantages[i,:] = accumulated_advantage
        #Normalize advantages
        #advantages = advantages - np.mean(advantages)
        advantages = advantages/np.std(advantages)
        return advantages


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

    def evaluate(self):
        observation = self.env_core.reset()
        action_fetch, log_prob_fetch = self.sample_action()
        fetch_list = [action_fetch, log_prob_fetch, self.value_prediction]
        reward_list = []
        sigma_list = []
        for step in range(CREATION_STEPS):
            feed_dict = self.feed_dictionary(observation)
            action, log_prob, value_estimate = self.session.run(fetch_list, feed_dict=feed_dict)
            sigma = self.session.run(self.actor.output_list[-1], feed_dict = feed_dict)
            sigma_list.append(sigma)
            self.env_core.step_async(action)
            observation, reward, done, info = self.env_core.step_wait()
            reward_list.append(reward)
        reward_array = np.stack(reward_list)
        return [np.mean(reward_array), np.mean(np.stack(sigma_list))]

def main():
    network_description = [{'size':256, 'activation':tf.nn.relu}, {'size':256, 'activation':tf.nn.relu},{'size':256, 'activation':tf.nn.relu}]
    trainer = PPO_model(network_description, network_description, 'LunarLanderContinuous-v2')
    evaluations = []
    for i in range(ITERATIONS):
        trainer.train()
        if i%5==0:
            evaluations.append(trainer.evaluate())
            print('Evaluation: ')
            print(evaluations)
        #print('results: ')
        #print(trainer.epoch_logger)
        #print(trainer.reward_logger)
        #print('logger: ')
        #print(trainer.minibatch_logger[-2])
if __name__ == '__main__':
    main()
