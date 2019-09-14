import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
import time
import matplotlib.pyplot as plt
np.set_printoptions(threshold=999999)

#HYPERPARAMETERS
#Initializer for weights and biases
INITIALIZER = tf.glorot_uniform_initializer
#Clipping parameter from PPO
CLIPPINGPARAMETER = 0.2
#Learn rate for actor
ACTOR_LEARN_RATE = 0.0003
#Learn rate for critiv
CRITIC_LEARN_RATE = 0.001
#Batch size in optimization for actor and critic
OPTIMIZATION_BATCH_SIZE = 256
#Batch size and number of subprocesses for parallel computation of environments
CREATION_BATCH_SIZE = 8
#Number of steps taken total in creation of samples for one iteration
CREATION_EPISODES = 128
#Gamma value discount
GAMMA = 0.99
#Lambda discount for GAE
GAE_LAMBDA = 0.95
#Which advantage function to use (Advantage or GAE)
ADVANTAGE_FUNCTION = 'Generalized_Advantages'
#Activation function for actions
ACTION_ACTIVATION = 'softmax'
#How many iterations to run this
ITERATIONS = 1000
#Number of epochs in optimization
EPOCHS = 5
#Multiplicative factor for entropy loss
ENTROPY_LOSS_FACTOR = 0.01
#Early stopping stops when approx KL is larget than this
KL_EARLY_STOPPING = .04
#How many iterations the early stopping is not used in the beginning
BURN_IN_ITERATIONS = 5
#How many steps to use in evaluation (also steps that might be visualized)
EVALUATION_STEPS = 2050

OUTPUT_ACTIVATIONFUNCTIONS = {'tanh' : tf.nn.tanh, 'softmax': tf.nn.softmax}

#Whether to visualize during evaluation
VISUALIZE = False

#Network is implementing a feed forward network with possibly several outputs
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
            layer_name = self.name+'_layer_'+str(i)
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

        linear = tf.matmul(input, weights)
        drive = linear + bias

        if activation == None:
            return linear
        elif activation in OUTPUT_ACTIVATIONFUNCTIONS:
            activation_function = OUTPUT_ACTIVATIONFUNCTIONS[activation]
            return activation_function(linear)
        else:
            return activation(drive)


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
        #Keeping track of iterations run with this model
        self.iteration = 0
        #String used by gym to create environment
        self.env_name = env_name

        #core is a parallelized model of the environments, environments is inside a normalizing wrapper, that uses an updating average
        self.env_core = SubprocVecEnv([self.env_function for env in range(CREATION_BATCH_SIZE)])

        #self.normalized_environments = VecNormalize(self.env_core, cliprew = 100.)
        #self.normalized_environments.reset()
        #If you change this back, then oyu also got to change line 218 or so
        self.normalized_environments = self.env_core

        #Input size and output size of network
        self.input_size = len(self.env_function().observation_space.high)
        self.output_size = self.env_function().action_space.n

        #The tensorflow model
        self.session = tf.Session()
        self.actor = Network(self.input_size, actor_description, [{'size': self.output_size,'activation':ACTION_ACTIVATION}], 'actor')
        self.critic = Network(self.input_size, critic_description, [{'size':1, 'activation':None}], 'critic')

        #The list of variables, which should be saved
        self.save_list = self.actor.save_list + self.critic.save_list

        # THis has to be normal for the entropy to be correctly computed! If you change this, also change entropy computation
        self.action_distribution = tfp.distributions.Categorical(probs=self.actor.output_list[0])
        self.value_prediction = self.critic.output_list[0]

        #placeholders for training
        self.advantage_placeholder = tf.placeholder(tf.float32, shape=(None), name='advantage_placeholder')
        self.old_action_placeholder = tf.placeholder(tf.float32, shape=(None), name='old_action_placeholder')
        self.old_log_prob_placeholder = tf.placeholder(tf.float32, shape=(None), name='old_log_prob_placeholder')
        self.value_target_placeholder = tf.placeholder(tf.float32, shape=(None,1), name='value_target_placeholder')



        self.logger_dict = {'value_loss' : self.value_loss(), 'objective_function':self.clipped_ppo_objective(), 'entropy' : self.approx_entropy_loss()}
        self.logger_keys = list(self.logger_dict.keys())
        self.logger_fetches = [self.logger_dict[key] for key in self.logger_keys]

        #Loggers keep track of progress on the respective level of description
        self.iteration_logger = []
        self.epoch_logger = [[]]
        self.minibatch_logger = [[[]]]

        #Reward Logger to track training progress
        self.reward_logger = []

        self.actor_optimizer = tf.train.AdamOptimizer(ACTOR_LEARN_RATE)
        self.critic_optimizer = tf.train.AdamOptimizer(CRITIC_LEARN_RATE)
        self.retrieve_list = self.logger_fetches + self.train_step() + self.approximate_kl_divergence()

        #Saving and restoring is not implemented yet.
        if save_file==None:
            self.session.run(tf.global_variables_initializer())
        else:
            self.restore(save_file)


    def train(self):
        data = self.create_data()
        self.train_on_samples(data)


    def create_data(self):

        observation = self.normalized_environments.reset()
        action_fetch, log_prob_fetch = self.sample_action()

        fetch_list = [action_fetch, log_prob_fetch, self.value_prediction]

        observation_list = []
        action_list = []
        log_prob_list = []
        value_estimate_list = []
        reward_list = []
        done_list = []
        #Generate at least CREATION_EPISODES
        created_episodes = 0
        #for step in range(CREATION_STEPS):
        while created_episodes < CREATION_EPISODES:
            observation_list.append(observation)
            feed_dict = self.feed_dictionary(observation)
            action, log_prob, value_estimate = self.session.run(fetch_list, feed_dict=feed_dict)
            observation, reward, done, info = self.env_step(action)
            #check how many episodes already have been generated
            created_episodes = created_episodes + np.sum(done)
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
        print('Landed in this iteration' + str(np.sum(reward_array>80)))
        print('Landed values' + str(reward_array[reward_array>80]))
        #print('actual done values' + str(reward_array[done_array]))
        print('samples with finishing reward ' + str(np.argwhere(reward_array>80)))
        #print('samples actually finished here: ' + str(np.argwhere(done_array)))
        #self.reward_logger.append(np.mean(reward_array) * np.sqrt(self.normalized_environments.ret_rms.var + self.normalized_environments.epsilon))
        self.reward_logger.append(np.mean(reward_array))
        #normalize reward array
        reward_array = reward_array - np.mean(reward_array)
        reward_array = reward_array/np.std(reward_array)

        value_target_array = self.compute_values(reward_array, done_array)
        advantage_array = None
        if ADVANTAGE_FUNCTION == 'Advantages':
            advantage_array = self.compute_advantages(value_estimate_array, reward_array, done_array)
        elif ADVANTAGE_FUNCTION == 'Generalized_Advantages':
            advantage_array = self.compute_generalized_advantage_estimations(value_estimate_array, reward_array, done_array)

        #Write down how good we are now on avaerage
        #But since VecNormalize is used, this has to be scaled by the normalizing factor, the normalizationfactor is taken from
        #From here on only the finished samples will be used, that is the samples from runs, where a final state has been reached
        finished_samples = self.compute_finished_samples(done_array)
        num_finished_samples = np.sum(finished_samples)

        #Total number of samples
        sample_number = reward_array.size

        print(str(num_finished_samples) + ' out of ' + str(sample_number) + 'usable')
        print(str(num_finished_samples/np.sum(done_array)) + ' steps per run')

        observation_array = np.reshape(observation_array, (sample_number, self.input_size))
        action_array = np.reshape(action_array, (sample_number))
        log_prob_array = np.reshape(log_prob_array, (sample_number))
        value_estimate_array = np.reshape(value_estimate_array, (sample_number))
        value_target_array = np.reshape(value_target_array, (sample_number))
        #Debug only for done and reward array
        reward_array = np.reshape(reward_array, (sample_number))
        done_array = np.reshape(done_array, (sample_number))
        advantage_array = np.reshape(advantage_array, (sample_number))
        finished_samples = np.reshape(finished_samples, (sample_number))

        #Only use finished samples (That is samples where the run has been finished)
        observation_array = observation_array[finished_samples]
        action_array = action_array[finished_samples]
        log_prob_array = log_prob_array[finished_samples]
        value_estimate_array = value_estimate_array[finished_samples]
        value_target_array = value_target_array[finished_samples]
        #Done and reward array are only needed here for debug purposes
        done_array = done_array[finished_samples]
        reward_array = reward_array[finished_samples]
        #Output of network has size [Batchsize,1], so this has to be reshaped respectively
        value_target_array = np.expand_dims(value_target_array,-1)
        advantage_array = advantage_array[finished_samples]
        print('advantages and rewards for final moves')
        print(advantage_array.flatten()[done_array.flatten()])
        print(reward_array.flatten()[done_array.flatten()])

        #This is debug utility
        #print('Targets: Mean: ' + str(np.mean(value_target_array)) + ' Std: ' + str(np.std(value_target_array)))
        #print('Estimates: Mean: ' + str(np.mean(value_estimate_array)) + ' Std: ' + str(np.std(value_estimate_array)))

        #Prepare data in a ditionary for ease of use
        data = {'observations':observation_array, 'advantages':advantage_array, 'old_actions':action_array, 'old_log_probs':log_prob_array, 'target_values':value_target_array, 'data_length':num_finished_samples}

        #update iteration
        self.iteration = self.iteration + 1

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
                """
                print('val pred shape')
                print(self.session.run(self.value_prediction, feed_dict = mini_batch).shape)
                print('val est shape')
                print(self.session.run(self.value_target_placeholder, feed_dict = mini_batch).shape)
                """
                fetches = self.session.run(self.retrieve_list, feed_dict = mini_batch)
                #Debug
                """
                print('debug_val_loss')
                print(self.session.run(self.debug_val_loss(), feed_dict=mini_batch).shape)
                print('pred_shape')
                print(self.session.run(self.value_prediction, feed_dict=mini_batch).shape)
                print('target_shape')
                print(self.session.run(self.value_target_placeholder, feed_dict=mini_batch).shape)
                """
                #Approx KL is used for early stopping, get latest KL
                approx_kl = fetches[-1]
                #Get entries for logging, which can be used for visualization and/or debugging
                logging_fetches = fetches[0:len(self.logger_keys)]
                self.update_minibatch_logger(logging_fetches)
                #print(approx_kl)
                #Early stopping if KL is too big
                if (approx_kl > KL_EARLY_STOPPING) and self.iteration > BURN_IN_ITERATIONS :
                    print('break cause KL')
                    break
            #Update Logger
            self.update_epoch_logger()
            #Early stopping
            if (approx_kl > KL_EARLY_STOPPING) and self.iteration > BURN_IN_ITERATIONS:
                print('break cause KL')
                break
        #Update Logger
        self.update_iteration_logger()


    def generate_minibatches(self, data):
        """
        Args:
            data (dict) : Has keys: 'observations', 'advantages', 'old_actions', 'old_log_probs', 'target_values', 'data_length', each containing:
                data['observations'] (ndarray): shape(data_length, observation_size)
                data['advantages'] (ndarray): shape(data_length)
                data['old_actions'] (ndarray): shape(data_length)
                data['old_log_probs'] (ndarray): shape(data_length)
                data['target_values'] (ndarray): shape(data_length)
                data['data_length'] (int) : len of total data sequence

        Returns:
            minibatch_list with feed_dicts, each of length of OPTIMIZATION_BATCH_SIZE
        """
        #Does not have data_length , as this has to be handled alone
        data_key_list = ['advantages','old_actions','old_log_probs','target_values', 'observations']

        #List of placeholders, that need feeding
        placeholder_list = [self.advantage_placeholder, self.old_action_placeholder, self.old_log_prob_placeholder, self.value_target_placeholder]

        #Shuffle data
        random_permutation = np.random.permutation(range(data['data_length']))
        for key in data_key_list:
            #print('data_shape ' + key)
            #print(data[key].shape)
            data[key] = data[key][random_permutation]

        #Slice data into minibatches
        mini_batch_list = []
        indices = range(0,data['data_length']-OPTIMIZATION_BATCH_SIZE, OPTIMIZATION_BATCH_SIZE)
        for start_index in indices:
            end_index = start_index + OPTIMIZATION_BATCH_SIZE
            minibatch_dict = self.feed_dictionary(data['observations'][start_index:end_index])
            for data_key, feed_key in zip(data_key_list, placeholder_list):
                minibatch_dict[feed_key] = data[data_key][start_index:end_index]
            mini_batch_list.append(minibatch_dict)
        #plt.hist(np.exp(data['old_log_probs']),40)
        #plt.show()
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
        return actions, log_probs

    def value_loss(self):
        #Compute Mean square error for critic training
        value_f_loss = tf.square(self.value_prediction - self.value_target_placeholder)
        loss = tf.reduce_mean(value_f_loss)
        return loss

    def clipped_ppo_objective(self):
        #Clipped objective function from PPO Paper
        #Probability ratio of old and new parameters
        prob_ratio = tf.exp(self.action_distribution.log_prob(self.old_action_placeholder)-self.old_log_prob_placeholder)
        #Compute PPO surrogate objective function
        clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1-CLIPPINGPARAMETER, 1+CLIPPINGPARAMETER)
        clipped_objective = tf.minimum(prob_ratio*self.advantage_placeholder, clipped_prob_ratio*self.advantage_placeholder)
        loss = -tf.reduce_mean(clipped_objective)
        return loss

    def approx_entropy_loss(self):
        #This is motivated by OpenAI baselines
        entropy = tf.reduce_mean(self.action_distribution.entropy())
        #entropy = - tf.reduce_mean(self.action_distribution.log_prob(self.old_action_placeholder))
        entropy_loss = - entropy
        return entropy_loss

    def policy_loss(self):
        #Policy loss is sum of surrogate objective and entropy loss, is used to optimize the actor
        return self.clipped_ppo_objective() + ENTROPY_LOSS_FACTOR*self.approx_entropy_loss()

    def approximate_kl_divergence(self):
        #Computes the approximate kl div between old and new log probs for early stopping
        return [-tf.reduce_mean(self.action_distribution.log_prob(self.old_action_placeholder) - self.old_log_prob_placeholder)]

    def feed_dictionary(self, observation, old_action=None, old_log_prob=None, value_target=None):
        #Creates feed dictionaries for training
        feed_dict = {self.actor.input_placeholder : observation, self.critic.input_placeholder : observation}
        if old_action!=None:
            feed_dict[self.old_action_placeholder]=old_action
        if old_log_prob!=None:
            feed_dict[self.old_log_prob_placeholder]=old_log_prob
        if value_target!=None:
            feed_dict[self.value_target_placeholder]=value_target
        return feed_dict

    def env_step(self, action):
        #Compute steps in parallelized Environments
        self.normalized_environments.step_async(action)
        return self.normalized_environments.step_wait()

    def train_step(self):
        # Train step combines both optimizations in one list
        return [self.actor_optimizer.minimize(self.policy_loss()), self.critic_optimizer.minimize(self.value_loss())]


    def update_minibatch_logger(self, log_entries):
        #Updates the minibatch_logger
        log_dict = {}
        for entry, key in zip(log_entries, self.logger_keys):
            log_dict[key] = entry
        self.minibatch_logger[-1][-1].append(log_dict)

    def update_epoch_logger(self):
        #Updates the epoch_logger
        last_entries = self.minibatch_logger[-1][-1]
        log_dict = {}
        for key in self.logger_keys:
            log_dict[key] = np.mean([dict[key] for dict in last_entries])
        self.epoch_logger[-1].append(log_dict)
        self.minibatch_logger[-1].append([])

    def update_iteration_logger(self):
        #Updates iteration_logger
        last_entries = self.epoch_logger[-1]
        log_dict = {}
        for key in self.logger_keys:
            log_dict[key] = np.mean([dict[key] for dict in last_entries])
        self.iteration_logger.append(log_dict)
        self.epoch_logger.append([])
        self.minibatch_logger.append([[]])

    def env_function(self):
        #Returns a function that returns a new gym, is needed for the parallelized Environments
        return gym.make(self.env_name)

    def compute_values(self, rewards, dones):
        """
        rewards (ndarray): rewards of shape (CREATION_STEPS, CREATION_BATCH_SIZE)
        dones (reward_array): dones of shape (CREATION_STEPS, CREATION_BATCH_SIZE)
        """
        #Computes Values (discounted sums of rewards)
        #Make sure these have right shapes
        assert rewards.shape == dones.shape
        assert rewards.shape[0] == dones.shape[0]

        #Compute Values from end to front
        rewards = np.flip(rewards,axis=0)
        dones = np.flip(dones, axis=0)
        values = np.zeros(rewards.shape)

        accumulated_reward = np.zeros((CREATION_BATCH_SIZE))

        creation_steps = dones.shape[0]

        for i in range(creation_steps):
            accumulated_reward[dones[i,:]] = 0
            accumulated_reward = accumulated_reward*GAMMA
            values[i,:] = rewards[i,:] + accumulated_reward
            accumulated_reward = values[i,:]

        #Flip back from end-->front to front-->end direction
        values = np.flip(values,axis=0)
        return values

    #For both Advantage Calculations i could implement the boostrapping mechanism,
    #which can also seen in several OpenAI implementations - This would make the implementation mor eunstable though

    def compute_advantages(self, value_estimations, rewards, dones):
        value_estimations = np.squeeze(value_estimations)
        value_state_t =value_estimations
        value_state_t_plus_one = np.append(value_estimations, np.zeros((1,CREATION_BATCH_SIZE)), axis=0)[1:]
        value_state_t_plus_one[dones] = 0
        advantages = rewards + GAMMA*value_state_t_plus_one - value_state_t
        #Normalize advantage         #advantages = advantages - np.mean(advantages)

        #advantages = advantages - np.mean(advantages)
        advantages = advantages/np.std(advantages)
        return advantages


    def compute_generalized_advantage_estimations(self, value_estimations, rewards, dones):
        #Computes GAE (generalized advantage estimations (from the Schulman paper))

        value_estimations = np.squeeze(value_estimations)
        assert rewards.shape == value_estimations.shape
        assert rewards.shape == dones.shape
        value_state_t = value_estimations
        value_state_t_plus_one = np.append(value_estimations, np.zeros((1,CREATION_BATCH_SIZE)), axis=0)[1:,:]
        value_state_t_plus_one[dones] = 0
        deltas = rewards + GAMMA*value_state_t_plus_one - value_state_t

        advantages = np.zeros(value_estimations.shape)
        gamma_lambda = GAMMA*GAE_LAMBDA
        accumulated_advantage = np.zeros((1,CREATION_BATCH_SIZE))
        creation_steps = rewards.shape[0]

        for j in range(creation_steps):
            i = (j+1)*-1
            accumulated_advantage = gamma_lambda * accumulated_advantage
            accumulated_advantage[np.expand_dims(dones[i,:], axis=0)] = 0
            accumulated_advantage = accumulated_advantage + deltas[i,:]
            advantages[i,:] = accumulated_advantage
        #Normalize advantages
        advantages = advantages - np.mean(advantages)
        advantages = advantages/np.std(advantages)
        return advantages

    #Still has to be implemented, is in TODO
    def save(self, file_name):
        stub = None
    #Still has to be implemented, is in TODO
    def restore(self, file_name):
        stub = None

    def compute_finished_samples(self, dones):
    #Computes a Boolean map of which samples are finished ()

        finished_samples = np.ones(dones.shape)==1
        run_done = dones[-1,:]
        creation_steps = dones.shape[0]

        for i in range(creation_steps):
            if np.all(run_done):
                break
            run_done = run_done + dones[-(i+1),:]
            finished_samples[-(i+1)] = run_done
        return finished_samples

    def evaluate(self):
        #Run an evaluation run
        observation = self.env_core.reset()
        action_fetch, log_prob_fetch = self.sample_action()
        fetch_list = [action_fetch, log_prob_fetch, self.value_prediction]
        reward_list = []
        sigma_list = []

        for step in range(EVALUATION_STEPS):
            feed_dict = self.feed_dictionary(observation)
            action, log_prob, value_estimate, finishing_probs = self.session.run(fetch_list + [self.actor.output_list[0]], feed_dict=feed_dict)
            sigma = self.session.run(self.actor.output_list[-1], feed_dict = feed_dict)
            sigma_list.append(sigma)
            if VISUALIZE:
                self.normalized_environments.get_images()
            self.normalized_environments.step_async(action)
            observation, reward, done, info = self.normalized_environments.step_wait()
            reward_list.append(reward)
            if np.sum(done) > 0:
                print('finishing probs ' + str(finishing_probs[done]))
                print('finising rewards ' + str(reward[done]))
        reward_array = np.stack(reward_list)
        #sigma is interesting as it shows the exploration rate of the algorithm
        return [np.mean(reward_array), np.mean(np.stack(sigma_list),axis=(0,1))]

def main():
    network_description = [{'size':64, 'activation':tf.nn.tanh}, {'size':64, 'activation':tf.nn.tanh}]
    trainer = PPO_model(network_description, network_description, 'LunarLander-v2')
    evaluations = []
    for i in range(ITERATIONS):
        trainer.train()
        print('Training reward average')
        print(trainer.reward_logger[-1])

        #This is interesting because i can get the mean
        if i%3==0:
            evaluations.append(trainer.evaluate())
            print('Evaluation (Reward / mean std): ')
            print(evaluations)

        #Every eight iterations (CREATION_EPISODES*8 full iterations through the environment) print the progress
        if i%20==0:
            plt.subplot(2,2,1)
            plt.title('Rewards')
            plt.plot(trainer.reward_logger)

            plt.subplot(2,2,2)
            plt.title('Value_loss')
            plt.plot([dict['value_loss'] for dict in trainer.iteration_logger[0:-2]])

            plt.subplot(2,2,3)
            plt.title('PPO objective')
            plt.plot([dict['objective_function'] for dict in trainer.iteration_logger[0:-2]])

            plt.subplot(2,2,4)
            plt.title('Entropy')
            plt.plot([dict['entropy'] for dict in trainer.iteration_logger[0:-2]])

            plt.show()

if __name__ == '__main__':
    main()
