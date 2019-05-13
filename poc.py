import ppo
import tensorflow as tf
iterations = 500

def main():
    network_description = [{'size':128, 'activation':tf.nn.relu}, {'size':256, 'activation':tf.nn.relu},{'size':256, 'activation':tf.nn.relu}]
    trainer = ppo.PPO_model(network_description, network_description, 'LunarLanderContinuous-v2')

    for i in range(iterations):
        trainer.train_step()
        print(trainer.reward_logger)

if __name__ == '__main__':
    main()
