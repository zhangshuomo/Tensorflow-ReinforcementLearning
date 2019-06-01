import gym
from Policy_Gradient import PolicyGradient


env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    discount_factor=0.99,
    # output_graph=True,
)

for i_episode in range(3000):

    observation = env.reset()

    while True:
        # env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.add_to_memory(observation, action, reward)

        if done:
            print("episode:", i_episode, "  reward:", sum(RL.returns))

            vt = RL.learn()
            break

        observation = observation_
