from RL_brain import QLearning, Sarsa, SarsaLambda
from maze import Maze

episodes = 100
maze = Maze()
agent = SarsaLambda([0, 1, 2, 3],learning_rate=0.05)
for e in range(episodes):
    maze.title('maze-episode:%d' % (e + 1))
    state = maze.reset()
    action = agent.chooseAction(state)
    in_episode = True
    # print(state)
    while in_episode:
        next_state, reward = maze.step(action)
        next_action = agent.chooseAction(next_state)
        if next_state == 'terminition':
            in_episode = False
        agent.learn(state, action, reward, next_state,next_action)
        state = next_state
        action = next_action
maze.destroy()
