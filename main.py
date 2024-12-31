import numpy as np


class GridEnvironment:
    def __init__(self, size, goal_position):
        self.size = size
        self.goal_position = goal_position
        self.agent_position = [0, 0]  # Agent starts at top-left corner

    def reset(self):
        self.agent_position = [0, 0]
        return np.array(self.agent_position)

    def step(self, action):
        if action == 0 and self.agent_position[0] > 0:  # UP
            self.agent_position[0] -= 1
        elif action == 1 and self.agent_position[0] < self.size - 1:  # DOWN
            self.agent_position[0] += 1
        elif action == 2 and self.agent_position[1] > 0:  # LEFT
            self.agent_position[1] -= 1
        elif action == 3 and self.agent_position[1] < self.size - 1:  # RIGHT
            self.agent_position[1] += 1

    def is_done(self):
        return self.agent_position == self.goal_position

    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                if [i, j] == self.agent_position:
                    print("A", end=" ")  # Agent's position
                elif [i, j] == self.goal_position:
                    print("G", end=" ")  # Goal position
                else:
                    print(".", end=" ")
            print()


class DeterministicAgent:
    def __init__(self, env):
        self.env = env

    def action(self):
        if self.env.agent_position[0] < self.env.goal_position[0]:
            return 1
        elif self.env.agent_position[0] > self.env.goal_position[0]:
            return 0
        elif self.env.agent_position[1] < self.env.goal_position[1]:
            return 3
        elif self.env.agent_position[1] > self.env.goal_position[1]:
            return 2


def execute(agent):
    num_of_tries = 0

    while not agent.env.is_done() and num_of_tries < 100:
        action = agent.action()
        agent.env.step(action)
        agent.env.render()
        num_of_tries += 1


def main():
    env = GridEnvironment(size=4, goal_position=[3, 3])
    agent = DeterministicAgent(env)
    execute(agent)

    if agent.env.is_done():
        print("done!!")
        env.render()
    else:
        print("failed!!")


if __name__ == "__main__":
    main()
