import random
import numpy as np
from pydantic import BaseModel

from openai import OpenAI


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


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def action(self):
        return random.randint(0, 3)


class ActionResponse(BaseModel):
    action: int


class OpenAIAgent:
    def __init__(self, env):
        self.env = env
        self.client = OpenAI()

    def action(self):
        prompt = (
            "You are an agent and your goal is to reach the target position. "
            "The grid size is {self.env.size}x{self.env.size}. "
            "You will be given your current position and the target position. "
            "You need to choose the best action to move closer to the target. "
            "Actions are: 0 (UP), 1 (DOWN), 2 (LEFT), 3 (RIGHT)."
            "Choose action from 0 to 3. The output should be a single number."
        )

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"current position: {self.env.agent_position},"
                        f"target position: {self.env.goal_position}. "
                        "Give output in the range 0 to 3."
                    ),
                },
            ],
            response_format=ActionResponse,
        )

        response: ActionResponse | None = completion.choices[0].message.parsed
        if response:
            return response.action
        else:
            return 0


def execute(agent):
    num_of_tries = 0

    while not agent.env.is_done() and num_of_tries < 100:
        action = agent.action()
        agent.env.step(action)
        agent.env.render()
        num_of_tries += 1

    return num_of_tries


def main():
    env = GridEnvironment(size=4, goal_position=[3, 3])
    agent = DeterministicAgent(env)
    agent = RandomAgent(env)
    agent = OpenAIAgent(env)
    num_of_tries = execute(agent)

    if agent.env.is_done():
        print(f"done in {num_of_tries} steps")
        env.render()
    else:
        print("failed!!")


if __name__ == "__main__":
    main()
