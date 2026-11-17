from agent import Agent


class RewardManager:
    """No shaping, time penalties, or collision penalties."""

    @staticmethod
    def collect(agent: Agent, food: int) -> int:
        if food not in (-1, 1):
            raise ValueError("Only red (-1) and green (+1) food can be collected.")
        if food == 1:
            agent.green += 1
        else:
            agent.red += 1
        return int(food)
