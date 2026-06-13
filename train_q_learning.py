from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from envs.breakthrough_env import BreakthroughEnv, ACTION
from agents.tabular_agent import TabularAgent


@dataclass
class EpisodeResult:
    episode: int
    total_reward: float
    steps: int
    event: str
    epsilon: float

def q_learning_update(
    agent: TabularAgent,
    state: Any,
    action: Any,
    reward: float,
    next_state: Any,
    done: bool,
) -> None:

    old_q = agent.get_q(state, action)
    
    if done:
        target = reward
    else:
        target = reward + agent.gamma * agent.max_q(next_state)

    new_q = old_q + agent.alpha * (target - old_q)

    agent.set_q(state, action, new_q)



def train_one_episode(
    env: BreakthroughEnv,
    agent: TabularAgent,
    episode: int,
) -> EpisodeResult:
    state = env.reset()

    done = False
    total_reward = 0.0
    steps = 0
    info: dict[str, Any] = {"event": "unknown"}

    while not done:
        action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)
        q_learning_update(
            agent=agent,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        state = next_state
        total_reward += reward
        steps += 1

    return EpisodeResult(
        episode=episode,
        total_reward=total_reward,
        steps=steps,
        event=info.get("event", "unknown"),
        epsilon=agent.epsilon,
    )


def train(
    env: BreakthroughEnv,
    agent: TabularAgent,
    num_episodes: int = 2000,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.02,
    log_interval: int = 100,
) -> list[EpisodeResult]:

    history: list[EpisodeResult] = []

    for episode in range(1, num_episodes + 1):
        result = train_one_episode(
            env = env,
            agent = agent,
            episode = episode,
        )

        history.append(result)
        agent.epsilon = max(
            epsilon_min,
            agent.epsilon * epsilon_decay,
        )

        if episode % log_interval == 0:
            recent_results = history[-log_interval:]

            avg_reward = sum(
                item.total_reward for item in recent_results
            ) / len(recent_results)

            success_rate = sum(
                item.event == "success" for item in recent_results
            ) / len(recent_results)

            caught_rate = sum(
                item.event == "caught" for item in recent_results
            ) / len(recent_results)

            timeout_rate = sum(
                item.event == "timeout" for item in recent_results
            ) / len(recent_results)

            print(
                f"Episode {episode:5d} | "
                f"avg_reward={avg_reward:8.2f} | "
                f"success={success_rate:6.1%} | "
                f"caught={caught_rate:6.1%} | "
                f"timeout={timeout_rate:6.1%} | "
                f"epsilon={agent.epsilon:.3f} | "
                f"q_states={len(agent.q_table)}"
            )

    return history

def evaluate(
    env: BreakthroughEnv,
    agent: TabularAgent,
    max_render_steps: int = 30,
) -> None:

    state = env.reset()

    done = False
    total_reward = 0.0
    steps = 0
    info: dict[str, Any] = {"event": "unknown"}

    print("\n=== Greedy Policy Evaluation ===")
    print(env.render_text())

    while not done and steps < max_render_steps:
        action = agent.best_action(state)

        next_state, reward, done, info = env.step(action)

        state = next_state
        total_reward += reward
        steps += 1

        print(
            f"\nstep={steps}, "
            f"action={action}, "
            f"reward={reward}, "
            f"event={info.get('event')}"
        )
        print(env.render_text())

    print(
        f"\nEvaluation result: "
        f"total_reward={total_reward}, "
        f"steps={steps}, "
        f"event={info.get('event')}"
    )





def main() -> None:
    env = BreakthroughEnv(max_steps=100)

    agent = TabularAgent(
        actions=list(ACTION.keys()),
        alpha=0.1,
        gamma=0.95,
        epsilon=0.3,
        seed=42,
    )

    train(
        env=env,
        agent=agent,
        num_episodes=3000,
        epsilon_decay=0.995,
        epsilon_min=0.02,
        log_interval=100,
    )

    evaluate(
        env=env,
        agent=agent,
        max_render_steps=30,
    )


if __name__ == "__main__":
    main()
