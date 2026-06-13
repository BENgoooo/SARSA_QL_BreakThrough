from envs.breakthrough_env import BreakthroughEnv, ACTION_NAME

def main() -> None:
    env = BreakthroughEnv()
    state = env.reset()

    print("Initial environment:")
    print(env.render_text())
    print()
    env.print_state_info()
    print("-" * 50)

    actions = [
        3,  # RIGHT
        3,  # RIGHT
        0,  # UP
        0,  # UP
        3,  # RIGHT
        4,  # STAY
        0,  # UP
    ]

    for step_idx, action in enumerate(actions, start=1):
        next_state, reward, done, info = env.step(action)

        print(f"Step{step_idx}")
        print(f"Action:{ACTION_NAME[action]}")
        print(f"Reward:{reward}")
        print(f"Done:{done}")
        print(f"Info: {info}")
        print(f"Next state: {next_state}")
        print(env.render_text())
        print("-" * 50)

        if done:
            break

if __name__ == "__main__":
    main()

