from agents.tabular_agent import TabularAgent


def main():
    actions = [0, 1, 2, 3]

    agent = TabularAgent(
        actions=actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.2,
        seed=42,
    )

    state = (
        (0, 0),          # hero 位置
        ((2, 2, 1),),    # guard 信息，这里只是测试用的示例状态
    )

    print("初始 Q 表：")
    print(agent.q_table)

    print("\n测试状态：")
    print(state)

    print("\n读取初始 Q(s, a)：")
    for action in actions:
        print(f"Q(state, {action}) = {agent.get_q(state, action)}")

    print("\n设置 Q(state, 2) = 1.5")
    agent.set_q(state, 2, 1.5)

    print("\n更新后的 Q 值：")
    for action in actions:
        print(f"Q(state, {action}) = {agent.get_q(state, action)}")
    
    print("\n更新后的 Q 表：")
    print(agent.q_table)

    print("\n当前状态下的最大 Q 值：")
    print(agent.max_q(state))

    print("\n当前状态下的最优动作：")
    print(agent.best_action(state))

    print("\nepsilon-greedy 连续选 10 次动作：")
    for i in range(10):
        action = agent.select_action(state)
        print(f"第 {i + 1} 次选择动作: {action}")


if __name__ == "__main__":
    main()