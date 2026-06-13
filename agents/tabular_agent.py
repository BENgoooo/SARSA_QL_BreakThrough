from collections import defaultdict
from typing import Any, Hashable, Sequence
import numpy as np

class TabularAgent:

    def __init__(
        self,
        actions: Sequence[Any],
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:

        self.actions = list(actions)
        self.n_actions =  len(self.actions)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.rng = np.random.default_rng(seed)

        """
            Q 表本质上是一个字典，用来存储Q(s, a), 即状态 s 下，执行动作 a 的价值。
        {
            state_1: [Q(state_1, action_0), Q(state_1, action_1), Q(state_1, action_2), Q(state_1, action_3)],
            state_2: [Q(state_2, action_0), Q(state_2, action_1), Q(state_2, action_2), Q(state_2, action_3)],
            ...
        }
        """
        self.q_table: defaultdict[Hashable, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype = np.float64)
        ) # lambda是一个不用写函数名，直接写return内容的函数写法

    def state_to_key(self, state: Any) -> Hashable:
        """
        将环境返回的 state 转成 Q 表可以使用的 key。

        Q 表本质上是一个字典：
            Q[state][action_index]

        因此 state 必须是可哈希的。
        如果环境返回的是 list、dict、numpy array 等不可哈希对象，
        就需要先转换成 tuple。
        """

        if isinstance(state, np.ndarray):
            return self.state_to_key(state.tolist())

        if isinstance(state, list):
            return tuple(self.state_to_key(item) for item in state)

        if isinstance(state, tuple):
            return tuple(self.state_to_key(item) for item in state)

        if isinstance(state, dict):
            return tuple(
                sorted((key, self.state_to_key(value)) for key, value in state.items())
            )

        return state

    def action_to_index(self, action: Any) -> int:
        try:
            return self.actions.index(action)
        except ValueError as exc:
            raise ValueError(f"未知动作: {action}") from exc

    def select_action(self, state: Any) -> Any:
        """
        使用 epsilon-greedy 策略选择动作。

        逻辑如下：
        1. 以 epsilon 的概率随机选动作，用于探索；
        2. 以 1 - epsilon 的概率选择当前 Q 值最大的动作，用于利用已有经验。
        """
        state_key = self.state_to_key(state)

        if self.rng.random() < self.epsilon:
            action_index = self.rng.integers(self.n_actions)
            return self.actions[action_index]

        q_values = self.q_table[state_key]
        # 如果多个动作并列最大，随机选择其中一个，避免永远偏向第一个动作。
        max_q = np.max(q_values)
        best_action_indices = np.flatnonzero(q_values == max_q)
        action_index = self.rng.choice(best_action_indices)

        return self.actions[action_index]

    def get_q(self, state: Any, action: Any) -> float:
        """
        读取 Q(s, a)。
        """

        state_key = self.state_to_key(state)
        action_index = self.action_to_index(action)

        return float(self.q_table[state_key][action_index])    

    def set_q(self, state: Any, action: Any, value: float) -> None:
        """
        设置 Q(s, a)。
        """

        state_key = self.state_to_key(state)
        action_index = self.action_to_index(action)

        self.q_table[state_key][action_index] = value    

    def max_q(self, state: Any) -> float:
        """
        返回某个状态下最大的 Q 值：

            max_a Q(s, a)

        这个函数后面主要给 Q-learning 使用。
        """

        state_key = self.state_to_key(state)
        return float(np.max(self.q_table[state_key]))

    def best_action(self, state: Any) -> Any:
        """
        返回当前状态下 Q 值最大的动作。

        这个函数不做 epsilon 探索，只用于评估或展示当前策略。
        """

        state_key = self.state_to_key(state)
        q_values = self.q_table[state_key]

        max_q = np.max(q_values)
        best_action_indices = np.flatnonzero(q_values == max_q)
        action_index = self.rng.choice(best_action_indices)

        return self.actions[action_index]