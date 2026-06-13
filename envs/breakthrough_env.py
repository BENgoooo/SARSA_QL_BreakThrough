import random
from dataclasses import dataclass, replace
from typing import Optional

Position = tuple[int, int]
State = tuple[int, ...]

ACTION: dict[int, Position] = { 
    0: (-1, 0), # 向上
    1: (1, 0), # 向下
    2: (0, -1), # 向左
    3: (0, 1), # 向右
    4: (0, 0), # 不移动
}

ACTION_NAME: dict[int, str] = {
    0: "向上",
    1: "向下",
    2: "向左",
    3: "向右",
    4: "不移动",
}

REVERSE_ACTION: dict[int, int] = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
    4: 4,
}

@dataclass
class Guard:
    row: int
    col: int
    direction: int
    patrol_axis: str

    @property
    def position(self) -> Position:
        return (self.row, self.col)

class BreakthroughEnv:
    def __init__(
        self,
        height: int=8,
        width: int=8,
        start: Position=(7, 0),
        goal: Position=(0, 7),
        guards: Optional[list[Guard]]=None,
        obstacles: Optional[list[Position]]=None,
        max_steps: int=100,
    ) -> None:

        self.height = height
        self.width = width

        self.start = start
        self.goal = goal
        self.max_steps = max_steps


        if obstacles is None:
            obstacles=[
                (1, 2), (1, 3), (1, 4), (1, 5),
                (2, 2), (2, 5),
                (3, 2), (3, 3), (3, 5),
                (4, 4), (4, 5),
                (5, 1), (5, 2), (5, 4),
                (6, 4),
            ]

        if guards is None:
            guards=[
                Guard(row=2, col=6, direction=1, patrol_axis="vertical"),
                Guard(row=4, col=2, direction=3, patrol_axis="horizontal"),
                Guard(row=6, col=5, direction=3, patrol_axis="horizontal"),
                Guard(row=1, col=6, direction=1, patrol_axis="vertical"),
            ]

        self.obstacles = set(obstacles)
        self.initial_guards = guards
        self.hero_pos: Position = self.start
        self.guards: list[Guard] = []
        self.step_count = 0
        self.done = False

    @property
    def num_actions(self) -> int:
        return len(ACTION)

    def reset(self) -> State:
        """
        初始化hero与guard位置，并将计步归零，同时返回一个状态向量
        """
        self.hero_pos = self.start
        self.guards = [replace(guard) for guard in self.initial_guards]
        self.step_count = 0
        self.done = False
        return self.get_state()

    def get_state(self) -> State:
        """
        将环境变量 编码为 状态向量
        """
        state_values: list[int] = [self.hero_pos[0], self.hero_pos[1]]
        for guard in self.guards:
            state_values.extend([guard.row, guard.col, guard.direction])
        return tuple(state_values)

    def step(self, action: int) -> tuple[State, float, bool, dict[str, any]]:
        """
        执行一个动作，返回下一个状态、奖励、是否结束、信息
        """
        if self.done:
            raise RuntimeError("Episode has finished. Please call reset() before step().")

        if action not in ACTION:
            raise ValueError(f"Invalid action: {action}. Must be in range({self.num_actions}).")

        self.step_count += 1
        reward = -1.0
        info: dict = {
            "event": "running",
            "action_name": ACTION_NAME[action],
        }
        # 记录旧位置
        hero_old_position = self.hero_pos
        guard_old_position = [guard.position for guard in self.guards]
        # 移动一步
        hero_next_position, hit_wall = self._move_hero(action)
        #print(hero_next_position, hit_wall)
        self._move_guards()

        if hit_wall:
            reward = -2.0
            info["hit_wall"] = True
        else:
            info["hit_wall"] = False

        # 记录新位置
        self.hero_pos = hero_next_position
        guard_new_position = [guard.position for guard in self.guards]
        
        # 检测碰撞 与 抓捕
        guard_hits_hero = self.hero_pos in guard_new_position
        swap_collision = False #交换碰撞检测

        for guard_old_pos, guard_new_pos in zip(guard_old_position, guard_new_position):
            if hero_old_position == guard_new_pos and self.hero_pos == guard_old_pos:
                swap_collision = True
                break

        if guard_hits_hero or swap_collision:
            reward = -50.0
            self.done = True
            info["event"] = "caught"
        
        elif self.hero_pos == self.goal:
            reward = 100.0
            self.done = True
            info["event"] = "success"

        elif self.step_count >= self.max_steps:
            reward = -30.0
            self.done = True
            info["event"] = "timeout"

        # 记录状态信息
        next_state = self.get_state()
        return next_state, reward, self.done, info


    def _move_hero(self, action: int) -> tuple[Position, bool]:
        dr, dc = ACTION[action]
        row, col = self.hero_pos

        next_pos = (row + dr, col + dc)
        # 撞墙检测
        if not self._is_valid_position(next_pos):
            return self.hero_pos, True
        
        return next_pos, False

    def _move_guards(self) -> None:
        for guard in self.guards:
            next_pos = self._get_guard_next_position(guard)

            if self._is_valid_position(next_pos):
                guard.row, guard.col = next_pos
            else:
                guard.direction = REVERSE_ACTION[guard.direction]
                next_pos_after_reverse = self._get_guard_next_position(guard)

                if self._is_valid_position(next_pos_after_reverse):
                    guard.row, guard.col = next_pos_after_reverse

    def _get_guard_next_position(self, guard: Guard) -> Position:
        dr, dc = ACTION[guard.direction]
        return guard.row + dr, guard.col + dc

    def _is_valid_position(self, position: Position) -> bool:
        row, col = position

        if row < 0 or row >= self.height:
            return False

        if col < 0 or col >= self.width:
            return False

        if position in self.obstacles:
            return False

        return True

    def render_text(self) -> str:
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        for row, col in self.obstacles:
            grid[row][col] = "#"

        goal_row, goal_col = self.goal
        grid[goal_row][goal_col] = "T"

        for guard in self.guards:
            row, col = guard.position
            grid[row][col] = "G"

        hero_row, hero_col = self.hero_pos

        if grid[hero_row][hero_col] == "G":
            grid[hero_row][hero_col] = "X"
        else:
            grid[hero_row][hero_col] = "H"

        lines = []
        for row in grid:
            lines.append(" ".join(row))

        return "\n".join(lines)

    def print_state_info(self) -> None:
        print(f"Hero position: {self.hero_pos}")
        print(f"Goal position: {self.goal}")
        print(f"Step count: {self.step_count}/{self.max_steps}")

        for idx, guard in enumerate(self.guards):
            print(
                f"Guard {idx}: position={guard.position}, "
                f"direction={ACTION_NAME[guard.direction]}, "
                f"axis={guard.patrol_axis}"
            )

        print(f"State: {self.get_state()}")
    
    

        


