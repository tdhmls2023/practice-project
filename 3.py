import time
import heapq
import numpy as np

# 目标状态：4x4网格，数字1-15按顺序排列，0表示空白格
goal_state = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]])


# 曼哈顿距离启发式函数：计算当前状态到目标状态的估计代价
def manhattan_distance(state):
    """
    计算曼哈顿距离：
    每个数字当前位置与目标位置在行和列上的绝对距离之和
        state: 当前状态（4x4 numpy数组）
        distance: 曼哈顿距离总和
    """
    distance = 0
    for i in range(4):  # 遍历所有行
        for j in range(4):  # 遍历所有列
            value = state[i, j]  # 当前位置的数字
            if value != 0:  # 空白格不计算距离
                # 计算数字在目标状态中的正确位置
                #   value-1是因为数字从1开始，而索引从0开始
                #   除以4得到行索引，取余得到列索引
                target_x, target_y = divmod(value - 1, 4)
                # 累加曼哈顿距离：|当前行-目标行| + |当前列-目标列|
                distance += abs(target_x - i) + abs(target_y - j)
    return distance


# 节点类：表示搜索树中的一个状态节点
class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        """
            g: 从起始节点到当前节点的实际代价（移动步数）
            h: 从当前节点到目标节点的估计代价（启发函数值）
        """
        self.state = state  # 当前状态
        self.parent = parent  # 父节点指针
        self.g = g  # 实际代价（已走步数）
        self.h = h  # 启发函数值（估计剩余步数）
        self.f = g + h  # 评估函数值（总代价估计）

    def __lt__(self, other):
        """
        重载小于运算符，用于优先队列比较
        优先队列（堆）会根据f值自动排序，选择f值最小的节点
        """
        return self.f < other.f


# A*算法实现
def a_star_solver(start_state):
    # A*搜索算法主函数

    start_time = time.time()  # 记录算法开始时间

    # g=0: 起始节点移动步数为0
    # h=manhattan_distance(start_state): 估计到目标状态的距离
    start_node = Node(start_state, g=0, h=manhattan_distance(start_state))

    open_list = []  # 使用Python的heapq实现最小堆
    closed_set = set()  # 使用集合快速查找

    # 将起始节点加入开放列表
    heapq.heappush(open_list, start_node)

    # 统计信息
    expanded_nodes = 0  # 已扩展节点数（从开放列表中取出的节点）
    generated_nodes = 1  # 已生成节点数（包括起始节点）

    # 主搜索循环
    while open_list:  # 当开放列表不为空时继续搜索
        # 从开放列表中取出f值最小的节点（堆顶元素）
        current_node = heapq.heappop(open_list)
        expanded_nodes += 1  # 统计扩展节点数

        # 检查是否到达目标状态
        if np.array_equal(current_node.state, goal_state):
            # 找到解决方案，记录结束时间
            end_time = time.time()
            # 重建解决方案路径
            solution_path = reconstruct_path(current_node)
            return solution_path, expanded_nodes, generated_nodes, end_time - start_time

        # 将当前节点状态加入关闭集合
        # 将numpy数组转换为字符串作为键（集合不能直接存储numpy数组）
        closed_set.add(str(current_node.state))

        # 扩展当前节点的所有邻居（可能的下一个状态）
        for neighbor in get_neighbors(current_node):
            generated_nodes += 1  # 统计生成节点数

            # 检查邻居节点是否已在关闭集合中（避免重复探索）
            if str(neighbor.state) in closed_set:
                continue  # 跳过已探索的状态

            # 将邻居节点加入开放列表
            heapq.heappush(open_list, neighbor)

    # 如果开放列表为空仍未找到解，返回None
    return None, expanded_nodes, generated_nodes, time.time() - start_time


# 获取当前节点的所有合法邻居节点
def get_neighbors(node):
    # 生成当前节点的所有可能移动（邻居节点）

    neighbors = []
    state = node.state  # 当前状态
    # 找到空白格（0）的位置
    zero_positions = np.where(state == 0)
    # zero_positions[0] 是行索引数组，zero_positions[1] 是列索引数组
    x, y = zero_positions[0][0], zero_positions[1][0]  # 提取第一个元素

    # 定义四个可能的移动方向：上、下、左、右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 尝试每个方向
    for dx, dy in directions:
        nx, ny = x + dx, y + dy  # 计算移动后的位置

        # 检查移动是否合法（不超出边界）
        if 0 <= nx < 4 and 0 <= ny < 4:
            # 创建新状态（深拷贝，避免修改原状态）
            new_state = state.copy()
            # 交换空白格和相邻数字的位置
            new_state[x, y], new_state[nx, ny] = new_state[nx, ny], new_state[x, y]

            # 计算启发函数值（曼哈顿距离）
            h = manhattan_distance(new_state)

            # g = node.g + 1: 实际代价增加1步
            # h = 曼哈顿距离：估计剩余代价
            new_node = Node(new_state, parent=node, g=node.g + 1, h=h)

            neighbors.append(new_node)

    return neighbors


# 重建从起始节点到目标节点的路径
def reconstruct_path(node):
    # 从目标节点开始，通过父节点指针回溯到起始节点

    path = []
    # 从目标节点开始，一直回溯到起始节点（起始节点的parent为None）
    while node:
        path.append(node.state)  # 添加当前状态到路径
        node = node.parent  # 移动到父节点
    # 反转路径，使顺序从起始状态到目标状态
    return path[::-1]  # [::-1]是Python的切片反转操作


# 打印解决方案路径
def print_solution(solution_path):
    print("解法:")
    print("-" * 40)
    for i, state in enumerate(solution_path):
        print(f"步骤 {i}:")
        print(state)  # 打印状态矩阵
        print()


# 主函数
def main():
    # 设置起始状态（可修改为其他合法状态）
    start_state = np.array(
        [[1, 2, 3, 4], [5, 6, 0, 8], [9, 10, 7, 11], [13, 14, 15, 12]]
    )

    print("起始状态:")
    print(start_state)
    print("\n目标状态:")
    print(goal_state)
    print("\n正在使用A*算法搜索解决方案...")
    print("-" * 40)

    # 运行A*算法
    solution_path, expanded_nodes, generated_nodes, runtime = a_star_solver(start_state)

    # 输出结果
    if solution_path:
        print_solution(solution_path)
        print("-" * 40)
        print("算法统计信息:")
        print(f"解决方案步数: {len(solution_path) - 1}")  # 减1是因为起始状态不算一步
        print(f"扩展节点数: {expanded_nodes}")
        print(f"生成节点数: {generated_nodes}")
        print(f"运行时间: {runtime:.8f} 秒")
        print(f"平均每步搜索时间: {runtime / (len(solution_path) - 1):.8f} 秒" if len(solution_path) > 1 else "N/A")
    else:
        print("没有找到解决方案！")
        print("可能的原因:")
        print("1. 起始状态不可解")
        print("2. 算法内存/时间不足")
        print("3. 程序有错误")


# 程序入口
if __name__ == "__main__":
    main()