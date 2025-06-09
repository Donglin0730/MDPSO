import math
import os
import random
import time
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearnex.neighbors import NearestNeighbors
from scipy.spatial import KDTree

print(f"运行时间{datetime.now()}")

# 参数定义
city_length = 50
num_stations = 50
num_hot_points = 5
num_particles = 50
max_radius = 6
num_specialarea = 5
base_signal = 100
special_radius = 2
iterations = 200
total_count = 20
c1 = 2.0
c2 = 2.0

# 基站半径
station_radius = [max_radius] * num_stations

# 生成所有可能的空间点，最后用来计算覆盖率
def generate_points(city_length, step_size=1):
    points = [(x, y) for x in range(0, city_length + 1, step_size)
              for y in range(0, city_length + 1, step_size)]
    return points


# 所有可能的空间点，用来计算覆盖率
points = generate_points(city_length)


# 生成特殊区域
def generate_specialarea(num_specialarea):
    specialarea = [(random.uniform(0, city_length), random.uniform(0, city_length)) for _ in
                   range(num_specialarea)]  # list
    return specialarea


def check_illegal(stations, specialarea):
    for i in range(len(stations)):
        for j in range(len(specialarea)):
            dis = np.linalg.norm(stations[i] - specialarea[j])
            if dis < special_radius:
                return True
            else:
                continue


# 随机生成粒子
def initialize_particles(num_particles, num_stations):
    # 初始化粒子数组
    particles = np.zeros((num_particles, num_stations * 2))

    for i in range(num_particles):
        for j in range(num_stations):
            station_x = np.random.uniform(0, city_length)
            station_y = np.random.uniform(0, city_length)

            # 将坐标填入粒子数组
            particles[i, j * 2] = station_x
            particles[i, j * 2 + 1] = station_y

    return particles


# 得到热点连接基站的字典
def assign_points_to_stations(stations, hot_points):
    point_to_station = {}  # 用来记录每个点与哪个基站相连, 键为坐标元组，值为基站索引
    # 遍历每个热点，尝试找到最优的基站连接
    for index, point in enumerate(hot_points):
        point_to_station[tuple(point)] = 0
        for i, station in enumerate(stations):
            # 计算距离
            distance = np.linalg.norm(point - station)
            signal = base_signal - (32.4 + 20 * math.log10(base_signal) + 20 * math.log10(distance))
            # 如果点在基站的覆盖半径之内,则表示可以连接
            if distance <= station_radius[i] and signal > signal_need[index]:
                point_to_station[tuple(point)] += 1
                continue
    return point_to_station


def calculate_coverage_ratio(stations, points):  # 计算三维空间中被覆盖的点比例
    all_points = len(points)
    covered_points = 0
    for i in range(len(points)):
        for j in range(len(stations)):
            distance = np.linalg.norm(points[i] - stations[j])
            if distance <= station_radius[j]:
                covered_points += 1
                break
    covered_ratio = covered_points / all_points
    return covered_ratio


# 计算适应度函数
def fitness_function(particles):
    fitness_values = np.zeros(num_particles)
    for i in range(num_particles):
        stations = particles[i].reshape(num_stations, 2)
        point_to_station = assign_points_to_stations(stations, hot_points)
        a = calculate_coverage_ratio(stations, points)
        fitness = -a
        # 惩罚项
        # 对覆盖次数不足的粒子进行惩罚
        k = 0
        for j in point_to_station.values():
            if j < signal_cover[k]:
                fitness += 10000 * (signal_cover[k] - j)
            k += 1
        # 对处于非方法区域的粒子进行惩罚
        if check_illegal(stations, specialarea):
            fitness += 10000
        fitness_values[i] = fitness

    return fitness_values


def caculate_neibor(particles, pbest, pbest_fitness):
    num_particles = len(particles)
    neibor = [0] * num_particles
    neibor_pbest = pbest.copy()

    # 构建k-d树用于空间优化
    tree = KDTree(particles)

    for i in range(num_particles):
        # 查询距离粒子i最近的其他粒子
        dist, idx = tree.query(particles[i], k=num_particles)

        # 在找到的邻居中，挑选适应度更好的粒子
        best_fit_particle = i
        best_fit_value = pbest_fitness[i]

        # 遍历最近的邻居
        for j in idx[1:]:  # 从第二个元素开始，因为第一个元素是自己
            if pbest_fitness[j] < best_fit_value:  # 适应度更好
                best_fit_particle = j
                best_fit_value = pbest_fitness[j]

        # 记录最优邻居
        neibor[i] = best_fit_particle
        neibor_pbest[i] = pbest[best_fit_particle]

    return neibor, neibor_pbest


def calculate_density(particles, k=5):
    # 使用k近邻找到每个粒子的k个最近邻
    nbrs = NearestNeighbors(n_neighbors=k).fit(particles)
    distances, indices = nbrs.kneighbors(particles)
    # 计算密度：使用k个邻居的平均距离的倒数作为密度
    densities = k / np.sum(distances[:, 1:], axis=1)  # 距离的第一个邻居是自己
    return densities


# 可视化
def plot_bases_and_points_2d(bases, station_radius, points, specialarea):
    plt.figure(figsize=(10, 10))

    # 绘制热点 (points)，调整透明度 (alpha)
    plt.scatter(points[:, 0], points[:, 1], color='red', alpha=0.6, label='Special Points', s=50)

    # 绘制基站 (bases)，调整透明度 (alpha)
    plt.scatter(bases[:, 0], bases[:, 1], color='blue', alpha=0.8, label='Base Stations')

    # 绘制非法区域
    specialarea = np.array(specialarea).flatten().reshape(num_specialarea, 2)
    plt.scatter(specialarea[:, 0], specialarea[:, 1], color='black', alpha=0.8, label='Illegal Area')

    # 绘制基站的覆盖范围（圆形）
    for base, radius in zip(bases, station_radius):
        circle = plt.Circle((base[0], base[1]), radius, color='green', alpha=0.3)
        plt.gca().add_artist(circle)

    # 绘制非法区域
    for special in specialarea:
        circle = plt.Circle((special[0], special[1]), 2, color='gray', alpha=0.5)
        plt.gca().add_artist(circle)

    # 设置坐标范围
    plt.xlim([0, city_length])
    plt.ylim([0, city_length])

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 保持长宽比例一致
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend()
    plt.title('Base Stations and Special Points')
    plt.grid()

    plt.show()


# 保存结果
def save_to_excel(file_path, data_array):
    # 尝试读取现有的Excel文件
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
    else:
        existing_df = pd.DataFrame()  # 如果文件不存在，则创建一个新的空DataFrame

    # 获取当前的列数，用来决定新列的位置
    new_column_index = existing_df.shape[1] + 1

    # 将数据转换为 DataFrame
    data_df = pd.DataFrame(data_array)

    # 将新的列添加到现有的 DataFrame
    existing_df[f'Column_{new_column_index}'] = data_df.iloc[:, 0]  # 只取数据的第一列

    # 保存更新后的 DataFrame 到 Excel 文件
    existing_df.to_excel(file_path, index=False)


# 主函数
if __name__ == "__main__":
    # 随机初始化粒子位置和速度
    best_particle = np.zeros(num_stations * 2)
    best_particle_fitness = 1e18
    total_gbestfitness = np.zeros((total_count, iterations + 1))
    for count in range(total_count):
        start_time = time.time()

        # 随机初始化
        hot_points = np.array([np.random.uniform(0, city_length, num_hot_points), np.random.uniform(0, city_length, num_hot_points)]).T
        signal_cover = [random.randint(1, 3) for _ in range(num_hot_points)]  # 信号点的最小信号需求
        signal_need = [random.randint(10, 15) for _ in range(num_hot_points)]
        specialarea = generate_specialarea(num_specialarea)
        # 固定位置
        # hot_points = [[20.5512779, 18.13451444], [41.31778478, 22.97795211], [4.02745779, 6.89095213],
        #               [37.75660687, 26.03606089], [36.30811607, 9.30089926]]
        # specialarea = [(22.293950952145863, 4.66454375021112), (49.77274338808096, 6.220277742272284),
        #                (11.264934035933887, 10.345654294574864), (25.984632729674082, 25.33536699142813),
        #                (15.397345243763949, 30.435698950482664)]

        # 初始化粒子
        particles = initialize_particles(num_particles, num_stations)
        velocities = np.random.uniform(-1, 1, size=(num_particles, num_stations * 2))

        pbest = particles.copy()
        gbest = particles[np.random.randint(num_particles)].copy()  # (100,)
        gworst = particles[np.random.randint(num_particles)].copy()

        # 计算适应度
        pbest_fitness = np.array([fitness_function(particles)])  # (tool,50)
        gbest_fitness = np.min(pbest_fitness)
        gworst_fitness = np.max(pbest_fitness)
        better = particles[np.argsort(pbest_fitness[:3])]

        gbest = pbest[np.argmin(pbest_fitness)]
        gworst = pbest[np.argmax(pbest_fitness)]

        gbestfitness = np.zeros(iterations)
        n = 0  # 停滞因子
        neibor, neibor_pbest = caculate_neibor(particles, pbest, pbest_fitness.flatten())
        for run in range(iterations):
            dis = np.linalg.norm(gworst - particles)
            P = run / iterations
            # 更新速度和位置
            r = random.uniform(0, 1)
            w = 0.9
            if r > P:  # 前期，探索
                if run // 5 == 0:
                    neibor, neibor_pbest = caculate_neibor(particles, pbest, pbest_fitness.flatten())
                velocities = (w * velocities + c1 * np.random.rand(num_particles, num_stations * 2) * (
                        (neibor_pbest + pbest) / 2 - particles) + c2 * np.random.rand(num_particles,
                                                                                      num_stations * 2) * (
                                      gbest - particles) - np.random.rand(
                    num_particles, num_stations * 2) * (gworst - particles))
            else:  # 后期，开发
                velocities = (w * velocities + c1 * np.random.rand(num_particles, num_stations * 2) * (
                        pbest - particles) + c2 * np.random.rand(num_particles, num_stations * 2) * (gbest - particles))
            particles = particles + velocities

            LOGIC = (particles < 0) | (particles > city_length)
            U = np.random.uniform(0, city_length, (num_particles, num_stations * 2))
            # 处理越界粒子，越界了的用最随机位置代替
            particles = LOGIC * U + (1 - LOGIC) * particles

            # 计算当前适应度
            current_fitness = fitness_function(particles)
            # 更新个体最优，如果当前适应度低，则更新为当前，否则保留之前的个人最佳
            better_fitness_mask = current_fitness < pbest_fitness
            LOGIC = better_fitness_mask.transpose()
            pbest = LOGIC * particles + (1 - LOGIC) * pbest
            pbest_fitness = better_fitness_mask * current_fitness + (1 - better_fitness_mask) * pbest_fitness

            # 更新全局最差
            if np.max(current_fitness) > gworst_fitness:
                gworst = particles[np.argmax(current_fitness)].copy()
                gworst_fitness = np.max(current_fitness)

            # 更新全局最优
            if np.min(current_fitness) < gbest_fitness:
                gbest = particles[np.argmin(current_fitness)].copy()
                gbest_fitness = np.min(current_fitness)
                n = 0
            else:
                # 如果全局最优没有更新，停滞因子n增加
                num = int(num_particles * 0.1)
                n += 1
                prob = (math.exp(n) - 1) / 100 * run / iterations
                if random.uniform(0, 1) < prob:
                    # 陷入了局部最优
                    worst_indices = np.argsort(pbest_fitness.flatten())[-num:]
                    # 计算密度
                    densities = calculate_density(particles)
                    # 找出密度最小的粒子
                    min_density_index = np.argsort(densities)[:num]
                    min_density_particle = particles[min_density_index]
                    particles[worst_indices] = min_density_particle

            gbestfitness[run] = gbest_fitness
            print(f"Iteration {run + 1}: current_fitness = {gbest_fitness}")

        end_time = time.time()
        execution_time = end_time - start_time
        print("代码块执行时间: " + str(execution_time) + " 秒")
        gbestfitness = np.append(gbestfitness, execution_time)

        if (gbest_fitness < best_particle_fitness):
            best_particle = gbest.copy()
            best_particle_fitness = gbest_fitness

        file_path = r'path.xlsx'
        save_to_excel(file_path, gbestfitness)

        total_gbestfitness[count] = gbestfitness
        print(f"第{count + 1}次迭代：最优位置为{gbest}")
    print(f"平均适应度：{np.mean(total_gbestfitness[:, -2])}")
    print(f"平均时间：{np.mean(total_gbestfitness[:, -1])}")

# 导出最优粒子到excel
best_particle = ', '.join(map(str, best_particle))
existing_df = pd.read_excel(file_path)

# 如果文件不为空，添加新列到最后的第一行
if not existing_df.empty:
    # 获取当前的列数，确定新列的名称
    new_column_name = f'Column_{existing_df.shape[1] + 1}'
    # 在新列的第一个单元格中放入列表字符串
    existing_df.loc[0, new_column_name] = best_particle
else:
    # 如果文件为空，创建一个新的 DataFrame 并添加到第一列
    existing_df = pd.DataFrame(columns=[f'Column_1'])  # 创建第一列
    existing_df.loc[0, 'Column_1'] = best_particle  # 将列表字符串放入第一个单元格

# 保存更新后的 DataFrame 到 Excel 文件
existing_df.to_excel(file_path, index=False)
