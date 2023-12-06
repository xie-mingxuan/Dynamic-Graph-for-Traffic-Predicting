import numpy as np
import pandas as pd


def save_subway_info(filepath: str):
    """
    将地铁数据和对应的 id 映射关系储存到 npy 文件中
    :param filepath: 地铁站点信息
    :return:
    """
    df = pd.read_csv(filepath, encoding='utf-8')
    station_list = []
    for index, row in df.iterrows():
        if row["station"] not in station_list:
            station_list.append(row["station"])
    station_array = np.array(station_list)

    np.save("../DG_data/station_info_list.npy", station_array)
    return station_array


def save_station_distance(info_path: str, distance_path: str):
    """
    将地铁各个站点之间的距离转化为邻接矩阵
    :param info_path: 地铁站点信息
    :param distance_path: 地铁站之间的距离信息
    """
    station_array = save_subway_info(info_path)
    distance_matrix = np.full((station_array.size, station_array.size), np.inf)

    np.fill_diagonal(distance_matrix, 0)

    df = pd.read_csv(distance_path)
    for index, row in df.iterrows():
        get_on_station = row["station1"]
        get_off_station = row["station2"]
        get_on_station_id = np.where(station_array == get_on_station)[0][0]
        get_off_station_id = np.where(station_array == get_off_station)[0][0]
        is_double_direction = row["double_direction"]

        distance_matrix[get_on_station_id][get_off_station_id] = row["distance"]
        if is_double_direction:
            distance_matrix[get_off_station_id][get_on_station_id] = row["distance"]

    np.save("../DG_data/station_neighbor_distance.npy", distance_matrix)


def floyd_warshall(matrix_path: str):
    """
    floyd 算法，用于从得到的邻接矩阵中计算出任意两点之间的最短距离
    :param matrix_path: 邻接矩阵路径
    """
    adjacency_matrix = np.load(matrix_path)
    num_vertices = adjacency_matrix.shape[0]
    # 初始化距离矩阵
    distance_matrix = np.copy(adjacency_matrix)
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                distance_matrix[i, j] = min(distance_matrix[i, j], distance_matrix[i, k] + distance_matrix[k, j])

    np.save("../DG_data/station_distance.npy", distance_matrix)


if __name__ == '__main__':
    save_station_distance("../DG_data/station_info.csv", "../DG_data/station_distance.csv")
    floyd_warshall("../DG_data/station_neighbor_distance.npy")

    station_array = np.load("../DG_data/station_info_list.npy")
    neighbor_distance_matrix = np.load("../DG_data/station_neighbor_distance.npy")
    distance_matrix = np.load("../DG_data/station_distance.npy")
    print(station_array.size)
    print(neighbor_distance_matrix.size)
    print(distance_matrix.size)
