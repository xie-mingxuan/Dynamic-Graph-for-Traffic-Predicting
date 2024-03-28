import pandas as pd


def statistic_record(path: str, trans: str = "railway"):
    with open(path, 'r', encoding="utf-8", errors="ignore") as f:
        user_data = pd.read_csv(f, header=None, dtype=str)

    user_record = {}

    for index, row in user_data.iterrows():
        if not (trans == "railway" and row[2] == "R") or (trans == "bus" and row[2] == "B"):
            continue

        user = row[1]
        if user not in user_record:
            user_record[user] = 1
        else:
            user_record[user] += 1

    sorted_user_record = sorted(user_record.items(), key=lambda x: x[1], reverse=True)
    print(sorted_user_record)
    count_gt_10 = 0
    count_total = 0
    for pair in sorted_user_record:
        count_total += 1
        if pair[1] > 10:
            count_gt_10 += 1

    print(f"count > 10 = {count_gt_10}")
    print(f"total_count = {count_total}")


def get_person_record_ge(path: str, trans: str = "railway", record_num: int = 10):
    with open(path, 'r', encoding="utf-8", errors="ignore") as f:
        user_data = pd.read_csv(f, header=None, dtype=str)

    user_record = {}

    for index, row in user_data.iterrows():
        if not (trans == "railway" and row[2] == "R") or (trans == "bus" and row[2] == "B"):
            continue

        user = row[1]
        if user not in user_record:
            user_record[user] = 1
        else:
            user_record[user] += 1

    user_list = []
    for user in user_record:
        if user_record[user] >= record_num:
            user_list.append(user)
    return user_list


if __name__ == '__main__':
    statistic_record('../DG_data/t000/t000.csv')
    print(get_person_record_ge('../DG_data/t000/t000.csv', record_num=10))
