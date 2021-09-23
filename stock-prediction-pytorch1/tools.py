import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


def read_file(file_path):
    """ 读取文件

    :param file_path:
    :return:
    """
    data = pd.read_csv(file_path).sort_values('Date')
    print(data.head())
    return data


def split_data(stock, lookback):
    data_raw = stock.to_numpy()  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data);
    test_set_size = int(np.round(0.2 * data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]


def plt_show_source(data):
    """ 查看数据趋势

    :param data:
    :return:
    """
    plt.figure(figsize=(15, 9))
    plt.plot(data[['Close']])
    plt.xticks(range(0, data.shape[0], 500), data['Date'].loc[::500], rotation=45)
    plt.title("Amazon Stock Price", fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price (USD)', fontsize=18)
    plt.show()
