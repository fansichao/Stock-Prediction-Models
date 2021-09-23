""" 股票预测模型
activate ml-py36


https://github.com/fansichao/stock-prediction-pytorch
https://github.com/fansichao/Stock-Prediction-Models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch

import math, time
from sklearn.metrics import mean_squared_error

sns.set_style("darkgrid")

from models import LSTM, GRU
from tools import read_file, split_data, plt_show_source
import plotly.express as px
import plotly.graph_objects as go


# import chart_studio.plotly as py

class StockPrediction():
    """ 股票预测
    """

    def __init__(self):
        # 输入参数
        self.input_params = dict(
            # 输入维度
            input_dim=1,
            # 隐藏维度
            hidden_dim=32,
            # 堆叠LSTM的层数，默认值为1
            num_layers=1,
            # 输出层
            output_dim=1,
            # num_epochs=1000,
            # 过了N遍训练集中的所有样本
            num_epochs=1000,
            # 给定过去的数据
            lookback=200,
            # lookback=1000,
            # lr=0.01,
            # learning_rate 学习速率
            lr=0.003,
        )
        # 数据
        self.data = []

    def prepare_data(self, file_path):
        csv_data = read_file(file_path)
        # plt_show_source(csv_data)
        self.data = csv_data[['Close']]

    def prepare_train(self):
        """ 准备阶段
        """
        # MinMaxScaler 数据归一化
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        # fit_transoform 
        # 对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），
        # 然后对该数据进行转换transform，从而实现数据的标准化、归一化
        self.data['Close'] = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))

        # x_train, y_train, x_test, y_test = split_data(self.data, self.input_params['lookback'])
        # print('x_train.shape = ', x_train.shape)
        # print('y_train.shape = ', y_train.shape)
        # print('x_test.shape = ', x_test.shape)
        # print('y_test.shape = ', y_test.shape)
        # self.x_train = torch.from_numpy(x_train).type(torch.Tensor)
        # self.x_test = torch.from_numpy(x_test).type(torch.Tensor)
        # self.y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
        # self.y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
        # self.y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
        # self.y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

    def train_with_lstm(self):
        x_train, y_train, x_test, y_test = split_data(self.data, self.input_params['lookback'])
        print('x_train.shape = ', x_train.shape)
        print('y_train.shape = ', y_train.shape)
        print('x_test.shape = ', x_test.shape)
        print('y_test.shape = ', y_test.shape)
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
        y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
        y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
        y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

        model = LSTM(input_dim=self.input_params['input_dim'],
                     hidden_dim=self.input_params['hidden_dim'],
                     output_dim=self.input_params['output_dim'],
                     num_layers=self.input_params['num_layers'])

        criterion = torch.nn.MSELoss(reduction='mean')
        # optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        # Adam 一种可以替代传统随机梯度下降过程的一阶优化算法，它能基于训练数据迭代地更新神经网络权重
        optimiser = torch.optim.Adam(model.parameters(), lr=self.input_params['lr'])

        hist = np.zeros(self.input_params['num_epochs'])
        start_time = time.time()
        lstm = []

        # 随机梯度下降
        for t in range(self.input_params['num_epochs']):
            y_train_pred = model(x_train)

            loss = criterion(y_train_pred, y_train_lstm)
            print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()

            # 将模型的参数梯度初始化为 0  
            optimiser.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新所有参数
            optimiser.step()

        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))

        # 将标准化后的数据转换为原始数据
        predict = pd.DataFrame(self.scaler.inverse_transform(y_train_pred.detach().numpy()))
        original = pd.DataFrame(self.scaler.inverse_transform(y_train_lstm.detach().numpy()))

        print(predict)
        fig = plt.figure()
        # 调整子图布局
        fig.subplots_adjust(hspace=0.2, wspace=0.2)

        # 股票价格
        plt.subplot(1, 2, 1)
        ax = sns.lineplot(x=original.index, y=original[0], label="Data", color='royalblue')
        ax = sns.lineplot(x=predict.index, y=predict[0], label="Training Prediction (LSTM)", color='tomato')
        ax.set_title('Stock price', size=14, fontweight='bold')
        ax.set_xlabel("Days", size=14)
        ax.set_ylabel("Cost (USD)", size=14)
        ax.set_xticklabels('', size=10)
        plt.show()

        # # 训练损失
        # plt.subplot(1, 2, 2)
        # print(hist)
        # ax = sns.lineplot(data=hist, color='royalblue')
        # ax.set_xlabel("Epoch", size=14)
        # ax.set_ylabel("Loss", size=14)
        # ax.set_title("Training Loss", size=14, fontweight='bold')
        # fig.set_figheight(6)
        # fig.set_figwidth(16)
        # fig.show()

        # > 数据预测
        # make predictions
        y_test_pred = model(x_test)

        # invert predictions
        # X = scaler.inverse_transform(X[, copy]) 将标准化后的数据转换为原始数据
        y_train_pred = self.scaler.inverse_transform(y_train_pred.detach().numpy())
        y_train = self.scaler.inverse_transform(y_train_lstm.detach().numpy())
        y_test_pred = self.scaler.inverse_transform(y_test_pred.detach().numpy())
        y_test = self.scaler.inverse_transform(y_test_lstm.detach().numpy())

        # mean_squared_error 均方误差
        trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))
        lstm.append(trainScore)
        lstm.append(testScore)
        lstm.append(training_time)

        # > train and test
        # empty_like 生成和已有数组相同大小，类型的数组
        trainPredictPlot = np.empty_like(self.data)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[self.input_params['lookback']:len(y_train_pred) + self.input_params['lookback'], :] = y_train_pred

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(self.data)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(y_train_pred) + self.input_params['lookback'] - 1:len(self.data) - 1, :] = y_test_pred

        original = self.scaler.inverse_transform(self.data['Close'].values.reshape(-1, 1))

        predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
        predictions = np.append(predictions, original, axis=1)
        result = pd.DataFrame(predictions)

        # >> fig
        fig = go.Figure()
        fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                                            mode='lines',
                                            name='Train prediction')))
        fig.add_trace(go.Scatter(x=result.index, y=result[1],
                                 mode='lines',
                                 name='Test prediction'))
        fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                                            mode='lines',
                                            name='Actual Value')))
        fig.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=True,
                showticklabels=False,
                linecolor='white',
                linewidth=2
            ),
            yaxis=dict(
                title_text='Close (USD)',
                titlefont=dict(
                    family='Rockwell',
                    size=12,
                    color='white',
                ),
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='white',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Rockwell',
                    size=12,
                    color='white',
                ),
            ),
            showlegend=True,
            template='plotly_dark'
        )

        annotations = []
        annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                xanchor='left', yanchor='bottom',
                                text='Results (LSTM)',
                                font=dict(family='Rockwell',
                                          size=26,
                                          color='white'),
                                showarrow=False))
        fig.update_layout(annotations=annotations)

        fig.show()
        #   py.iplot(fig, filename='stock_prediction_lstm')
        return lstm
    #
    # def diff_with_res(self, lstm_data, gru_data, fig):
    #     lstm = pd.DataFrame(lstm_data, columns=['LSTM'])
    #     gru = pd.DataFrame(gru_data, columns=['GRU'])
    #     result = pd.concat([lstm, gru], axis=1, join='inner')
    #     result.index = ['Train RMSE', 'Test RMSE', 'Train Time']
    #     py.iplot(fig, filename='stock_prediction_gru')
    #


if __name__ == '__main__':
    file_path = 'data/all_stocks_2006-01-01_to_2018-01-01.csv'
    file_path = 'data/all_stocks_2017-01-01_to_2018-01-01.csv'
    file_path = 'data/CSCO_2006-01-01_to_2018-01-01.csv'
    file_path = 'data/002594.SZ.20120101-20210828.csv'

    stp = StockPrediction()
    stp.prepare_data(file_path=file_path)
    stp.prepare_train()
    lstm_data = stp.train_with_lstm()
    # fig, gru_data = stp.train_with_gru()
    # stp.diff_with_res(lstm_data, gru_data=gru_data, fig=fig)
