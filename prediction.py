import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 读取CSV数据
data = pd.read_csv('out_15min.csv')  # 替换为您的CSV文件路径

# 假设CSV文件中有一个名为'1'的列，包含客流数据
data['1'] = data['1'].astype(float)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data['1'] = scaler.fit_transform(data['1'].values.reshape(-1, 1))

# 创建输入输出序列
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1
X, Y = create_dataset(data['1'].values.reshape(-1, 1), look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 实例化模型、定义损失函数和优化器
input_size = 1
hidden_size = 120
num_layers = 3
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 140
for i in range(epochs):
    model.train()
    outputs = model(torch.from_numpy(X_train).float())
    optimizer.zero_grad()
    loss = criterion(outputs, torch.from_numpy(Y_train).float().unsqueeze(1))
    loss.backward()
    optimizer.step()
    if (i+1) % 10 == 0:
        print(f'Epoch [{i+1}/{epochs}], Loss: {loss.item():.4f}')

# 预测
model.eval()
predicted = model(torch.from_numpy(X_test).float())
predicted = predicted.detach().numpy()

# 反归一化
predicted = scaler.inverse_transform(predicted)
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

# 评估模型
mse = mean_squared_error(Y_test, predicted)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, predicted)
# 计算Adjusted R2
n = len(Y_test)  # 样本数量
p = 1  # 特征数量
adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

print(f'Test MSE: {mse}')
print(f'Test RMSE: {rmse}')
print(f'Test R2: {r2}')
print(f'Test Adjusted R2: {adjusted_r2}')

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(Y_test, label='Actual')
plt.plot(predicted, label='Predicted')
plt.title('Passenger Flow Prediction')
plt.xlabel('Time')
plt.ylabel('Passenger Flow')
plt.legend()
plt.show()