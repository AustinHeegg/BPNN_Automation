import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from torch import optim
from src.save_configs import *


class BPNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size, activation_function="sigmoid"):
        super(BPNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.output_size = output_size
        self.fcs = nn.ModuleList()

        self.fcs.append(nn.Linear(input_size, hidden_size))
        for _ in range(hidden_layer):
            self.fcs.append(nn.Linear(hidden_size, hidden_size))
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.activation = self.get_activation(activation_function)
        print(f"Activation function used: {activation_function}")

    def get_activation(self, activation_function):
        activation_functions = {
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "relu": torch.relu,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "selu": nn.SELU,
        }

        if activation_function in activation_functions:
            return activation_functions[activation_function]
        else:
            raise ValueError("Activation function is not supported!")

    def forward(self, x):
        for i in range(self.hidden_layer):
            x = self.activation(self.fcs[i](x))
        x = self.fc_out(x)
        return x

    @staticmethod
    def get_criterion(criterion_type):
        # 根据类型选择损失函数
        if criterion_type == 'mse':
            return nn.MSELoss()
        elif criterion_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError("Criterion type not supported!")

    @staticmethod
    def get_optimizer(optimizer_type, model, learning_rate):
        # 根据类型选择优化器
        if optimizer_type == 'sgd':
            return optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'adam':
            return optim.Adam(model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer type not supported!")

    @staticmethod
    def initialize_model(input_size, hidden_size, output_size, hidden_layers, learning_rate, activation_function,
                         criterion_type='mse', optimizer_type='sgd'):
        model = BPNN(input_size, hidden_size, output_size, hidden_layers, activation_function)
        criterion = BPNN.get_criterion(criterion_type)
        optimizer = BPNN.get_optimizer(optimizer_type, model, learning_rate)
        print(f'Criterion type: {criterion_type}')
        print(f'Optimizer type: {optimizer_type}')
        return model, criterion, optimizer

    @staticmethod
    def train_model(model, train_loader, criterion, optimizer, num_epochs, full_scale, val_loader=None):
        losses = []
        best_loss = float('inf')
        metrics = {}

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # 正向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # 计算并记录平均损失以及相对精度
            epoch_loss /= len(train_loader)
            losses.append(epoch_loss)
            relative_accuracy = 1 - (epoch_loss / full_scale)

            # Validate and check for model saving
            if val_loader is not None:

                # 计算性能指标
                predictions, targets = [], []
                with torch.no_grad():
                    for inputs, target in val_loader:
                        output = model(inputs)
                        predictions.extend(output.numpy())
                        targets.extend(target.numpy())

                predictions = np.array(predictions)
                targets = np.array(targets)
                mse = mean_squared_error(targets, predictions)
                r2 = r2_score(targets, predictions)

                metrics['MSE'] = mse
                metrics['R-Squared'] = r2
                metrics['Training_Loss'] = epoch_loss

            if (epoch + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}',
                      f'Relative Accuracy: {relative_accuracy:.4f}')

        return losses

    @staticmethod
    def test(model, X_test, y_test):
        model.eval()  # 将模型设置为评估模式
        with torch.no_grad():
            predictions = model(X_test)

            # 将张量转换为 NumPy 数组
            predictions_np = predictions.numpy()
            y_test_np = y_test.numpy()

            # 计算评估指标
            mse = mean_squared_error(y_test_np, predictions_np)
            r2 = r2_score(y_test_np, predictions_np)

            print("Performance Metrics:")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"R-squared Score: {r2:.4f}")

            return mse

    def loss_plot(losses):
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
