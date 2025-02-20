# BPNN_Automation

自动化的神经网络正向传播

bpnn_config为可改变的输入，根据需求进行修改。

data_dir: 训练数据所在文件夹\n
input_size: 输入节点数,
output_size: 输出节点数,
hidden_layer: 隐藏层层数,
hidden_size: 每层隐藏层节点数,

batch_size: 总batch size，一般不会改变,
learning_rate: 0.01,
num_epochs: 10000,
full_scale: 全量范围是-3.2至3.2，所以是6.4,

test_size: 选取多少百分比的数据作为test dataset,
random_state: 42
