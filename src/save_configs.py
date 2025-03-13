import torch
import json


# def generate_layer_names(input_size, output_size, hidden_layers, hidden_size):
#     layer_names = {
#         "weight_layers": [],
#         "bias_layers": []
#     }
#
#     layer_names["weight_layers"].append("ModelWeightLayer_Input")
#     for i in range(hidden_layers):
#         layer_names["weight_layers"].append(f"ModelWeightLayer_{i + 1}")
#     layer_names["weight_layers"].append("ModelWeightLayer_Output")
#
#     layer_names["bias_layers"].append("ModelBiasLayer_Input")
#     for i in range(hidden_layers):
#         layer_names["bias_layers"].append(f"ModelBiasLayer_{i + 1}")
#     layer_names["bias_layers"].append("ModelBiasLayer_Output")
#
#     return layer_names


def save_model_to_json(model, metrics=None, model_path=None, config=None):
    if config is None:
        raise ValueError("Configuration must be provided.")

    input_size = config['input_size']
    output_size = config['output_size']
    hidden_layers = config['hidden_layer']
    hidden_nodes = config['hidden_nodes']
    activation_functions = config['activation_function']

    layer_count = 1 + hidden_layers + 1
    layer_node_num = [input_size] + hidden_nodes + [output_size]

    # 生成层名称
    # layer_names = generate_layer_names(input_size, output_size, hidden_layers, hidden_nodes)

    # 封装模型参数结构
    model_data = {
        "model_param": {
            "layer_num":layer_count,
            "layer_neuron_num":layer_node_num,
            "activation_function":activation_functions,
            "model_coff": {},
            "#model_metrics": {
                "mse": metrics.get('MSE'),
            }
        }
    }

    # 动态生成层结构
    for i in range(hidden_layers):
        layer_name = f"bp_{i + 1}_layer"
        model_data["model_param"]["model_coff"][layer_name] = {
            "weight": [],
            "bias": []
        }

    # 输出层的名称
    output_layer_name = f"bp_{hidden_layers + 1}_layer"
    model_data["model_param"]["model_coff"][output_layer_name] = {
        "weight": [],
        "bias": []
    }

    # 遍历模型的所有参数并填充模型权重和偏置
    for name, param in model.named_parameters():
        param_data = param.data.numpy().tolist()  # 转换为可序列化格式
        print(f"Parameter name: {name}, Parameter value: {param.data.numpy()}")

        if 'bias' in name:
            # 匹配层次
            if '0' in name:  # 输入层偏置
                model_data["model_param"]["model_coff"]["bp_1_layer"]["bias"] = param_data
            elif 'out' in name:  # 输出层偏置
                model_data["model_param"]["model_coff"][output_layer_name]["bias"] = param_data
            else:  # 隐藏层偏置
                layer_index = int(name.split('.')[1])  # 从参数名提取层索引
                model_data["model_param"]["model_coff"][f"bp_{layer_index + 1}_layer"]["bias"] = param_data
        else:  # 权重
            if 'weight' in name:
                if '0' in name:  # 输入层权重
                    model_data["model_param"]["model_coff"]["bp_1_layer"]["weight"] = param_data
                elif 'out' in name:  # 输出层权重
                    model_data["model_param"]["model_coff"][output_layer_name]["weight"] = param_data
                else:  # 隐藏层权重
                    layer_index = int(name.split('.')[1])  # 从参数名提取层索引
                    model_data["model_param"]["model_coff"][f"bp_{layer_index + 1}_layer"]["weight"] = param_data

    # 写入 JSON 文件
    with open(model_path, 'w') as json_file:
        json.dump(model_data, json_file, indent=4)

    print(f"Model saved to {model_path}")
