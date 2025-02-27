import torch
import json


def generate_layer_names(input_size, output_size, hidden_layers, hidden_size):
    layer_names = {
        "weight_layers": [],
        "bias_layers": []
    }

    layer_names["weight_layers"].append("ModelWeightLayer_Input")
    for i in range(hidden_layers):
        layer_names["weight_layers"].append(f"ModelWeightLayer_{i}")
    layer_names["weight_layers"].append("ModelWeightLayer_Output")

    layer_names["bias_layers"].append("ModelBiasLayer_Input")
    for i in range(hidden_layers):
        layer_names["bias_layers"].append(f"ModelBiasLayer_{i}")
    layer_names["bias_layers"].append("ModelBiasLayer_Output")

    return layer_names


def save_model_to_json(model, metrics=None, model_path='best_model.json', config=None):
    if config is None:
        raise ValueError("Configuration must be provided.")

    input_size = config['input_size']
    output_size = config['output_size']
    hidden_layers = config['hidden_layer']
    hidden_size = config['hidden_size']

    # 生成层名称
    layer_names = generate_layer_names(input_size, output_size, hidden_layers, hidden_size)

    # 封装模型参数结构
    model_data = {
        "BpNeuralNetworkModelPara": {
            "ModelParam": {
                "InputSize": input_size,
                "OutputSize": output_size,
                "HiddenLayer": hidden_layers,
                "HiddenSize": hidden_size
            },

            "ModelWeightLayer_Input": [],
            **{name: [] for name in layer_names["weight_layers"][1:-1]},
            "ModelWeightLayer_Output": [],
            #
            "ModelBiasLayer_Input": [],
            **{name: [] for name in layer_names["bias_layers"][1:-1]},
            "ModelBiasLayer_Output": []
        },
        "#Metrics": {
            "MSE": metrics.get('MSE'),
        }
    }

    # 遍历模型的所有参数并填充模型权重和偏置
    for name, param in model.named_parameters():
        param_data = param.data.numpy().tolist()  # 转换为可序列化格式
        # print(f"Parameter name: {name}, Parameter value: {param.data.numpy()}")

        if 'bias' in name:
            if '0' in name:
                model_data["BpNeuralNetworkModelPara"]["ModelBiasLayer_Input"] = param_data
            elif 'out' in name:
                model_data["BpNeuralNetworkModelPara"]["ModelBiasLayer_Output"] = param_data
            else:
                # 获取层索引
                layer_index = name.split('.')[1]
                if layer_index == '1':
                    model_data["BpNeuralNetworkModelPara"]["ModelBiasLayer_0"] = param_data
                else:
                    model_data["BpNeuralNetworkModelPara"][
                        f"ModelBiasLayer_{int(layer_index) - 1}"] = param_data
        else:
            if '0' in name:
                model_data["BpNeuralNetworkModelPara"]["ModelWeightLayer_Input"] = param_data
            elif 'out' in name:
                model_data["BpNeuralNetworkModelPara"]["ModelWeightLayer_Output"] = param_data
            else:
                # 获取层索引
                layer_index = name.split('.')[1]
                if layer_index == '1':
                    model_data["BpNeuralNetworkModelPara"]["ModelWeightLayer_0"] = param_data
                else:
                    model_data["BpNeuralNetworkModelPara"][f"ModelWeightLayer_{int(layer_index) - 1}"] = param_data

    # 写入 JSON 文件
    with open(model_path, 'w') as json_file:
        json.dump(model_data, json_file, indent=4)

    print(f"Model saved to {model_path}")
