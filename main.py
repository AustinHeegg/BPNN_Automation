import re
from src.baseHJC import *
# from src.get_scanning import *
from src.process_files import *
from src.bpnn import *
from src.save_configs import *

def main():
    # 从JSON配置文件读取配置
    try:
        with open('bpnn_config', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found.  Please create the file.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in config.json. Please check the file.")

    # 获取配置
    # base_dir = config['base_dir']  # 设置数据的基础目录
    excel_path = 'C:/Users/h00018885/Desktop/data'  # 设置你的 Excel 文件路径
    files = os.listdir(excel_path)

    pattern = re.compile(r'(\d+)_rawdata_generate')
    x_list = []
    for filename in files:
        match = pattern.match(filename)
        if match:
            x = int(match.group(1))
            x_list.append(x)
    x_list = sorted(x_list)
    # tooling_numbers = read_tooling_info(excel_file)
    # unique_tooling_numbers = set(tooling_numbers)
    # print("Unique Tooling Numbers:", unique_tooling_numbers)

    select_files = config['select_files_or_not']
    rowSkip = config['rowSkip']
    selected_input_columns = config['selected_input_columns']
    selected_output_columns = config['selected_output_columns']
    selected_input_columns_final = config['selected_input_columns_final']
    selected_output_columns_final = config['selected_output_columns_final']
    input_size = config['input_size']
    output_size = config['output_size']
    hidden_layer = config['hidden_layer']
    hidden_nodes = config['hidden_nodes']
    activation_functions = config['activation_function']
    criterion_type = config['criterion_type']
    optimizer_type = config['optimizer_type']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    full_scale = config['full_scale']
    test_size = config['test_size']
    random_state = config['random_state']

    # for tooling_number in unique_tooling_numbers:
    #     print(f"Processing tooling number: {tooling_number}")
    #
    #     scanning_path = get_scanning_path(tooling_number, base_dir)
    #     if not scanning_path:
    #         print(f'No Scanning path found for tooling number {tooling_number}.')
    #         continue
    # 读取所有数据文件并分成training和test数据集
    for x in x_list:
        print(f"===================== Start training {x}_rawdata_generate!! =====================")
        data_files = getFileList(excel_path, ".csv", fileFullPath=True)
        if select_files == 1:
            selected_files = user_select_files(data_files)
        else:
            current_file = os.path.join(excel_path, f"{x}_rawdata_generate.csv")
            selected_files = [current_file]

        train_loader, test_loader, X_test, y_test = prepare_data(selected_files, batch_size, test_size, random_state,
                                                             selected_input_columns, selected_output_columns,
                                                             selected_input_columns_final, selected_output_columns_final,
                                                             rowSkip)

        # 初始化模型
        model, criterion, optimizer = BPNN.initialize_model(
            input_size,
            hidden_layer,
            hidden_nodes,
            output_size,
            learning_rate,
            activation_functions,
            criterion_type,
            optimizer_type
        )

        # 训练模型
        model.train()
        losses = BPNN.train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            num_epochs,
            full_scale,
            test_loader
        )

        # 绘制损失图像
        BPNN.loss_plot(losses)

        # 测试模型
        mse = BPNN.test(model, X_test, y_test)

        # 保存参数
        # mapping_file = "./工位安装映射.json"
        # mapping_dict = load_mapping(mapping_file)
        # organization = mapping_dict.get(tooling_number)
        model_path = f"./bpnn/bpnn_lens_{x}.json"
        metrics = {
            'MSE': mse,
        }
        save_model_to_json(model, metrics, model_path, config)  # 调用保存函数


if __name__ == '__main__':
    main()
