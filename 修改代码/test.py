import pickle  # 用于序列化和反序列化Python对象
import os  # 用于处理文件和目录路径
import random  # 用于生成随机数
import ndjson  # 用于读写ndjson格式的文件
import json  # 用于读写json格式的文件
import tiktoken  # 未知模块，可能是自定义模块
from prompt_message import generate_traj_label, generate_user_message_intention_traj_2, \
    generate_assistant_message_intention_traj, system_message_intention_traj, generate_vector_data  # 导入自定义模块中的函数
from visualize import plot_waymo_trajectory_dataset, plot_highD_dataset  # 可视化相关的自定义模块


def format_text_1(system, user, assistant):
    """
    格式化文本输出
    """
    text = f"""<s>[INST] <<SYS>>"""  # 定义文本格式
    text += f"""{system}"""
    text += f"""<</SYS>>\n\n"""
    text += f"""{user} [/INST] {assistant} </s>\n"""
    return text


def merge_data(datatype, timpesplit, i):
    """
    合并数据集
    """
    time_int = int(timpesplit[0])
    left_num = 0
    right_num = 0
    straight_num = 0
    datasets = []
    left = []
    right = []
    straight = []
    # if datatype == "train":
    #     begin_index = 1
    #     end_index = 51
    #     num_limit = 12000
    # else:
    #     begin_index = 51
    #     end_index = 61
    #     num_limit = 2000
    # for i in range(begin_index, end_index):
    if i < 10:
        num_str = f"0{i}"
    else:
        num_str = str(i)
    # 文件路径
    left_file = f"../process_highD/output_data/{datatype}/{timpesplit}/left_extracted_data_{num_str}.pkl"
    right_file = f"../process_highD/output_data/{datatype}/{timpesplit}/right_extracted_data_{num_str}.pkl"
    straight_file = f"../process_highD/output_data/{datatype}/straight/straight_extracted_data_{num_str}.pkl"

    # 读取数据
    left_data = pickle.load(open(left_file, 'rb'))
    right_data = pickle.load(open(right_file, 'rb'))
    straight_data = pickle.load(open(straight_file, 'rb'))

    print("hi", i)

    # 将数据添加到对应列表中
    for i in range(len(left_data['pieces'])):
        left.append({"scene": left_data['scene'], "pieces": left_data['pieces'][i]})
    for i in range(len(right_data['pieces'])):
        right.append({"scene": right_data['scene'], "pieces": right_data['pieces'][i]})
    for i in range(len(straight_data['pieces'])):
        straight.append({"scene": straight_data['scene'], "pieces": straight_data['pieces'][i]})

    # 随机采样数据
    # left_sample = random.sample(left, num_limit)
    # right_sample = random.sample(right, num_limit)
    # straight_sample = random.sample(straight, num_limit)

    # 合并数据集
    # datasets.extend(left_sample)
    # datasets.extend(right_sample)
    # datasets.extend(straight_sample)
    datasets.extend(left)
    datasets.extend(right)
    datasets.extend(straight)

    # 存储合并后的数据集
    # pickle.dump(datasets, open(f"../process_highD/output_data/{datatype}/{timpesplit}/extracted_data.pkl", 'wb'))
    return datasets


def create_llm_data():
    is_train = True
    prompt_type = "surrounding_thinking_2"
    total_num = 18000
    for i in range(1, 5):
        timesplit = str(i) + "s"
        time_int = int(timesplit[0])
        datatype = ""

        llm_data_path = f"./llm_data/{timesplit}"
        if not os.path.exists(llm_data_path):
            os.makedirs(llm_data_path)
        if is_train:
            datatype = "train"
        else:
            datatype = "val"

        if datatype == "train":
            begin_index = 1
            end_index = 51
            # num_limit = 12000
            # num_limit = 240
        else:
            begin_index = 51
            end_index = 61
            # num_limit = 2000
            # num_limit = 40
        train_messages = []
        for i in range(begin_index, end_index):
            train_path = f"../process_highD/output_data/train/{timesplit}/extracted_data.pkl"
            if not os.path.exists(train_path):
                datasets = merge_data(datatype, timesplit, i)
            else:
                datasets = pickle.load(open(train_path, 'rb'))

            val_path = f"../process_highD/output_data/val/{timesplit}/extracted_data.pkl"
            if not os.path.exists(val_path):
                datasets = merge_data(datatype, timesplit, i)
            else:
                datasets = pickle.load(open(val_path, 'rb'))
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

            num_language_tokens = 0
            num_system_tokens = 0
            num_user_tokens = 0
            num_assistant_tokens = 0


            llama_train_messages = []
            num = 0
            for dataset in datasets:
                scene = dataset["scene"]
                sample_info = dataset["pieces"]
                user_message = generate_user_message_intention_traj_2(scene, sample_info)
                assitant_message = generate_assistant_message_intention_traj(sample_info)
                system_message = system_message_intention_traj
                # plot_highD_dataset(scene, sample_info, total_num)
                # total_num += 1

                print(num)
                print(system_message)
                print(user_message)
                print(assitant_message)

                # 计算不同类型的token数
                num_language_tokens += len(encoding.encode(system_message))
                num_system_tokens += len(encoding.encode(system_message))
                num_language_tokens += len(encoding.encode(user_message))
                num_user_tokens += len(encoding.encode(user_message))
                num_language_tokens += len(encoding.encode(assitant_message))
                num_assistant_tokens += len(encoding.encode(assitant_message))

                # 构建消息字典
                train_message = {"messages":
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assitant_message}
                    ]
                }
                train_messages.append(train_message)
                llama_train = format_text_1(system_message, user_message, assitant_message)
                llama_train_messages.append({"text": llama_train}.copy())
                num += 1

            # 打印token数量信息
            print("#### Cost Summarization ####")
            print(f"Number of system tokens: {num_system_tokens}")
            print(f"Number of user tokens: {num_user_tokens}")
            print(f"Number of assistant tokens: {num_assistant_tokens}")
            print(f"Number of total tokens: {num_language_tokens}")


        if time_int > 1:
            pre_timesplit = str(time_int - 1) + "s"
            # pre_timesplit = str(time_int) + "s"
            new_train_messages = []
            files = open(f"./llm_data/{pre_timesplit}/{datatype}_{prompt_type}.json", 'r',
                         encoding='utf-8')  ## for chatgpt
            for file in files:
                new_train_messages.append(json.loads(file))
            new_train_messages.extend(train_messages)
            train_messages = new_train_messages

            new_llama_train_messages = []
            files = open(f"./llm_data/{pre_timesplit}/llama_{datatype}_{prompt_type}.json", 'r', encoding='utf-8')  ## for llama
            for file in files:
                new_llama_train_messages = json.loads(file)
            new_llama_train_messages.extend(llama_train_messages.copy())
            llama_train_messages = new_llama_train_messages

        print('gpt:', len(train_messages))
        print('llama:', len(llama_train_messages))

        with open(f"./llm_data/{timesplit}/{datatype}_{prompt_type}.json", "w") as f:
            ndjson.dump(train_messages, f)

        with open(f"./llm_data/{timesplit}/llama_{datatype}_{prompt_type}.json", "w") as f:  # 写入JSON文件
            json.dump(llama_train_messages, f)
        print(f"finish {timesplit}/{datatype}_{prompt_type}.json")  # 输出完成信息

def create_vector_data():
    # 创建向量数据的函数
    vector_data_path = "./vector_data"  # 向量数据保存路径
    if not os.path.exists(vector_data_path):  # 如果路径不存在，则创建
        os.makedirs(vector_data_path)

    is_train = True  # 是否为训练集
    for i in range(1, 5):  # 遍历1到4
        timesplit = str(i) + "s"  # 按时间划分，如"1s"
        time_int = int(timesplit[0])  # 提取时间整数部分
        datatype = ""
        if is_train:  # 如果是训练集
            datatype = "train"
            train_path = f"./output_data/train/{timesplit}/extracted_data.pkl"
            if not os.path.exists(train_path):
                datasets = merge_data(datatype, timesplit)
            else:
                datasets = pickle.load(open(train_path, 'rb'))
        else:
            datatype = "val"  # 否则为验证集
            val_path = f"./output_data/val/{timesplit}/extracted_data.pkl"
            if not os.path.exists(val_path):
                datasets = merge_data(datatype, timesplit)
            else:
                datasets = pickle.load(open(val_path, 'rb'))

        vector_data = []  # 存储向量数据的列表
        num = 0  # 计数器
        for dataset in datasets:
            scene = dataset["scene"]
            sample_info = dataset["pieces"]

            vector_info = generate_vector_data(scene, sample_info)  # 生成向量数据
            # gt = generate_vector_label(sample_info)
            gt = generate_traj_label(sample_info)  # 生成轨迹标签
            vector_data.append({"vector_info": vector_info, "gt": gt}.copy())  # 将数据添加到列表中
            num += 1  # 计数器加一

        if time_int > 1:  # 如果时间大于1
            pre_timesplit = str(time_int - 1) + "s"  # 获取前一个时间段
            new_vector_data = []  # 存储新向量数据的列表
            files = open(f"./vector_data/{pre_timesplit}/{datatype}_traj.json", 'r',
                         encoding='utf-8')  # 读取JSON文件
            for file in files:  # 遍历文件
                new_vector_data = json.loads(file)  # 加载JSON数据
            new_vector_data.extend(vector_data.copy())  # 扩展新向量数据
            vector_data = new_vector_data  # 更新向量数据

        print(len(vector_data))  # 输出向量数据长度

        with open(f"./vector_data/{timesplit}/{datatype}_traj.json", "w") as f:  # 写入JSON文件
            json.dump(vector_data, f)  # 将向量数据存储到JSON文件中

if __name__ == "__main__":
    create_llm_data()  # 调用创建llama数据的函数
    # create_vector_data()  # 调用创建向量数据的函数
