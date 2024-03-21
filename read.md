详细步骤见[复现日志](https://github.com/Wu-Yuanfei/LC/tree/main/%E5%A4%8D%E7%8E%B0%E6%97%A5%E5%BF%97)
## 配置环境
```conda env create -f environment.yaml -n new_env_name```，安装期间可能出现如下报错
```
  Preparing metadata (setup.py): started
  Preparing metadata (setup.py): finished with status 'error'

Pip subprocess error:
  error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [6 lines of output]
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/tmp/pip-install-2eobx_jr/flash-attn_35a12ec663d84bdc9a0efefeb15a4a65/setup.py", line 9, in <module>
          from packaging.version import parse, Version
      ModuleNotFoundError: No module named 'packaging'
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.

failed

CondaEnvException: Pip failed

```
原因在于安装安装flash-attn==2.3.6库时，因为缺少'packaging'模块，导致安装失败。期间尝试过许多办法都无法正常安装flash-attn==2.3.6库，暂时在<font color=red>environment.yaml</font>文件中删除flash-attn==2.3.6。继续运行```conda env create -f environment.yaml -n new_env_name```成功创建环境。后续代码运行出现问题再按照[方法](https://juejin.cn/post/7277802797470580771)安装flash-attn库.

## 复现LC_extraction.py
#### 代码简介
这段代码是一个用于处理和分析车辆轨迹数据的Python脚本。主要功能包括：

1. 通过解析参数，指定输入文件路径和设置参数，如是否可视化数据、绘制bounding boxes等。
2. 遍历60个不同的时间间隔文件，对每个文件中的车辆进行特定条件的筛选和数据提取。
3. 根据车辆的历史轨迹数据和相关信息，判断车辆的行驶方向（左变道、右变道、直行）。
4. 对符合条件的车辆进行数据提取和处理，包括提取车辆的轨迹数据、速度、加速度等信息，并保存到相应的列表中。
5. 最终将左转车辆、右转车辆和直行车辆的提取数据分别保存到对应的文件中，使用pickle模块进行序列化保存。

整体来说，这段代码的作用是对车辆的轨迹数据进行分析和处理，根据行驶方向将数据分类保存。
#### 复现操作
主要复现操作有：
将highD数据集data放置到**lc_llm-master**目录之下，文件具体格式见下  
lc_llm-master：
* create_llm_data:
* data:
* finetune:
* process_highD:

  * data_management:
  * visualization:
  * LC_extraction.py


#### 修改部分LC_extraction.py代码
在第79行处将原本代码：
```
            created_arguments["input_path"] = "/mnt/data_disk/mpeng/highD/data/{}_tracks.csv".format(no_str)
            created_arguments["input_static_path"] = "/mnt/data_disk/mpeng/highD/data/{}_tracksMeta.csv".format(no_str)
            created_arguments["input_meta_path"] = "/mnt/data_disk/mpeng/highD/data/{}_recordingMeta.csv".format(no_str)
            created_arguments["pickle_path"] = "/mnt/data_disk/mpeng/highD/pickle/{}.pickle".format(no_str)
            created_arguments["background_image"] = "/mnt/data_disk/mpeng/highD/data/{}_highway.png".format(no_str)
            created_arguments["output_left"] = "./output_data/{}/{}s".format(data_type, time_splip+1)
            created_arguments["output_right"] = "./output_data/{}/{}s".format(data_type, time_splip+1)
            created_arguments["output_straight"] = "./output_data/{}/straight".format(data_type)

```
修改为：
```
            created_arguments["input_path"] = "data/{}_tracks.csv".format(no_str)
            created_arguments["input_static_path"] = "data/{}_tracksMeta.csv".format(no_str)
            created_arguments["input_meta_path"] = "data/{}_recordingMeta.csv".format(no_str)
            created_arguments["pickle_path"] = "data/{}.pickle".format(no_str)
            created_arguments["background_image"] = "data/{}_highway.png".format(no_str)
            created_arguments["output_left"] = "./output_data/{}/{}s".format(data_type, time_splip+1)
            created_arguments["output_right"] = "./output_data/{}/{}s".format(data_type, time_splip+1)
            created_arguments["output_straight"] = "./output_data/{}/straight".format(data_type)

```
后续成功运行会输出pkl文件，文件具体格式如下：

lc_llm-master：
* create_llm_data:
* data:
* finetune:
* output_data:(xx:01-50)
  * train:
    * 1s:
        * left_extracted_data_xx.pkl
        * right_extracted_data_xx.pkl
    * 2s:
    * 3s:
    * 4s:
    * 5s:
    * 6s:
    * straight:
        * straight_extracted_data_xx.pkl
  * val:(xx:51-60)
    * 1s:
        * left_extracted_data_xx.pkl
        * right_extracted_data_xx.pkl
    * 2s:
    * 3s:
    * 4s:
    * 5s:
    * 6s:
    * straight:
        * straight_extracted_data_xx.pkl
* process_highD:

  * data_management:
  * visualization:
  * LC_extraction.py


## 复现create_data.py
安装ndjson库，```pip install ndjson```
#### 修改路径操作
line 44 修改
```
        left_file = f"./output_data/{datatype}/{timpesplit}/left_extracted_data_{num_str}.pkl"
        right_file = f"./output_data/{datatype}/{timpesplit}/right_extracted_data_{num_str}.pkl"
        straight_file = f"./output_data/{datatype}/straight/straight_extracted_data_{num_str}.pkl"
```
为
```
        left_file = f"../process_highD/output_data/{datatype}/{timpesplit}/left_extracted_data_{num_str}.pkl"
        right_file = f"../process_highD/output_data/{datatype}/{timpesplit}/right_extracted_data_{num_str}.pkl"
        straight_file = f"../process_highD/output_data/{datatype}/straight/straight_extracted_data_{num_str}.pkl"
```

line 65 修改
```
pickle.dump(datasets, open(f"./output_data/{datatype}/{timpesplit}/extracted_data.pkl", 'wb'))
```
为
```
    pickle.dump(datasets, open(f"../process_highD/output_data/{datatype}/{timpesplit}/extracted_data.pkl", 'wb'))
```

line 81 修改
```
        if is_train:
            datatype = "train"
            train_path = f"./output_data/train/{timesplit}/extracted_data.pkl" 
            if not os.path.exists(train_path):
                datasets = merge_data(datatype, timesplit)
            else:
                datasets = pickle.load(open(train_path, 'rb'))
        else:
            datatype = "val" 
            val_path = f"./output_data/val/{timesplit}/extracted_data.pkl"  
            if not os.path.exists(val_path):
                datasets = merge_data(datatype, timesplit)
            else:
                datasets = pickle.load(open(val_path, 'rb'))
```
为
```
        if is_train:
            datatype = "train"
            train_path = f"../process_highD/output_data/train/{timesplit}/extracted_data.pkl"
            if not os.path.exists(train_path):
                datasets = merge_data(datatype, timesplit)
            else:
                datasets = pickle.load(open(train_path, 'rb'))
        else:
            datatype = "val" 
            val_path = f"../process_highD/output_data/val/{timesplit}/extracted_data.pkl"
            if not os.path.exists(val_path):
                datasets = merge_data(datatype, timesplit)
            else:
                datasets = pickle.load(open(val_path, 'rb'))
```

line 143 修改
```
        if time_int > 1:
            pre_timesplit = str(time_int - 1) + "s"
            new_train_messages = []
            files = open(f"./llm_data/{pre_timesplit}/{datatype}_{prompt_type}.json", 'r', encoding='utf-8')  ## for chatgpt
            for file in files:
                new_train_messages.append(json.loads(file))
            new_train_messages.extend(train_messages)
            train_messages = new_train_messages

            new_llama_train_messages = []
            files = open(f"./{pre_timesplit}/llama_{datatype}_{prompt_type}.json", 'r', encoding='utf-8')  ## for llama
            for file in files:
                new_llama_train_messages = json.loads(file)
            new_llama_train_messages.extend(llama_train_messages.copy())   
            llama_train_messages = new_llama_train_messages

        print(len(train_messages))
        print(len(llama_train_messages))

        with open(f"./llm_data/{timesplit}/{datatype}_{prompt_type}.json", "w") as f:
            ndjson.dump(train_messages, f)

        with open(f"./llm_data/{timesplit}/llama_{datatype}_{prompt_type}.json", "w") as f:
            json.dump(llama_train_messages, f)

```
为
```
        if time_int > 1:
            pre_timesplit = str(time_int - 1) + "s"
            new_train_messages = []
            files = open(f"./llm_data/{pre_timesplit}/{datatype}_{prompt_type}.json", 'r', encoding='utf-8')  ## for chatgpt
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

        print(len(train_messages))
        print(len(llama_train_messages))

        with open(f"./llm_data/{timesplit}/{datatype}_{prompt_type}.json", "w") as f:
            ndjson.dump(train_messages, f)

        with open(f"./llm_data/{timesplit}/llama_{datatype}_{prompt_type}.json", "w") as f:
            json.dump(llama_train_messages, f)
        print(f"finish {timesplit}/{datatype}_{prompt_type}.json")
```

#### 内存爆炸
运行时可能出现异常终止，查看系统资源使用率面板可发现出现了内存爆炸，这是因为
```
        left_data = pickle.load(open(left_file, 'rb'))
        right_data = pickle.load(open(right_file, 'rb'))
        straight_data = pickle.load(open(straight_file, 'rb'))
        print("hi", i)
        for i in range(len(left_data['pieces'])):
            left.append({"scene": left_data['scene'], "pieces": left_data['pieces'][i]})
        for i in range(len(right_data['pieces'])):
            right.append({"scene": right_data['scene'], "pieces": right_data['pieces'][i]})
        for i in range(len(straight_data['pieces'])):
            straight.append({"scene": straight_data['scene'], "pieces": straight_data['pieces'][i]})

```
代码两层循环，将处理好的数据全部加载到了left、right、straight上，导致内存不足，尝试修改代码，分批次处理数据。

#### 原代码主体框架
1. 主函数调用create_llm_data()函数
2. create_llm_data()函数：   
    1. 设置变量
    2. 遍历output_data/train(val)/中1-5s文件夹中的数据
    3. 设置llm_data_path，如果不存在，则建一个路径
    4. 判断是否是“train”，设置路径（根据is_train作出相应更改），如果没有对应路径的pkl文件则调用merge_data(),反之则使用pickle.load()下载相应的pkl文件
    5. merge_data():
        1. 变量设置，根据datatype选择类型并修改相应的参数
        2. 遍历1-50(训练集)/51-60(验证集)的文件(左变道、右变道、直行)，将其中的数据按照分类分别存入left、right、straight
        3. 按照num_limit分别随机在left、right、straight中提取相应数量的left_sample、right_sample、straight_sample
        4. 最终将其添加到datasets中
    6. 文本数据编码，编码成适合gpt-3.5-turbo的格式
    7. 设置变量
    8. 遍历datasets
        1. 根据每个dataset调用generate_user_message_intention_traj_2()、generate_assistant_message_intention_traj()、system_message_intention_traj
        2. 输出消息
        3. 计算不同类型token数
        4. 将system_message、user_message、assitant_message添加到train_message和llama_train中，最终添加到train_messages、llama_train_messages
     9. 如果time_int>1:(即从第2s数据集开始，第1s数据不做处理)
        1. pre_timesplit = str(time_int - 1) + "s" eg.:2s->1s,3s->2s,1s->1s
        2. 打开./llm_data/{pre_timesplit}/{datatype}_{prompt_type}.json，即打开上次运行结束的json文件
        3. 遍历全部文件，将每次遍历的文件通过json.load()添加到new_train_messages
        4. 将train_messages中元素添加到另一个列表new_train_messages的尾部
        5. 将new_train_messages赋值给train_messages
     10. 将train_messages添加到./llm_data/{timesplit}/{datatype}_{prompt_type}.json(llama类似操作)

#### 修改后代码主体框架
修改代码见[修改后代码](https://github.com/Wu-Yuanfei/LC/blob/main/%E4%BF%AE%E6%94%B9%E4%BB%A3%E7%A0%81/test.py)
1. 主函数调用create_llm_data()函数
2. create_llm_data()函数：   
    1. 设置变量
    2. 遍历output_data/train(val)/中1-5s文件夹中的数据
    3. 设置llm_data_path，如果不存在，则建一个路径
    4. 判断是否是“train”，设置路径（根据is_train作出相应更改），如果没有对应路径的pkl文件则调用merge_data(),反之则使用pickle.load()下载相应的pkl文件
    5. 遍历begin_index到end_index-1(训练集：1-50，验证集：51-60)的文件
        1. 设置train_path
        2. 如inferenceatasets
            1. 根据每个dataset调用generate_user_message_intention_traj_2()、generate_assistant_message_intention_traj()、system_message_intention_traj
            2. 输出消息
            3. 计算不同类型token数
            4. 构建消息字典：将system_message、user_message、assitant_message添加到train_message和llama_train中，最终添加到train_messages、llama_train_messages
    6. 如果time_int>1:(即从第2s数据集开始，第1s数据不做处理)
        1. pre_timesplit = str(time_int - 1) + "s" eg.:2s->1s,3s->2s,1s->1s
        2. 打开./llm_data/{pre_timesplit}/{datatype}_{prompt_type}.json，即打开上次运行结束的json文件
        3. 遍历全部文件，将每次遍历的文件通过json.load()添加到new_train_messages
        4.  将train_messages中元素添加到另一个列表new_train_messages的尾部
        5.  将new_train_messages赋值给train_messages
    7.  将train_messages添加到./llm_data/{timesplit}/{datatype}_{prompt_type}.json(llama类似操作)

## 复现finetune
在[huggingface网站](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)上下载好Llama-2-7b-chat-hf模型文件(或者通过在服务器上拷贝Llama-2-7b-chat-hf模型文件)，修改inetune_lora.sh部分参数：
* suffix_str：jso文件部分文件名
* localhost：GPU,单个GPU设置为0
* model_name_or_path：Llama-2-7b-chat-hf模型文件路径
* train_files：训练集路径
* validation_files：验证集路径
  
激活llm环境，运行```./finetune_lora.sh```文件，首次运行可能报错，显示缺少c++编译器,安装GCC即可。登录[Weights & Biases网站](https://wandb.ai/site)，注册创建好项目后复制API key在终端输入```wandb login```,输入API key，再次运行finetune_lora文件，成功运行。


## 复现inference
修改inference.py文件部分参数

* parser.add_argument("--base_model_path", type=str, default="/home/yuanfei/hf"):Llama-2-7b-chat-hf模型文件

* parser.add_argument("--val_data_path", type=str, default="/home/yuanfei/PycharmProjects/lc_llm/lc_llm-master/create_llm_data/llm_data/4s/llama_val_surrounding_thinking_2.json"):指定验证数据的路径

* parser.add_argument("--new_model_path", type=str, default="/home/yuanfei/PycharmProjects/lc_llm/lc_llm-master/finetune/outputs/highD/intention_traj/Llama-2-7B-chat_ep1_2_surrounding_thinking_2")：微调后新模型的路径(finetune_lora)

* parser.add_argument("--reponse_dir", type=str, default="./reponse/highD/finetune/test.pkl")：存放输出文件的路径
  
修改batch_size=4.(原本批次为32,可能会导致GPU(单卡3090,24G显存)显存不足，将batch_size调小)
```
for results in tqdm(prediction_pipe(pipe_dataset, batch_size=4), total=len(pipe_dataset)):
```
