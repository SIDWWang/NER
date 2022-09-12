import os
import json
import logging
import numpy as np


class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config = config

    def process(self):
        """
        process train and test data
        """
        for file_name in self.config.files:
            self.preprocess(file_name)

    def preprocess(self, mode):
        """
        params:
            words：将json文件每一行中的文本分离出来，存储为words列表
            labels：标记文本对应的标签，存储为labels
        examples:
            words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
            labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
        """
        input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        word_list = []
        label_list = []
        with open(input_dir, 'r', encoding='utf-8') as f:
            # 先读取到内存中，然后逐行处理
            for line in f.readlines():
                # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                json_line = json.loads(line.strip())
                # text = json_line['text']
                # words = list(text)  # 自动将句子按字符分开
                # # 如果没有label，则返回None
                # label_entities = json_line.get('label', None)  # 参照下面的例子, 该项对应 label 之后的内容
                # labels = ['O'] * len(words)  # [len(words) 个 'O'] 都初始化为 `O`

                # if label_entities is not None:
                #     for key, value in label_entities.items():  # key 对应 name 和 company, value 对应后面存储内容
                #         for sub_name, sub_index in value.items():  # sub_name 对应 叶老桂等, sub_value 对应后面的索引
                #             for start_index, end_index in sub_index:  # 对应列表中的两个数,是标签开始和结束的位置
                #                 assert ''.join(words[start_index:end_index + 1]) == sub_name
                #                 if start_index == end_index:  # 单个字作为索引
                #                     labels[start_index] = 'S-' + key
                #                     else:
                #                         labels[start_index] = 'B-' + key  # 开头
                #                         labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)  # 中间的字

                text = json_line['text']
                words = list(text)
                # 如果没有label，则返回None
                label_entities = json_line.get('label', None)
                labels = ['O'] * len(words)

                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                word_list.append(words)
                label_list.append(labels)
                # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("--------{} data process DONE!--------".format(mode))

if __name__ == "__main__":
    import config
    Pro = Processor(config)
    Pro.process()