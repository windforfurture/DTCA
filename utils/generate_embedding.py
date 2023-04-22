from transformers import AutoFeatureExtractor, AutoTokenizer
import torch
import os
import collections
from datasets import load_dataset
from PIL import Image

class PreProcess:
    def __init__(self,text_route: str, image_route:str, data_type: str,  text_model: str, image_model: str,text_type: str='.txt'):
        self.text_route = text_route
        self.image_route = image_route
        self.data_type = data_type
        self.text_type = text_type
        self.text_model = text_model
        self.image_model = image_model

    # process text
    def get_joint_dataset(self, i_label=False):
        data_file_name = self.data_type + self.text_type
        file_path = os.path.join(self.text_route, data_file_name)
        sentence_d = collections.defaultdict(list)
        sentence_l = []
        image_l = []
        label_l = []
        pair_l = []
        with open(file_path,'r',encoding="utf-8") as f:
            while True:
                text = f.readline().rstrip('\n').split()
                if text == []:
                    break
                aspect = f.readline().rstrip('\n').split()
                sentiment = f.readline().rstrip('\n')
                image_path = f.readline().rstrip('\n')
                start_pos = text.index("$T$")
                end_pos = start_pos + len(aspect) - 1
                text = text[:start_pos] + aspect + text[start_pos+1:]
                sentence_d[" ".join(text)].append((start_pos,end_pos,sentiment,image_path))
            for key,value in sentence_d.items():
                text = key.split()
                sentence_l.append(text)
                n_key =len(text)
                s_label = [0] * n_key
                s_pair = []
                image_l.append(value[0][3])
                for vv in value:
                    v_sentiment = int(vv[2]) + 1
                    if i_label:
                        s_label[vv[0]] = v_sentiment + 1
                    else:
                        s_label[vv[0]] = v_sentiment + 2
                    for i in range(vv[0]+1,vv[1] + 1):
                        if i_label:
                            s_label[i] = v_sentiment + 4
                        else:
                            s_label[i] = 1
                    s_pair.append((str(vv[0])+"-"+str(vv[1]),v_sentiment))
                label_l.append(s_label)
                pair_l.append(s_pair)
        return sentence_l, image_l, label_l, pair_l



    def generate_image_input(self, image_l):
        images = []
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.image_model)
        for image_path in image_l:
            image_file_path = os.path.join(image_route, image_path)
            image = Image.open(image_file_path)
            image= image.convert('RGB')
            images.append(image)
        inputs = feature_extractor(images, return_tensors="pt")
        return inputs

    def generate_text_input(self, sentence_l, label_l, pair_l):
        # tokenizer = AutoTokenizer.from_pretrained(self.text_model,add_prefix_space=True)
        tokenizer = AutoTokenizer.from_pretrained(self.text_model)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenized_inputs = tokenizer(sentence_l, truncation=True, is_split_into_words=True,
                                     padding='max_length',max_length=60, return_tensors='pt')
        tokenized_inputs["pairs"] = pair_l
        labels = []
        text_labels = []
        cross_labels = []
        image_labels = []
        max_label_length = 0
        for i, label in enumerate(label_l):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            label_ids = []
            cross_label_ids = []
            pre_word_idx = None
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                    cross_label_ids.append(0)
                else:
                    if pre_word_idx != word_idx:
                        label_ids.append(label[word_idx])
                        cross_label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                        cross_label_ids.append(0)
                pre_word_idx = word_idx
            sent_cnt = 0
            sent_sum = 0
            for pair in pair_l[i]:
                sent_cnt += 1
                sent_sum += pair[1]
            cross_labels.append(cross_label_ids)
            text_labels.append(label_ids)
        tokenized_inputs["labels"] = torch.tensor(text_labels)
        tokenized_inputs["cross_labels"]= torch.tensor(cross_labels)
        return tokenized_inputs

    # # process text
    # def generate_vilt_input(self, sentence_l, image_l, label_l, pair_l):
    #     processor = ViltProcessor.from_pretrained("models/vilt-b32-mlm")
    #     new_image_l = []
    #     for i, sentence in enumerate(sentence_l):
    #         image_path = os.path.join(image_route, image_l[i])
    #         image = Image.open(image_path)
    #         image = image.convert('RGB')
    #         new_image_l.append(image)
    #     encoding = processor(new_image_l, sentence_l, padding='max_length', max_length=40, is_split_into_words=True, truncation=True, return_tensors="pt")
    #
    #     new_label_l = []
    #     n = len(label_l)
    #     for i in range(n):
    #         word_ids = encoding.word_ids(batch_index=i)  # Map tokens to their respective word.
    #         label_ids = []
    #         pre_word_idx = None
    #         for word_idx in word_ids:  # Set the special tokens to -100.
    #             if word_idx is None:
    #                 label_ids.append(-100)
    #             else:
    #                 if pre_word_idx != word_idx:
    #                     label_ids.append(label_l[i][word_idx])
    #                 else:
    #                     label_ids.append(-100)
    #             pre_word_idx = word_idx
    #         new_label_l.append(label_ids)
    #
    #     encoding["pairs"] = pair_l
    #     encoding["labels"] = torch.tensor(new_label_l)
    #     return encoding




if __name__ =='__main__':
    data_types = ['train','dev','test']
    text_route = 'datasets/twitter2015'
    image_route = 'datasets/images/twitter2015_images'

    # text_route = 'datasets/twitter2017'
    # image_route = 'datasets/images/twitter2017_images'

    text_model='models/bert-base-uncased'
    # text_model = 'models/roberta-base'
    # text_model= 'models/albert-base-v2'
    # text_model = 'models/electra-base-discriminator'

    image_model='models/vit-base-patch16-224-in21k'
    # image_model = 'models/swin-tiny-patch4-window7-224'
    # image_model = 'models/deit-base-patch16-224'
    # image_model = 'models/convnext-tiny-224'
    text_type = '.txt'
    output_route = 'datasets/embedding'
    for data_type in data_types:
        preProcess = PreProcess(text_route,image_route,data_type,text_model,image_model,text_type)
        data_type_dir = os.path.join(output_route, data_type)
        if not os.path.exists(data_type_dir):
            os.makedirs(data_type_dir)
        sentence_l,image_l,label_l,pair_l = preProcess.get_joint_dataset(i_label=True)
        text_inputs = preProcess.generate_text_input(sentence_l,label_l,pair_l)
        torch.save(text_inputs, os.path.join(data_type_dir,"single_inputs.pt"))
        #
        # sentence_l, image_l, label_l, pair_l = preProcess.get_joint_dataset()
        # text_inputs = preProcess.generate_text_input(sentence_l, label_l, pair_l)
        # torch.save(text_inputs, os.path.join(data_type_dir, "inputs.pt"))

        # image_inputs = preProcess.generate_image_input(image_l)
        # torch.save(image_inputs,os.path.join(data_type_dir,"image_inputs.pt"))

        #

