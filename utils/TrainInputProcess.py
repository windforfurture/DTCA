from transformers import AutoTokenizer
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
import os
import collections
from PIL import Image,ImageFile
import argparse

from transformers import AutoTokenizer,GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel, BertModel, RobertaModel, BlipForConditionalGeneration,BlipProcessor, CLIPProcessor, AutoProcessor


class TrainInputProcess:
    def __init__(self, 
                text_model, 
                text_model_type, 
                image_model,
                train_type,
                dataset_type=None,
                output_dir=None, 
                finetune_task=None,
                pretrain_task=None, 
                pretrain_output_dir=None,
                attention_type=None,
                image_gen_model_type=None,
                image_gen_text_model=None,
                data_text_dir=None, 
                data_image_dir=None, 
                pretrain_data_text_dir=None, 
                pretrain_data_image_dir=None):
        self.text_model = text_model
        self.text_model_type = text_model_type
        self.image_model = image_model
        self.train_type = train_type
        self.dataset_type = dataset_type
        self.attention_type = attention_type
        self.image_gen_model_type = image_gen_model_type
        self.image_gen_text_model = image_gen_text_model
        self.output_dir = output_dir
        self.finetune_task = finetune_task
        self.pretrain_task = pretrain_task
        self.pretrain_output_dir = pretrain_output_dir
        self.pretrain_data_text_dir = pretrain_data_text_dir
        self.pretrain_data_image_dir = pretrain_data_image_dir
        self.data_text_dir = data_text_dir
        self.data_image_dir = data_image_dir

        self.dataset_types = ['train','dev','test']
        self.text_type = '.txt'
        self.data_dict = dict()
        self.input = dict()
        self.pretrain_input= None
        if self.text_model_type == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model)
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        elif self.text_model_type == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model, add_prefix_space=True)

        self.image_process = ViTImageProcessor.from_pretrained(self.image_model)

    def generate_input(self):
        if self.train_type == 0:
            #  pre process
            self.get_text_dataset()
            if self.finetune_task == 'im2t':
                self.generate_im2t_input()
            elif self.finetune_task in ('clipc', 'dualc'):
                self.generate_dualc_input()
            else:
                os.error("No matched task！")
                exit()
        elif self.train_type == 1:
            if self.pretrain_task == "mlm":
                self.generate_mlm_input()
            else:
                os.error("No matched task！")
                exit()
    
    def generate_output_file(self, file_type=0):
        file_name = 'input.pt'
        # fine-tune input.pt
        if file_type == 0:
            inputs_dir = None
            if self.finetune_task == 'im2t':
                inputs_dir = os.path.join(self.output_dir,self.finetune_task, self.image_gen_model_type, self.attention_type, self.text_model_type, self.dataset_type)
            elif self.finetune_task in ('clipc','dualc'):
                inputs_dir = os.path.join(self.output_dir,self.finetune_task, self.dataset_type)
            if not os.path.exists(inputs_dir):
                os.makedirs(inputs_dir)
            inputs_path = os.path.join(inputs_dir, file_name)
            torch.save(self.input, inputs_path)
        # pretrain input.pt
        elif file_type == 1:
            inputs_dir = os.path.join(self.pretrain_output_dir, self.pretrain_task)
            if not os.path.exists(inputs_dir):
                os.mkdir(inputs_dir)
            pretrain_inputs_path = os.path.join(inputs_dir, file_name)
            torch.save(self.pretrain_input, pretrain_inputs_path)
            

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        mlm_probability=0.15
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def generate_mlm_input(self):
        sentence_l = []
        images = []
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        for text_no in range(2499,22892):
            if text_no in {3151,3910,5995}:
                continue
            text_file_path = os.path.join(self.pre_train_data_text_dir,str(text_no)+".txt")
            if os.path.exists(text_file_path):
                with open(text_file_path,'r',encoding="utf-8") as f:
                    sentence_l.append(f.readline().split())
            image_file_path = os.path.join(self.pre_train_data_text_dir,str(text_no)+".jpg")
            if os.path.exists(image_file_path):
                image = Image.open(image_file_path)
                image = image.convert('RGB')
                images.append(image)

        inputs = self.tokenizer(sentence_l, truncation=True, is_split_into_words=True,
                                     padding='max_length', max_length=60, return_tensors='pt')
        image_inputs=self.feature_extractor(images, return_tensors="pt")
        inputs["input_ids"],labels = self.torch_mask_tokens(inputs["input_ids"])
        inputs["pixel_values"] = image_inputs["pixel_values"]
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones(len(sentence_l), 197)), 1)
        inputs["labels"] = torch.cat((labels,torch.ones(len(sentence_l),197)*(-100)),1).long()
        self.pretrain_input = inputs

    # process fine-tune text
    # process_label: False-- 5 class; True-- 7 class.
    def get_text_dataset(self, process_label=False):
        for dataset_type in self.dataset_types:
            data_file_name = dataset_type + self.text_type
            text_path = os.path.join(self.data_text_dir, data_file_name)
            sentence_d = collections.defaultdict(list)
            sentence_l = []
            image_l = []
            label_l = []
            pair_l = []
            with open(text_path,'r',encoding="utf-8") as f:
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
                        if process_label:
                            s_label[vv[0]] = v_sentiment + 1
                        else:
                            s_label[vv[0]] = v_sentiment + 2
                        for i in range(vv[0]+1,vv[1] + 1):
                            if process_label:
                                s_label[i] = v_sentiment + 4
                            else:
                                s_label[i] = 1
                        s_pair.append((str(vv[0])+"-"+str(vv[1]),v_sentiment))
                    label_l.append(s_label)
                    pair_l.append(s_pair)
                self.data_dict[dataset_type] = (sentence_l, image_l, label_l, pair_l)

    def generate_im2t_input(self):
        # mac gpu
        device = torch.device("cuda")
        batch_size = 8
        # VisionEncoderDecoder
        if self.image_gen_model_type == 'ved':
            model = VisionEncoderDecoderModel.from_pretrained(self.image_gen_text_model).to(device)
            tokenizer = GPT2TokenizerFast.from_pretrained(self.image_gen_text_model)
            image_processor = ViTImageProcessor.from_pretrained(self.image_gen_text_model)
        elif self.image_gen_model_type == 'blip':
            model = BlipForConditionalGeneration.from_pretrained(self.image_gen_text_model).to(device)
            processor = BlipProcessor.from_pretrained(self.image_gen_text_model)

        for dataset_type in self.dataset_types:
            sentence_l, image_l, label_l, pair_l = self.data_dict[dataset_type]
            # load a fine-tuned image captioning model and corresponding tokenizer and image processor
            # let's perform inference on an image
            images = []
            for image_path in image_l:
                image_file_path = os.path.join(self.data_image_dir, image_path)
                image = Image.open(image_file_path)
                image= image.convert('RGB')
                images.append(image)
            images_n = len(images)
            generated_input_ids_l = None
            generated_sentence_l = []
            for i in range(0,images_n, batch_size):
                i_end = i+batch_size
                if i_end > images_n:
                    i_end = images_n
                print('\r[ %d / %d]' % (i, images_n), end='')

                generated_text_l = []
                if self.image_gen_model_type == 'ved':
                    pixel_values = image_processor(images[i:i_end], return_tensors="pt").pixel_values.to(device)
                    generated_ids = model.generate(pixel_values, max_new_tokens=60)
                    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    generated_text_l = [generated_text_i.split() for generated_text_i in generated_text]


                elif self.image_gen_model_type == 'blip':
                    pixel_values = processor(images[i:i_end], return_tensors="pt").pixel_values.to(device)
                    generated_ids = model.generate(pixel_values, max_new_tokens=30)
                    for generated_id in generated_ids:
                        generated_text = processor.decode(generated_id,skip_special_tokens=True)
                        generated_text_l.append(generated_text.split())
                if self.attention_type == 'cross':
                    generated_input_ids = self.tokenizer(generated_text_l, truncation=True, is_split_into_words=True,
                                                     padding='max_length', max_length=60, return_tensors='pt')[
                    "input_ids"]
                    if generated_input_ids_l is None:
                        generated_input_ids_l = generated_input_ids
                    else:
                        generated_input_ids_l = torch.cat((generated_input_ids_l,generated_input_ids),0)
                elif self.attention_type == 'self':
                    generated_sentence_l += generated_text_l
            tokenized_inputs = None
            new_sentence_l = []
            if self.attention_type == 'cross':
                tokenized_inputs = self.tokenizer(sentence_l, truncation=True, is_split_into_words=True,
                                         padding='max_length', max_length=60, return_tensors='pt')
            elif self.attention_type == 'self':
                for i,sentence in enumerate(sentence_l):
                    new_sentence_l.append(sentence + ['[sep]'] + generated_sentence_l[i])
                tokenized_inputs = self.tokenizer(new_sentence_l, truncation=True, is_split_into_words=True,
                                                  padding='max_length', max_length=90, return_tensors='pt')
            text_labels = []
            cross_labels = []
            for i, label in enumerate(label_l):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                label_ids = []
                cross_label_ids = []
                label_n = len(label)
                pre_word_idx = None
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None or word_idx >= label_n:
                        if self.attention_type == 'cross':
                            label_ids.append(-100)
                            cross_label_ids.append(0)
                        elif self.attention_type == 'self':
                            label_ids.append(0)
                    else:
                        if pre_word_idx != word_idx:
                            label_ids.append(label[word_idx])
                            if self.attention_type == 'cross':
                                cross_label_ids.append(label[word_idx])
                        else:
                            if self.attention_type == 'cross':
                                label_ids.append(-100)
                                cross_label_ids.append(0)
                            elif self.attention_type == 'self':
                                label_ids.append(0)
                    pre_word_idx = word_idx
                cross_labels.append(cross_label_ids)
                text_labels.append(label_ids)
            tokenized_inputs["labels"] = torch.tensor(text_labels)
            tokenized_inputs["pairs"] = pair_l
            if self.attention_type == 'cross':
                tokenized_inputs["cross_labels"] = torch.tensor(cross_labels)
                tokenized_inputs["generated_input_ids"] = generated_input_ids_l
            self.input[dataset_type] = tokenized_inputs

    def generate_dualc_input(self):
        for dataset_type in self.dataset_types:
            sentence_l, image_l, label_l, pair_l = self.data_dict[dataset_type]
            images = []

            for image_path in image_l:
                image_file_path = os.path.join(self.data_image_dir, image_path)
                image = Image.open(image_file_path)
                image = image.convert('RGB')
                images.append(image)

            processor = AutoProcessor.from_pretrained(self.image_model)
            if self.finetune_task == 'clipc':
                clip_tokenizer = AutoTokenizer.from_pretrained(self.image_model)
            pixel_values = processor(images=images,return_tensors="pt")["pixel_values"]
            new_sentence_l = []
            for sentence in sentence_l:
                new_sentence_l.append(" ".join(sentence))
            if self.finetune_task == 'clipc':
                clip_tokenized_inputs = clip_tokenizer(new_sentence_l, truncation=True, padding='max_length', max_length=60, return_tensors='pt')
            tokenized_inputs = self.tokenizer(sentence_l, truncation=True, is_split_into_words=True, padding='max_length', max_length=60, return_tensors='pt')
            text_labels = []
            cross_labels = []
            for i, label in enumerate(label_l):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                label_ids = []
                cross_label_ids = []
                label_n = len(label)
                pre_word_idx = None
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None or word_idx >= label_n:
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
                cross_labels.append(cross_label_ids)
                text_labels.append(label_ids)
            tokenized_inputs["labels"] = torch.tensor(text_labels)
            tokenized_inputs["pairs"] = pair_l
            tokenized_inputs["cross_labels"] = torch.tensor(cross_labels)
            tokenized_inputs["pixel_values"] = pixel_values
            if self.finetune_task == 'clipc':
                tokenized_inputs["clip_input_ids"] = clip_tokenized_inputs["input_ids"]
            self.input[dataset_type] = tokenized_inputs


def main():
    parser = argparse.ArgumentParser()
    # twitter 2015 or 2017
    parser.add_argument('--dataset_type', type=str, default='2015', nargs='?', help='display an string')
    # text model: roberta, bert, albert, electra
    parser.add_argument('--text_model_type', type=str, default='roberta', nargs='?', help='display an string')
    # image model: vit
    parser.add_argument('--image_model_type', type=str, default='vit', nargs='?', help='display an string')

    # train type: 0-finetune 1-pretrain
    parser.add_argument('--train_type', type=int, default=0, nargs='?', help='display an int')
    # dualc: two-stream co-attention;
    parser.add_argument('--finetune_task', type=str, default='dualc', nargs='?',help='display an string')

    # image captioning for MABSA
    parser.add_argument('--attention_type', type=str, default=None, nargs='?', help='display an string')
    parser.add_argument('--image_gen_model_type', type=str, default=None, nargs='?', help='display an string')
    parser.add_argument('--image_gen_text_model', type=str, default=None, nargs='?', help='display an string')

    # inputs output dir
    parser.add_argument('--output_dir', type=str, default='datasets/finetune', nargs='?', help='display an string')

    # used in pretraining tasks
    parser.add_argument('--pretrain_task', type=str, default='mlm', nargs='?')
    parser.add_argument('--pretrain_output_dir', type=str, default='datasets/pretrain', nargs='?', help='display an string')
    parser.add_argument('--pretrain_data_text_dir', type=str, default='datasets/MVSA/data', nargs='?', help='display an string')
    parser.add_argument('--pretrain_data_image_dir', type=str, default='datasets/MVSA/data', nargs='?', help='display an string')

    args = parser.parse_args()

    dataset_type = args.dataset_type
    text_model_type = args.text_model_type
    image_model_type = args.image_model_type

    train_type = args.train_type
    finetune_task = args.finetune_task

    attention_type = args.attention_type
    image_gen_model_type = args.image_gen_model_type
    image_gen_text_model = args.image_gen_text_model

    output_dir = args.output_dir

    pretrain_task = args.pretrain_task
    pretrain_output_dir = args.pretrain_output_dir
    pretrain_data_text_dir = args.pretrain_data_text_dir
    pretrain_data_image_dir = args.pretrain_data_image_dir


    if text_model_type == 'bert':
        text_model = 'models/bert-base-uncased'
    elif text_model_type == 'roberta':
        text_model = 'models/roberta-base'

    if image_model_type == 'vit':
        image_model = 'models/vit-base-patch16-224-in21k'


    if finetune_task == 'im2t':
        attention_type = 'cross'
        image_gen_model_type = 'ved'
        if image_gen_model_type == 'ved':
            image_gen_text_model = 'models/vit-gpt2-image-captioning'
        elif image_gen_model_type == 'blip':
            image_gen_text_model = 'models/blip-image-captioning-base'
    elif finetune_task == 'clipc':
        image_model = 'models/clip-vit-base-patch32'

    data_text_dir = None
    data_image_dir = None
    if dataset_type == '2015':
        data_text_dir = 'datasets/twitter2015'
        data_image_dir = 'datasets/images/twitter2015_images'
    elif dataset_type == '2017':
        data_text_dir = 'datasets/twitter2017'
        data_image_dir = 'datasets/images/twitter2017_images'


    trainInputProcess = TrainInputProcess(text_model,
                                  text_model_type,
                                  image_model,
                                  train_type,
                                  dataset_type,
                                  finetune_task=finetune_task,
                                  pretrain_task=pretrain_task,
                                  pretrain_output_dir=pretrain_output_dir,
                                  attention_type=attention_type,
                                  image_gen_model_type = image_gen_model_type,
                                  output_dir=output_dir,
                                  image_gen_text_model=image_gen_text_model,
                                  data_text_dir=data_text_dir,
                                  data_image_dir=data_image_dir,
                                  pretrain_data_text_dir=pretrain_data_text_dir,
                                  pretrain_data_image_dir=pretrain_data_image_dir)
    trainInputProcess.generate_input()
    trainInputProcess.generate_output_file()


if __name__ == '__main__':
    main()