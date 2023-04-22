import os.path

from transformers import AutoConfig,TrainingArguments,Trainer,EvalPrediction,ViTForImageClassification
from transformers import BertForTokenClassification,RobertaForTokenClassification,AlbertForTokenClassification,ElectraForTokenClassification,\
    ViTForImageClassification,SwinForImageClassification,DeiTModel, ConvNextForImageClassification
from model import DTCAModel
import torch
from utils.MyDataSet import MyDataSet,MyDataSet1,MyDataSet2
from utils.metrics import cal_f1
from typing import Callable, Dict
import numpy as np
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int,default=4, nargs='?',help='display an integer')
parser.add_argument('--output_result_file', type=str,default="./result.txt",nargs='?', help='display an integer')
parser.add_argument('--output_dir', type=str,default="./results",nargs='?', help='display an integer')
parser.add_argument('--lr', type=float, default=2e-5,nargs='?', help='display an integer')
parser.add_argument('--epochs', type=int, default=20,nargs='?', help='display an integer')
parser.add_argument('--alpha', type=float, default=0.6,nargs='?', help='display an integer')
parser.add_argument('--beta', type=float, default=0.6,nargs='?', help='display an integer')
parser.add_argument('--text_model_name',type=str,default="roberta",nargs='?')
parser.add_argument('--image_model_name',type=str,default="vit",nargs='?')
parser.add_argument('--random_seed', type=int, default=2022,nargs='?')
args = parser.parse_args()
alpha = args.alpha
beta = args.beta
batch_size = args.batch_size
output_dir = args.output_dir
lr = args.lr
epochs = args.epochs
text_model_name = args.text_model_name
image_model_name = args.image_model_name
output_result_file = args.output_result_file
random_seed = args.random_seed


def set_random_seed(random_seed):
    """Set random seed"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True

# set_random_seed(random_seed)
# def get_cross_prob(p_word_ids,p_image_aspect):
#     p_aspect_l = []
#     p_word_ids_len = len(p_word_ids(0))
#     for i,aspect in enumerate(p_image_aspect):
#         aspect_len = len(aspect)
#         if aspect_len == 0:
#             avg_prob = 1 / p_word_ids_len
#             p_aspect_l.append([avg_prob] * p_word_ids_len)
#         else:
#             i_aspect_l = []
#             aspect_d = dict()
#             for aspect_i in aspect:
#                 start_pos,end_pos = aspect_i[0].split("-")
#                 start_pos = int(start_pos)
#                 end_pos = int(end_pos)
#                 avg_prob = aspect_i[1] / (end_pos-start_pos)
#                 for aspect_idx in range(start_pos,end_pos):
#                     aspect_d[aspect_idx] = avg_prob
#             for word_id in p_word_ids(i):
#                 if word_id is None:
#                     i_aspect_l.append(0)
#                 else:
#                     if aspect_d.get(word_id) is not None:
#                         i_aspect_l.append(aspect_d[word_id])
#                     else:
#                         i_aspect_l.append(0)
#             p_aspect_l.append(i_aspect_l)
#     p_aspect_l = torch.Tensor(p_aspect_l)
#     return p_aspect_l
#
# def get_aspect_label(p_word_ids,p_image_aspect):
#     p_aspect_l = []
#     p_word_ids_len = len(p_word_ids(0))
#     for i, aspect in enumerate(p_image_aspect):
#         aspect_len = len(aspect)
#         if aspect_len == 0:
#             p_aspect_l.append([0] * 60)
#         else:
#             i_aspect_l = []
#             aspect_d = dict()
#             for aspect_i in aspect:
#                 start_pos, end_pos = aspect_i[0].split("-")
#                 start_pos = int(start_pos)
#                 end_pos = int(end_pos)
#                 aspect_d[start_pos] = aspect_i[1] + 2
#                 # for aspect_idx in range(start_pos+1, end_pos+1):
#                 #     aspect_d[aspect_idx] = 1
#             pre_word_id = None
#             for word_id in p_word_ids(i):
#                 if word_id is None:
#                     i_aspect_l.append(-100)
#                 else:
#                     if aspect_d.get(word_id) is not None:
#                         if pre_word_id == word_id:
#                             i_aspect_l.append(-100)
#                         else:
#                             i_aspect_l.append(aspect_d[word_id])
#                     else:
#                         if pre_word_id == word_id:
#                             i_aspect_l.append(-100)
#                         else:
#                             i_aspect_l.append(0)
#                 pre_word_id = word_id
#             p_aspect_l.append(i_aspect_l)
#     p_aspect_l = torch.tensor(p_aspect_l)
#     return p_aspect_l


train_text_inputs = torch.load("./datasets/embedding/train/inputs.pt")
train_image_inputs = torch.load("./datasets/embedding/train/image_inputs.pt")
# train_image_aspect = torch.load("./datasets/embedding/train/image_aspect.pt")
# train_image_sentiment = torch.load("./datasets/embedding/train/image_sentiment.pt")
train_word_ids = train_text_inputs.word_ids
# train_text_inputs["image_labels"] = train_image_sentiment
# train_text_inputs["cross_labels"] = get_aspect_label(train_word_ids,train_image_aspect)
train_pairs = train_text_inputs["pairs"]
train_text_inputs.pop("pairs")
train_dataset  = MyDataSet1(dataset_type="train",text_inputs=train_text_inputs,image_inputs=train_image_inputs)

dev_text_inputs = torch.load("./datasets/embedding/dev/inputs.pt")
dev_image_inputs = torch.load("./datasets/embedding/dev/image_inputs.pt")
# dev_image_aspect = torch.load("./datasets/embedding/dev/image_aspect.pt")
# dev_image_sentiment = torch.load("./datasets/embedding/dev/image_sentiment.pt")
dev_word_ids = dev_text_inputs.word_ids
# dev_text_inputs["image_labels"] = dev_image_sentiment
# dev_text_inputs["cross_labels"] = get_aspect_label(dev_word_ids,dev_image_aspect)
dev_pairs = dev_text_inputs["pairs"]
dev_text_inputs.pop("pairs")
dev_dataset  = MyDataSet1(dataset_type="dev",text_inputs=dev_text_inputs,image_inputs=dev_image_inputs)

test_text_inputs = torch.load("./datasets/embedding/test/inputs.pt")
test_image_inputs = torch.load("./datasets/embedding/test/image_inputs.pt")
# test_image_aspect = torch.load("./datasets/embedding/test/image_aspect.pt")
# test_image_sentiment = torch.load("./datasets/embedding/test/image_sentiment.pt")
test_word_ids = test_text_inputs.word_ids
# test_text_inputs["image_labels"] = test_image_sentiment
# test_text_inputs["cross_labels"] = get_aspect_label(test_word_ids,test_image_aspect)
test_pairs = test_text_inputs["pairs"]
test_text_inputs.pop("pairs")
test_dataset  = MyDataSet1(dataset_type="test",text_inputs=test_text_inputs,image_inputs=test_image_inputs)



if text_model_name == 'bert':
    model_path1 = './models/bert-base-uncased'
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = BertForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'roberta':
    model_path1 = "./models/roberta-base"
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = RobertaForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'albert':
    model_path1 = "./models/albert-base-v2"
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = AlbertForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'electra':
    model_path1 = './models/electra-base-discriminator'
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = AlbertForTokenClassification.from_pretrained(model_path1).state_dict()
else:
    os.error("出错了")
    exit()

if image_model_name == 'vit':
    model_path2 = "./models/vit-base-patch16-224-in21k"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = ViTForImageClassification.from_pretrained(model_path2).state_dict()
elif image_model_name == 'swin':
    model_path2 = "./models/swin-tiny-patch4-window7-224"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = SwinForImageClassification.from_pretrained(model_path2).state_dict()
elif image_model_name == 'deit':
    model_path2 = "./models/deit-base-patch16-224"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = DeiTModel.from_pretrained(model_path2).state_dict()
elif image_model_name == 'convnext':
    model_path2 = './models/convnext-tiny-224'
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = ConvNextForImageClassification.from_pretrained(model_path2).state_dict()
else:
    os.error("出错了")
    exit()


vb_model = DTCAModel(config1,config2,text_num_labels=5,text_model_name=text_model_name,image_model_name=image_model_name,alpha=alpha,beta=beta)
vb_model_dict = vb_model.state_dict()

for k,v in image_pretrained_dict.items():
    if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
        vb_model_dict[k] = v
for k,v in text_pretrained_dict.items():
    if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
        vb_model_dict[k] = v
vb_model.load_state_dict(vb_model_dict)

best_metric = dict()
text_best_metric = dict()


def predict(p_dataset, p_inputs, p_pairs):
    outputs = trainer.predict(p_dataset)
    pred_labels = np.argmax(outputs.predictions[0], -1)
    return cal_f1(pred_labels,p_inputs,p_pairs)


def build_compute_metrics_fn(text_inputs,pairs) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        text_logits, cross_logits = p.predictions
        text_pred_labels = np.argmax(text_logits,-1)
        pred_labels = np.argmax(cross_logits,-1)
        precision, recall, f1 = cal_f1(pred_labels,text_inputs,pairs)
        text_precision, text_recall, text_f1 = cal_f1(text_pred_labels, text_inputs, pairs)
        if best_metric.get("f1") is not None:
            if f1 > best_metric["f1"]:
                best_metric["f1"] = f1
                best_metric["precision"] = precision
                best_metric["recall"] = recall
                with open("my_model_result.txt", "w", encoding="utf-8") as f:
                    # for pred_pair in pred_pair_list:
                    #     f.write(str(pred_pair) + '\n')
                    f.write(str(pred_labels.tolist())+ '\n')
        else:
            best_metric["f1"] = f1
            best_metric["precision"] = precision
            best_metric["recall"] = recall
            with open("my_model_result.txt", "w", encoding="utf-8") as f:
                # for pred_pair in pred_pair_list:
                #     f.write(str(pred_pair) + '\n')
                f.write(str(pred_labels.tolist())+ '\n')
        if text_best_metric.get("f1") is not None:
            if text_f1 > text_best_metric["f1"]:
                text_best_metric["f1"] = text_f1
                text_best_metric["precision"] = text_precision
                text_best_metric["recall"] = text_recall
        else:
            text_best_metric["f1"] = text_f1
            text_best_metric["precision"] = text_precision
            text_best_metric["recall"] = text_recall
        return {"precision": precision,"recall":recall, "f1": f1,"text_precision": text_precision,"text_recall":text_recall, "text_f1": text_f1}
    return compute_metrics_fn


training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_steps=10000,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
)


trainer = Trainer(
    model=vb_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=build_compute_metrics_fn(text_inputs=test_text_inputs,pairs=test_pairs),
    # tokenizer=tokenizer,
    # data_collator=data_collator,
)
trainer.train()
# output = trainer.predict(test_dataset=test_dataset)
# _, cross_logits = output.predictions
# pred_labels = np.argmax(cross_logits,-1)
# precision, recall, f1, pred_pair_list = cal_f1(pred_labels,test_text_inputs,test_pairs,is_result=True)
# exit()
with open(output_result_file,"a",encoding="utf-8") as f:
    model_para = dict()
    model_para["text_model"] = text_model_name
    model_para["image_model"] = image_model_name
    model_para["batch_size"] = batch_size
    model_para["alpha"] = alpha
    model_para["beta"] = beta
    f.write("参数:"+str(model_para) + "\n")
    f.write("multi: "+ str(best_metric)+"\n")
    f.write("text: "+ str(text_best_metric)+"\n")
    f.write("\n")

