import os.path

from transformers import AutoConfig,TrainingArguments,Trainer,EvalPrediction
from transformers import BertForTokenClassification,RobertaForTokenClassification,AlbertForTokenClassification, ViTForImageClassification,SwinForImageClassification,DeiTModel, ConvNextForImageClassification
from model import DTCAModel
import torch
from utils.MyDataSet import  MyDataSet2
from utils.metrics import cal_f1
from typing import Callable, Dict
import numpy as np
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str,default='2015', nargs='?',help='display a string')
parser.add_argument('--task_name', type=str,default='dualc', nargs='?',help='display a string')
parser.add_argument('--batch_size', type=int,default=4, nargs='?',help='display an integer')
parser.add_argument('--output_result_file', type=str,default="./result.txt",nargs='?', help='display a string')
parser.add_argument('--output_dir', type=str,default="./results",nargs='?', help='display a string')
parser.add_argument('--lr', type=float, default=2e-5,nargs='?', help='display a float')
parser.add_argument('--epochs', type=int, default=1,nargs='?', help='display an integer')
parser.add_argument('--alpha', type=float, default=0.6,nargs='?', help='display a float')
parser.add_argument('--beta', type=float, default=0.6,nargs='?', help='display a float')
parser.add_argument('--text_model_name',type=str,default="roberta",nargs='?')
parser.add_argument('--image_model_name',type=str,default="vit",nargs='?')
parser.add_argument('--random_seed', type=int, default=2022,nargs='?')

args = parser.parse_args()
dataset_type = args.dataset_type
task_name = args.task_name
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
                # with open("my_model_result.txt", "w", encoding="utf-8") as f:
                #     f.write(str(pred_labels.tolist())+ '\n')
        else:
            best_metric["f1"] = f1
            best_metric["precision"] = precision
            best_metric["recall"] = recall
            # with open("my_model_result.txt", "w", encoding="utf-8") as f:
            #     f.write(str(pred_labels.tolist())+ '\n')
        if text_best_metric.get("f1") is not None:
            if text_f1 > text_best_metric["f1"]:
                text_best_metric["f1"] = text_f1
                text_best_metric["precision"] = text_precision
                text_best_metric["recall"] = text_recall
        else:
            text_best_metric["f1"] = text_f1
            text_best_metric["precision"] = text_precision
            text_best_metric["recall"] = text_recall
        return {"precision": precision,"recall":recall, "f1": f1}
    return compute_metrics_fn



# set random seed
set_random_seed(random_seed)

data_input_file = os.path.join("datasets/finetune",task_name,dataset_type,"input.pt")
data_inputs = torch.load(data_input_file)
train_word_ids = data_inputs["train"].word_ids
train_pairs = data_inputs["train"]["pairs"]
data_inputs["train"].pop("pairs")
train_dataset  = MyDataSet2(inputs=data_inputs["train"])

dev_word_ids = data_inputs["dev"].word_ids
dev_pairs = data_inputs["dev"]["pairs"]
data_inputs["dev"].pop("pairs")
dev_dataset  = MyDataSet2(inputs=data_inputs["dev"])

test_word_ids = data_inputs["test"].word_ids
test_pairs = data_inputs["test"]["pairs"]
data_inputs["test"].pop("pairs")
test_dataset  = MyDataSet2(inputs=data_inputs["test"])


# text pretrained model selected
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

# image pretrained model selected
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

# init DTCAModel
vb_model = DTCAModel(config1,config2,text_num_labels=5,text_model_name=text_model_name,image_model_name=image_model_name,alpha=alpha,beta=beta)
vb_model_dict = vb_model.state_dict()

# load pretrained model weights
for k,v in image_pretrained_dict.items():
    if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
        vb_model_dict[k] = v
for k,v in text_pretrained_dict.items():
    if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
        vb_model_dict[k] = v
vb_model.load_state_dict(vb_model_dict)

best_metric = dict()
text_best_metric = dict()

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_steps=10000,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    label_names=["labels","cross_labels"]
)


trainer = Trainer(
    model=vb_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=build_compute_metrics_fn(text_inputs=data_inputs["test"],pairs=test_pairs),
)
trainer.train()

# output = trainer.predict(test_dataset=test_dataset)

# save results
with open(output_result_file,"a",encoding="utf-8") as f:
    model_para = dict()
    model_para["dataset_type"] = dataset_type
    model_para["text_model"] = text_model_name
    model_para["image_model"] = image_model_name
    model_para["batch_size"] = batch_size
    model_para["alpha"] = alpha
    model_para["beta"] = beta
    f.write("参数:"+str(model_para) + "\n")
    f.write("multi: "+ str(best_metric)+"\n")
    f.write("text: "+ str(text_best_metric)+"\n")
    f.write("\n")

