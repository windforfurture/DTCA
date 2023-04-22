from utils.generate_embedding import PreProcess
import os

data_types = ['train','dev','test']
text_route = './datasets/twitter2015'
image_route = './datasets/images/twitter2015_images'
text_model='./models/bert-base-uncased'
image_model='./models/vit-base-patch16-224-in21k'
text_type = '.txt'
output_route = './datasets/embedding'

#
# for data_type in data_types:
#     image = []
#     image_sentiment_l = []
#     image_aspect_l = []
#     preProcess = PreProcess(text_route,image_route,data_type,text_model,image_model,text_type)
#     data_type_dir = os.path.join(output_route, data_type)
#     if not os.path.exists(data_type_dir):
#         os.makedirs(data_type_dir)
#     sentence_l,image_l,label_l,pair_l = preProcess.get_joint_dataset()
#     n = len(sentence_l)
#     a_num = 0
#     pos = 0
#     neu = 0
#     neg = 0
#     ma = 0
#     ms = 0
#     mean_total = 0
#     max_len = 0
#
#     for i in range(n):
#         ss=set()
#         a_num += len(pair_l[i])
#         mean_total += len(sentence_l[i])
#         max_len = max(len(sentence_l[i]), max_len)
#         if len(pair_l[i]) > 1:
#             ma += 1
#         for pair in pair_l[i]:
#             ss.add(pair[1])
#             if pair[1] == 0:
#                 neg += 1
#             elif pair[1] == 1:
#                 neu += 1
#             else:
#                 pos += 1
#         if len(ss)>1:
#             ms += 1
#     print(data_type,n,a_num, pos,neu,neg,ma,ms, mean_total // n,max_len)
#

for data_type in data_types:
    image = []
    image_sentiment_l = []
    image_aspect_l = []
    preProcess = PreProcess(text_route,image_route,data_type,text_model,image_model,text_type)
    data_type_dir = os.path.join(output_route, data_type)
    if not os.path.exists(data_type_dir):
        os.makedirs(data_type_dir)
    sentence_l,image_l,label_l,pair_l = preProcess.get_joint_dataset()
    n = len(sentence_l)



    for i in range(n):
        ss=set()
        pos = 0
        neu = 0
        neg = 0
        ma = 0
        ms = 0
        for pair in pair_l[i]:
            ss.add(pair[1])
        if 0 in ss and 1 in ss and 2 in ss:
            print(data_type," ".join(sentence_l[i]),image_l[i])