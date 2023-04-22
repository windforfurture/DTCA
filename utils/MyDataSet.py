from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, dataset_type, embeddings, labels):
        """
        dataset_type: ['train', 'dev', 'test']
        """

        # self.inputs = inputs
        # self.inputs["labels"] = labels[:,:60]
        self.sample_list = list()
        self.dataset_type = dataset_type
        for i,embedding in enumerate(embeddings):
            self.sample_list.append((embedding,labels[i]))

    def __getitem__(self, index):
        inputs_embeds,labels = self.sample_list[index]
        return  {"inputs_embeds":inputs_embeds, "labels":labels}
        # d = dict()
        # for key in self.inputs.keys():
        #     d[key] = self.inputs[key][index]
        # return d


    def __len__(self):
        return len(self.sample_list)
        # return len(self.inputs["input_ids"])

class MyDataSet1(Dataset):
    def __init__(self, dataset_type, text_inputs, image_inputs):
        """
        dataset_type: ['train', 'dev', 'test']
        """
        self.inputs = text_inputs
        self.inputs["pixel_values"] = image_inputs['pixel_values']

    def __getitem__(self, index):
        d = dict()
        for key in self.inputs.keys():
            d[key] = self.inputs[key][index]
        return d


    def __len__(self):
        return len(self.inputs["input_ids"])


class MyDataSet2(Dataset):
    def __init__(self, inputs):
        """
        dataset_type: ['train', 'dev', 'test']
        """
        self.inputs = inputs

    def __getitem__(self, index):
        d = dict()
        for key in self.inputs.keys():
            d[key] = self.inputs[key][index]
        return d


    def __len__(self):
        return len(self.inputs["input_ids"])
