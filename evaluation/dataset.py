import os
import random

class ImageDataSet(object):

    def __init__(self, base_path):

        self.base_path = base_path
        self.label2idx = {"invoice": 0, "non_invoice": 1}
        self.idx2label = {value: key for key, value in self.label2idx.items()}

    def read_label_files(self, dataset_path):
        labels = list(self.label2idx.keys())
        image_dict = {}
        for label in labels:
            images_names = os.listdir(os.path.join(dataset_path, label))
            for image_name in images_names:
                image_path = os.path.join(os.path.join(dataset_path, label), image_name)
                image_dict[image_path] = int(self.label2idx[label])

        return list(image_dict.keys()), list(image_dict.values())

    def build_dataset(self, dataset_type='train'):

        if "test" is dataset_type:
            dataset_path = os.path.join(self.base_path, "test")
        elif "val" in dataset_type:
            dataset_path = os.path.join(self.base_path, "val")
        else:
            dataset_path = os.path.join(self.base_path, "train")

        return self.read_label_files(dataset_path)

