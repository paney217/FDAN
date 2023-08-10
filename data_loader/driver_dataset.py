import torch
from torchvision import transforms
import torch.utils.data as data
import cv2
import pandas as pd
from PIL import Image


class DriverDataset(data.Dataset):
    def __init__(self, img_root, img_file, transform=None):
        self.root = img_root

        self.df = pd.read_csv(img_root + img_file, header=None)
        #print(len(self.df))
        
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.df.loc[index][0]
        class_id = self.df.loc[index][1]

        img_np = cv2.imread(self.root + img_path)
        #print(img_np.shape)
        img = Image.fromarray(img_np)
        if self.transform is not None:
            img = self.transform(img)

        class_id = torch.LongTensor([class_id])

        return img.squeeze(), class_id.squeeze()
                    #return img.squeeze(), class_id.squeeze()
    
    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_root = 'E:/PycharmProjects/2020/ddtest2/data_AUC/'
    train_datasets = DriverDataset(data_root, "Train_data_list1.csv", transform=data_transforms['train'])
    test_datasets = DriverDataset(data_root, "Test_data_list1.csv", transform=data_transforms['train'])

    print("train: {}, test: {}".format(len(train_datasets), len(test_datasets)))

    inputs, labels = train_datasets[10]
    print(inputs.shape, inputs.type(), labels.shape, labels.type())
