import pickle
import cv2
import numpy as np 
import os
import torch
from sklearn.model_selection import train_test_split
import torch

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
    
def get_dataset(path, has_class_directories=True):
    
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset

def get_image_paths(facedir):
    ## From Facenet Library
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def read_images(files):
    ims = []
    for file in files:
        image_rgb = cv2.imread(file)
        if image_rgb.shape[0]!=image_rgb.shape[1]:
            image_rgb = image_rgb[29:-29, 9:-9,::-1]
        else:
            cut = 1
            image_rgb = cv2.resize(image_rgb[cut:-cut,cut:-cut,:], (160,160))[:,:,::-1]
        ims.append(image_rgb)
    return np.array(ims)


with open("dataset_description.pkl","rb") as foo:
    dc = pickle.load(foo)

dataset_address = 'dataset/lfw2/lfw2'
dataset = get_dataset(dataset_address)
lfw_raw, label_list = get_image_paths_and_labels(dataset)

lfw_list = []
for im in dc['image_list']:
    lfw_list.append(os.path.join(dataset_address, im))
lfw_attributes = (dc['attributes']>0).astype(int)
lfw_labels = (lfw_attributes[:,0]>0).astype(int)
lfw_latent_vars = dc['latent_vars'] ##Latent representation of the image in the VAE trained on CelebA
lfw_data = ((read_images(lfw_list) - 127.5)/128, lfw_labels, lfw_latent_vars)
sex = lfw_attributes[:,0]
skin = lfw_attributes[:,3]
makeup = lfw_attributes[:,-13]
wavy_hair = lfw_attributes[:,26]
sens_multi = lfw_attributes[:,[3,38,40]]
dictionary = dc

x = lfw_data[0]
y = lfw_labels
s = wavy_hair

train_X, test_X, \
train_y, test_y, \
train_s, test_s, \
train_sm, test_sm = train_test_split(x, y, s, sens_multi, test_size=0.2, random_state=826)

train_X = torch.as_tensor(train_X, dtype=torch.float32).permute(0, 3, 1, 2)
test_X = torch.as_tensor(test_X, dtype=torch.float32).permute(0, 3, 1, 2)
train_y = torch.as_tensor(train_y, dtype=torch.float32)
test_y = torch.as_tensor(test_y, dtype=torch.float32)
train_s = torch.as_tensor(train_s, dtype=torch.float32)
test_s = torch.as_tensor(test_s, dtype=torch.float32)
train_sm = torch.as_tensor(train_sm, dtype=torch.float32)
test_sm = torch.as_tensor(test_sm, dtype=torch.float32)


input_data = {'x':train_X, 'y': train_y, 's': train_s, 'sm': train_sm}
target_data = {'x': test_X, 'y':test_y, 's':test_s, 'sm': test_sm}

os.makedirs("dataset/lfwa_w")

file = open("dataset/lfwa_w/train.pkl","wb")
pickle.dump(input_data,file)

file = open("dataset/lfwa_w/test.pkl","wb")
pickle.dump(target_data,file)