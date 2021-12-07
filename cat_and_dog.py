
import numpy as np
import os
#import matplotlib.pyplot as plt 
from PIL import Image
import cifar_input
np.random.seed(0)

raw_dir = './cat_and_dog/%s/'
saved_data_dir = './cat_and_dog/cat_and_dog_%s_data.npy'
saved_label_dir = './cat_and_dog/cat_and_dog_%s_label.npy'

N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000

IMG_WIDTH = 50
IMG_HEIGHT = 50
IMG_DEPTH = 3

def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))
    
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = np.array([int(i) for i in label_list])

    return image_list, label_list


def load_data(dataset, is_tune=False, is_crop_filp=False):
    data_set = []
    label_set = []
    for dtype in ['train']: # test dat is missing labels
        print ('load %s dataset...'%dtype)
        saved_data_dir = './cat_and_dog/cat_and_dog_%s_data.npy'
        saved_label_dir = './cat_and_dog/cat_and_dog_%s_label.npy'
        if not os.path.exists(saved_data_dir%dtype):
            image_list, label_list = get_files(raw_dir%dtype)
            # read images 
            image_array = []
            for img_dir in image_list:
                img = Image.open(img_dir)
                resize_img = img.resize((IMG_HEIGHT, IMG_WIDTH))
                new_img = np.asarray(resize_img)
                # convert int8 to float 
                image_array.append(new_img.astype(np.float32))
            
            image_array = np.array(image_array)
            data_set.append(image_array)
            label_set.append(label_list)
            np.save('./cat_and_dog/cat_and_dog_%s_data.npy'%dtype, image_array)
            np.save('./cat_and_dog/cat_and_dog_%s_label.npy'%dtype, label_list)
            
        else:
            data = np.load(saved_data_dir%dtype)
            label = np.load(saved_label_dir%dtype)
            data_set.append(data)
            label_set.append(label)
    
    #Train
    if is_crop_filp:
        train_data = cifar_input.padding(data_set[0])
        train_data = random_crop_and_flip(train_data, padding_size=2)     
    else:
        train_data = data_set[0]
        
    train_label_tmp = label_set[0]
    train_data_tmp = cifar_input.per_image_standardization(train_data).copy()

    test_data = train_data_tmp[-5000:].copy()
    test_label = train_label_tmp[-5000:].copy()
    train_data = train_data_tmp[:-5000].copy()
    train_label = train_label_tmp[:-5000].copy()  
    
    if is_tune:
        train_data = train_data_tmp[:-6000].copy()
        train_label = train_label_tmp[:-6000].copy()
        test_data = train_data_tmp[-6000:-5000].copy()
        test_label = train_label_tmp[:-6000:-5000].copy()
    
    print ('Loading dataset: [Cat&Dog%d], is_tune: [%s], is_preprocessed: [%s], is_crop_filp:[%s]'%(dataset, is_tune, 'True', str(is_crop_filp)))
    print ('Train_data: {}, Test_data: {}'.format(train_data.shape, test_data.shape))
    return (train_data, train_label), (test_data, test_label)
    
def random_crop_and_flip(batch_data, padding_size=2):
    '''
    Ref: https://www.tensorflow.org/api_docs/python/tf/image/random_crop
    '''
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT, y_offset:y_offset+IMG_WIDTH, :]
        cropped_batch[i, ...] = cifar_input.random_flip_left_right(image=cropped_batch[i, ...], axis=1)
    return cropped_batch
