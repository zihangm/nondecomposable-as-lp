import numpy as np
import cifar_input as utils

def load_data(dataset='C10+', is_tune=False, is_crop_filp=False):

    with open('stl10_data/train_X.bin') as f:
        raw = np.fromfile(f, dtype=np.uint8, count=-1)
        raw = np.reshape(raw, (-1, 3, 96, 96))
        raw = np.transpose(raw, (0,3,2,1))
        X_train_raw = raw
		
    with open('stl10_data/train_y.bin') as f:
        raw = np.fromfile(f, dtype=np.uint8, count=-1)
        y_train = raw - 1 # class labels are originally in 1-10 format. Convert them to 0-9 format

    with open('stl10_data/test_X.bin') as f:
        raw = np.fromfile(f, dtype=np.uint8, count=-1)
        raw = np.reshape(raw, (-1, 3, 96, 96))
        raw = np.transpose(raw, (0,3,2,1))
        X_test_raw = raw

    with open('stl10_data/test_y.bin') as f:
        raw = np.fromfile(f, dtype=np.uint8, count=-1)
        y_test = raw - 1 
        
    # use the below line only when online setting
    if is_crop_filp:
        train_data = utils.padding(X_train_raw)
        X_train_raw = utils.random_crop_and_flip(train_data, padding_size=2).copy()
    else:
        X_train_raw = X_train_raw
        
     # per image standarizartion
    test_data = utils.per_image_standardization(X_test_raw.astype(np.float32)).copy()
    train_data = utils.per_image_standardization(X_train_raw.astype(np.float32)).copy()
    
    
    if is_tune:
        test_data = train_data[:4000]
        test_labels = y_train[:4000]
        train_data = train_data[4000:]
        train_labels = y_train[4000:]
    
    print ('Loading dataset: [%s], is_tune: [%s], is_preprocessed: [%s], is_crop_filp:[%s]'%(dataset, is_tune, 'True', str(is_crop_filp)))
    print ('Train_data: {}, Test_data: {}'.format(train_data.shape, test_data.shape))
 
    train_labels = y_train.astype(np.float32)
    test_labels = y_test.astype(np.float32)
    return (train_data, train_labels), (test_data, test_labels)    

#(X,Y), (test_X, testY) = load_data()