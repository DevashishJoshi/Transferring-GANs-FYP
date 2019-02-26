from os import listdir
import numpy as np
import scipy.misc
import time
import pdb
from config import celeba_h, celeba_w, embeddings_file_name, embedding_size
from config import celeba_image_path as image_path

def load_embeddings():
    #l = os.listdir(config.embeddings_dir)
    embeddings = np.load(embeddings_file_name)
    return embeddings.item()

def make_generator(path, n_files, batch_size,image_size, IW = False, phase='train'):
    epoch_count = [1]
    #image_list_main = listdir(path)
    #image_list = []
    image_list = listdir(path + '/' + phase)        
    #image_list.extend([sub_class_path + '/' + i for i in sub_class_image])

    def get_epoch():
        images = np.zeros((batch_size, 3, image_size, image_size), dtype='int32')
        labels = np.zeros((batch_size, embedding_size), dtype='int32')
        #files = range(len(image_list))
        random_state = np.random.RandomState(epoch_count[0])
        #random_state.shuffle(files)
        embeddings = load_embeddings()
        epoch_count[0] += 1
        #random_state.shuffle(image_list)
        #image_list = [image_list[i] for i in files]
        for i, image_name in enumerate(image_list):
            image = scipy.misc.imread("{}".format(image_path + image_name))
            image_name_wo_ext = image_name.split('.')[0]
            label = embeddings[image_name_wo_ext]

            image = scipy.misc.imresize(image,(image_size,image_size))
            images[i % batch_size] = image.transpose(2,0,1)
            labels[i % batch_size] = label
            if i > 0 and i % batch_size == 0:
                yield (images,labels)
    
    return get_epoch
  
def load(batch_size, data_dir='/home/ishaan/data/imagenet64',image_size = 64, NUM_TRAIN = 7000):
    return make_generator(data_dir, NUM_TRAIN, batch_size,image_size, phase='train')
    
