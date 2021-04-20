import os
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    (t_rows, t_cols, _ ) = np.shape(T)

    heatmap = np.empty((n_rows, n_cols))

    T_normed = T/np.linalg.norm(T)

    t_cols_rad = np.math.floor(t_cols/2)
    t_rows_rad = np.math.floor(t_rows/2)

    I_extended = np.zeros((n_rows + 2*t_rows_rad, n_cols + 2*t_cols_rad, n_channels))
    I_extended[t_rows_rad:n_rows+t_rows_rad, t_cols_rad:n_cols+t_cols_rad, :] = I
    
    for i in range(n_cols):
        for j in range(n_rows):
            I_sub = I_extended[j:j+2*t_rows_rad+1, i:i+2*t_cols_rad+1, :]
            I_sub_normed = I_sub/np.linalg.norm(I_sub)
            heatmap[j, i] = np.sum(np.multiply(I_sub_normed, T_normed))

    '''
    END YOUR CODE
    '''

    return heatmap

def predict_boxes_old(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    thresh = 0.95
    (h_rows, h_cols) = np.shape(heatmap)
    
    extended_heatmap = np.hstack((np.zeros((h_rows, 1)), heatmap, (np.zeros((h_rows, 1)))))
    binary = 1*(extended_heatmap>thresh)
    edges = np.diff(binary)

    reduced_edges = edges[:, 1:h_cols+1]
    buffer = 10
    left_edges = np.argwhere(reduced_edges==1)
    print(np.sum(reduced_edges==1))
    right_edges = np.argwhere(reduced_edges==-1)
    print(left_edges, '\n', right_edges)
    assert len(left_edges) == len(right_edges)

    min_size = 10
    max_size = 400
    while len(left_edges)!=0:
        current_box = np.hstack((left_edges[0], right_edges[0]))
        
        left_edges = np.delete(left_edges, 0, axis=0)
        right_edges = np.delete(right_edges, 0, axis=0)
        remove_index = []
        
        for i, l in enumerate(left_edges):
            r = right_edges[i]
            if (l[0]-current_box[2]==1) and (np.mean(current_box[[1, 3]])-(l[1]+r[1])/2<buffer):
                if (0<current_box[1]-l[1]<buffer): 
                    current_box[1] = l[1]
                if  (0<r[1]-current_box[1]<buffer):
                    current_box[3] = r[1]
                current_box[2] = l[0]

                remove_index.append(i)
        remove_index.reverse()
        for s in remove_index:
            left_edges = np.delete(left_edges, s, axis=0)
            right_edges = np.delete(right_edges, s, axis=0)

        lst = current_box.tolist()
        pixels = (lst[2]-lst[0]+1)*(lst[3]-lst[1]+1)
        if  max_size >= pixels >= min_size:
            score = np.power(np.prod(heatmap[lst[0]:lst[2], lst[1]:lst[3]]), 1/pixels)
            output.append(lst+[score])
        current_box = []
    '''
    END YOUR CODE
    '''

    return output

def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    thresh = 0.90
    binary_mask = heatmap > thresh

    _ , mx = np.shape(binary_mask)

    x_mid=0
    while x_mid<mx: 
        if np.sum(binary_mask[:, x_mid])==0:
            x_mid+=1
        else: 
            y_mid = np.mean(np.nonzero(binary_mask[:, x_mid]))
            coors = [np.int(x_mid-kx/2), np.int(y_mid-ky/2), \
                np.int(x_mid+kx/2), np.int(y_mid)]
            pixels = (coors[2]-coors[0]+1)*(coors[3]-coors[1]+1)
            score = np.power(np.prod(heatmap[coors[0]:coors[2], coors[1]:coors[3]]), 4/pixels)
            output.append(coors+[score])
            x_mid+=kx

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I, plot=False):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    if np.mean(I[0:240, :, :])<40:
        T = night_ker
        print('n')
    elif np.mean(I[0:240, :, 0])<125:
        T = cloudy_ker
        print('c')
    else:
        T = day_ker
        print('d')

    heatmap = compute_convolution(I, T)
    output = predict_boxes(heatmap)

    if plot:
        img = Image.fromarray(I, 'RGB')
        draw = ImageDraw.Draw(img)
        for j in output:
            draw.rectangle(j[0:4])
        img.show()
    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# kernels
day_arr = np.asarray(Image.open(r'..\data\RedLights2011_Medium\RL-318.jpg'))
day_ker = day_arr[211:228, 301:308, :]
day_x, day_y, _ = np.shape(day_ker)

night_arr = np.asarray(Image.open(r'..\data\RedLights2011_Medium\RL-248.jpg'))
night_ker = night_arr[134:171, 500:517, :]
night_x, night_y, _ = np.shape(night_ker)

 
cloudy_arr = np.asarray(Image.open(r'..\data\RedLights2011_Medium\RL-047.jpg'))
cloudy_ker = cloudy_arr[230:267, 320:337, :]
cloudy_x, cloudy_y, _ = np.shape(cloudy_ker)

kx, ky, kz = [np.int(np.mean([day_y, night_y, cloudy_y])), \
    np.int(np.mean([day_x, night_x, cloudy_x])), 3]

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
