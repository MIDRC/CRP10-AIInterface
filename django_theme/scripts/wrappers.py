import json
import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from PIL import Image
from django.conf import settings
from matplotlib import pyplot as plt
import seaborn as sn
#from scripts.displays import *
from scripts.data_visualization import *
#change to make each block appear individually
def display_activations(activations, cmap=None, save=False, directory='.',
                         data_format='channels_last', fig_size=(24, 24),
                        reshape_1d_layers=False):

    import matplotlib.pyplot as plt
    import math
    index = 0
    
    #here is where the images are being combined, can seperate them here
    for layer_name, acts in activations.items():
        print(layer_name, acts.shape, end=' ')
        if acts.shape[0] != 1:
            print('-> Skipped. First dimension is not 1.')
            continue

        print('')
        # channel first
        if data_format == 'channels_last':
            c = -1
        elif data_format == 'channels_first':
            c = 1
        else:
            raise Exception('Unknown data_format.')
        
        nrows = int(math.sqrt(acts.shape[c]) - 0.001) + 1  # best square fit for the given number
        ncols = int(math.ceil(acts.shape[c] / nrows))
        hmap = None
        if len(acts.shape) <= 2:
            """
            print('-> Skipped. 2D Activations.')
            continue
            """
            # no channel
            fig, axes = plt.subplots(1, 1, squeeze=False, figsize=fig_size)
            img = acts[0, :]
            img2 = np.reshape(img, _convert_1d_to_2d(img.shape[0])) if reshape_1d_layers else [img]
            hmap = axes.flat[0].imshow(img2, cmap=cmap)
            axes.flat[0].axis('off')
        else:
            fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=fig_size)
            for i in range(nrows * ncols):

                if i < acts.shape[c]:
                    if len(acts.shape) == 3:
                        if data_format == 'channels_last':
                            img = acts[0, :, i]
                            print(type(img))
                            plt.savefig(str(i) + 't.png', bbox_inches='tight')



                        elif data_format == 'channels_first':
                            img = acts[0, i, :]
                            
                        else:
                            raise Exception('Unknown data_format.')
                        hmap = axes.flat[i].imshow([img], cmap=cmap)
                    elif len(acts.shape) == 4:
                        if data_format == 'channels_last':
                            img = acts[0, :, :, i]
                        elif data_format == 'channels_first':
                            img = acts[0, i, :, :]
                        else:
                            raise Exception('Unknown data_format.')
                        hmap = axes.flat[i].imshow(img, cmap=cmap)
                axes.flat[i].axis('off')
        fig.suptitle(layer_name)
        fig.subplots_adjust(right=0.8)
        cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        if hmap is not None:
            fig.colorbar(hmap, cax=cbar)
        if save:
            if not os.path.exists(directory):
                os.makedirs(directory)
            output_filename = os.path.join(directory, '{0}_{1}.png'.format(index, layer_name.split('/')[0]))
            plt.savefig(output_filename, bbox_inches='tight')
            return output_filename
        else:
            plt.show()
        # pyplot figures require manual closing
        index += 1
        plt.close(fig)
        return
   

def display_heatmaps(activations, input_image, directory='.', save=False, fix=True, merge_filters=False):  # noqa: C901
    """
    Plot heatmaps of activations for all filters overlayed on the input image for each layer
    :param activations: dict mapping layers to corresponding activations with the shape
    (1, output height, output width, number of filters)
    :param input_image: numpy array, input image for the overlay, should contain floats in range 0-1
    :param directory: string - where to store the activations (if save is True)
    :param save: bool, if the plot should be saved
    :param fix: bool, if automated checks and fixes for incorrect images should be run
    :param merge_filters: bool, if one heatmap (with all the filters averaged together) should be produced
    for each layer instead of a heatmap for each filter
    :return: None
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    import math

    def __scale(arr):
        """
        Scale a numpy array to have values 0-1
        :param arr: numpy array, the array to be scaled
        :return: numpy array
        """
        scaled = arr * (1/(np.amax(arr) - np.amin(arr)))
        scaled = scaled - np.amin(scaled)
        return scaled

    data_format = K.image_data_format()
    if fix:
        # fixes common errors made when passing the image
        # I recommend the use of keras' load_img function passed to np.array to ensure
        # images are loaded in in the correct format
        # removes the batch size from the shape
        if len(input_image.shape) == 4:
            input_image = input_image.reshape(input_image.shape[1], input_image.shape[2], input_image.shape[3])
        # removes channels from the shape of grayscale images
        if len(input_image.shape) == 3 and input_image.shape[2] == 1:
            input_image = input_image.reshape(input_image.shape[0], input_image.shape[1])
        # converts a 0-255 image to be 0-1
        if np.amin(input_image) >= 0 and 1 < np.amax(input_image) <= 255:
            input_image /= 255.0

    index = 0
    for layer_name, acts in activations.items():
        print(layer_name, acts.shape, end=' ')
        if acts.shape[0] != 1:
            print('-> Skipped. First dimension is not 1.')
            continue
        if len(acts.shape) <= 2:
            print('-> Skipped. 2D Activations.')
            continue
        print('')

        if merge_filters:
            nrows = 1
            ncols = 1
        else:
            nrows = int(math.sqrt(acts.shape[-1]) - 0.001) + 1  # best square fit for the given number
            ncols = int(math.ceil(acts.shape[-1] / nrows))

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
        fig.suptitle(layer_name)

        # loops over each subplot
        for i in range(nrows * ncols):
            # Hide the x-y axes of the plot as we aren't showing a graph
            axes.flat[i].axis('off') if hasattr(axes, 'flat') else axes.axis('off')

            if merge_filters:
                if len(acts.shape) == 3:
                    img = acts[0, :, :]
                    # gets the activation of the ith layer
                    if data_format == 'channels_last':
                        # normalise the activations of each neuron so they all contribute to the average equally
                        for j in range(0, acts.shape[-1]):
                            img[:, j] = __scale(img[:, j])
                        img = np.sum(img, axis=1)
                    elif data_format == 'channels_first':
                        for j in range(0, acts.shape[-1]):
                            img[j, :] = __scale(img[j, :])
                        img = np.sum(img, axis=0)
                    else:
                        raise Exception('Unknown data_format.')
                elif len(acts.shape) == 4:
                    img = acts[0, :, :, :]
                    if data_format == 'channels_last':
                        for j in range(0, acts.shape[-1]):
                            img[:, :, j] = __scale(img[:, :, j])
                        img = np.sum(img, axis=2)
                    elif data_format == 'channels_first':
                        for j in range(0, acts.shape[-1]):
                            img[j, :, :] = __scale(img[j, :, :])
                        img = np.sum(img, axis=0)
                    else:
                        raise Exception('Unknown data_format.')
                else:
                    raise Exception('Expect a tensor of 3 or 4 dimensions.')
            else:
                # if have reached a subplot that doesn't have an activation associated with it
                if i >= acts.shape[-1]:
                    # if this was a break, the x-y axes wouldn't be hidden for the subsequent blank subplots
                    continue
                if len(acts.shape) == 3:
                    # gets the activation of the ith layer
                    if data_format == 'channels_last':
                        img = acts[0, :, i]
                    elif data_format == 'channels_first':
                        img = acts[0, i, :]
                    else:
                        raise Exception('Unknown data_format.')
                elif len(acts.shape) == 4:
                    if data_format == 'channels_last':
                        img = acts[0, :, :, i]
                    elif data_format == 'channels_first':
                        img = acts[0, i, :, :]
                    else:
                        raise Exception('Unknown data_format.')
                else:
                    raise Exception('Expect a tensor of 3 or 4 dimensions.')

            img = Image.fromarray(img)
            # resizes the overlay to be same dimensions of input_image
            img = img.resize((input_image.shape[1], input_image.shape[0]), Image.BILINEAR)
            img = np.array(img)
            if hasattr(axes, 'flat'):
                axes.flat[i].imshow(input_image)
                # overlay the activation at 70% transparency  onto the image with a heatmap colour scheme
                # Lowest activations are dark blue, highest are dark red, mid are green-yellow
                axes.flat[i].imshow(img, alpha=0.3, cmap='jet', interpolation='bilinear')
            else:
                axes.imshow(input_image)
                axes.imshow(img, alpha=0.3, cmap='jet', interpolation='bilinear')
        if save:
            if not os.path.exists(directory):
                os.makedirs(directory)
            output_filename = os.path.join(directory, '{0}_{1}.png'.format(index, layer_name.split('/')[0]))
            plt.savefig(output_filename, bbox_inches='tight')
            return output_filename
        else:
            plt.show()
        index += 1
        plt.close(fig)
        return


#prob move to a different "functions" file 
def dcm_jpg(file, name):
    file = file.pixel_array.astype(float)
    scaled = (np.maximum(file,0)/file.max())*255
    result = np.uint8(scaled)
    result = Image.fromarray(result)
    result.save(name[:-3] + "jpg")
    os.remove(name)
    return