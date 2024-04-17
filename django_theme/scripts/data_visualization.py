import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def dcm_jpg(file, name):
    file = file.pixel_array.astype(float)
    scaled = (np.maximum(file,0)/file.max())*255
    result = np.uint8(scaled)
    result = Image.fromarray(result)
    result.save(name[:-3] + "jpg")
    os.remove(name)
    return
