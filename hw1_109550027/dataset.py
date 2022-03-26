import os
import cv2
import glob
def loadImages(dataPath):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    #raise NotImplementedError("To be implemented")
    dataset=[]
    files = glob.glob(os.path.join(dataPath+"/car","*"))
    for imgc in files:
        imc = cv2.imread(imgc,0)
        imc = cv2.resize(imc, (36, 16))
        tup = (imc,1)
        dataset.append(tup)
    files = glob.glob(os.path.join(dataPath+"/non-car","*"))
    for imgn in files:
        imn = cv2.imread(imgn,0)
        imn = cv2.resize(imn, (36, 16))
        tup = (imn,0)
        dataset.append(tup)
    # End your code (Part 1)
    
    return dataset
