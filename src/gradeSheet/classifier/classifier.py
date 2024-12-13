from gradeSheet.classifier.predict import predictGeneral
import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

def processCells(df, digitsModel, symbolsModel):
    rows = df.shape[0]
    cols = df.shape[1]
    for i in range(1, rows):
        for j in range(3, cols):
            cell = df.iloc[i, j]
            top, bottom, left, right = 15, 9, 15, 10
            height, width = cell.shape[:2]
            cell = cell[top:height-bottom, left:width-right]
            if cell is not None:
                right_crop = 100
                left_crop = 100
                cell = cell[:, left_crop:cell.shape[1]-right_crop]
                #cell = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_CUBIC)
                #cell = cv2.GaussianBlur(cell, (3, 3), 0)
                # cellX = cv2.Sobel(cell, cv2.CV_64F, 1, 0, ksize=3)
                # cellY = cv2.Sobel(cell, cv2.CV_64F, 0, 1, ksize=3)
                # cell = cv2.magnitude(cellX, cellY)
                # cell = cv2.convertScaleAbs(cell)
                #cell = cv2.threshold(cell, 100, 255, cv2.THRESH_BINARY)[1]
                #  
                if(j==3):
                    predection = predictGeneral(cell, digitsModel)
                    #number = ord(predection) - ord('a') + 1 
                    number = ord(predection) - ord('0') 
                    df.iloc[i, j] = number
                    show_images([cell],[predection])
                else:
                    predection = predictGeneral(cell, symbolsModel)
                    if(predection=="T"):
                        df.iloc[i, j] = 5
                    elif(predection=="S" or predection=="H1"):
                        df.iloc[i, j] = 0
                    elif(predection=="empty"):
                        df.iloc[i, j] = None
                    elif(predection=="Q"):
                        df.iloc[i, j] = "red"
                    else:
                        char = predection[0]
                        number = int(predection[1])
                        if(char=='H'):
                            df.iloc[i, j] = 5 - number
                        else:
                            df.iloc[i, j] = number
    return df


