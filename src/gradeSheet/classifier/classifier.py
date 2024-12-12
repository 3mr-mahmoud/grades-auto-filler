from gradeSheet.classifier.train import train
from gradeSheet.classifier.predict import predictGeneral
from utils.paper_extraction import *
from utils.commonfunctions import *
from gradeSheet.extractCells import extract_cells
import cv2
import pandas as pd

import pandas as pd

def createTable(image, digitsModel, symbolsModel):
    df, rows, cols = extract_cells(image)
    for i in range(1, rows):
        for j in range(3, cols):
            cell = df.iloc[i, j]
            top, bottom, left, right = 5, 5, 5, 5
            height, width = cell.shape[:2]

            cell = cell[top:height-bottom, left:width-right]

            if cell is not None:
                if(j==3):
                    predection = predictGeneral(cell, digitsModel)
                    number = ord(predection) - ord('a') + 1 
                    df.iloc[i, j] = number
                else:
                    show_images([cell])
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
                            print(predection)
                        else:
                            df.iloc[i, j] = number
                            print(predection)

    print(df)
    return df


