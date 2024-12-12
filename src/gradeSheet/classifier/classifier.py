from gradeSheet.classifier.train import train
from gradeSheet.classifier.predict import predictGeneral
from utils.paper_extraction import *
from utils.commonfunctions import *
from gradeSheet.extractCells import extract_cells
import cv2
import pandas as pd

digitsModel = train('../../../Dataset/Training Set/digits_dataset')
symbolsModel = train('../../../Dataset/Training Set/symbols_dataset')

def createTable(image, digitsModel, symbolsModel):
    cells, rows, cols = extract_cells(image)
    table = []
    for cell in cells:
        
        prediction = predictGeneral(cell, digitsModel)
        table.append(prediction)
    return table

