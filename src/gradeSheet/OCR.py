import pytesseract

import cv2

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'

from utils.commonfunctions import show_images
from gradeSheet.classifier.predict import predictGeneral

def sanitizeText(text):
    text = text.strip()
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    return text

def cropCell(cell, top_ratio=0.1, bottom_ratio=0.95, left_ratio=0.01, right_ratio=0.95):
    height, width = cell.shape
    # Calculate crop coordinates
    left = int(left_ratio * width)
    right = int(right_ratio * width)
    top = int(top_ratio * height)
    bottom = int(bottom_ratio * height)
    result = cell[top:bottom, left:right]
    return result
    


def processCellsOCR(df, symbolsModel):
    rows = df.shape[0]
    cols = df.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    for i in range(1, rows):
        for j in range(cols):
            cell = df.iloc[i, j]
            

            if(j == 3 or j == 0):
                cell = cropCell(cell)
                text = pytesseract.image_to_string(cell, config=ocr_config, lang='eng')
                text = sanitizeText(text)
                if(j == 3 and len(text) > 0):
                    text = text[0]
                df.iloc[i, j] = text
                show_images([cell], ["Prediction: " + text])
            elif(j > 3):
                cell = cropCell(cell, left_ratio=0.3, right_ratio=0.8)
                cell = cv2.dilate(cell, kernel, iterations=2)
                cell = cv2.erode(cell, kernel, iterations=1)
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
                
                show_images([cell], ["Prediction: " + predection])
    return df


