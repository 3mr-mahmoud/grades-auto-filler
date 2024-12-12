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
    


def processCellsOCR(df, symbolsModel):
    rows = df.shape[0]
    cols = df.shape[1]
    for i in range(1, rows):
        for j in range(cols):
            cell = df.iloc[i, j]
            top, bottom, left, right = 15, 9, 15, 10
            height, width = cell.shape[:2]
            cell = cell[top:height-bottom, left:width-right]

            if(j == 3 or j == 0):
                text = pytesseract.image_to_string(cell, config=ocr_config, lang='eng')
                text = sanitizeText(text)
                if(j == 3 and len(text) > 0):
                    text = text[0]
                df.iloc[i, j] = text
                show_images([cell])
                print(f"at j = {j}",text)
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


