from gradeSheet.classifier.predict import predictGeneral
import matplotlib.pyplot as plt
import numpy as np
import cv2


def processCells(df, digitsModel, symbolsModel):
    rows = df.shape[0]
    cols = df.shape[1]
    code = ""
    print(rows, cols)
    
    for i in range(1, rows):
        cell = df.iloc[i, 0]
        
        # If cell is a file path (string), load the image
        if isinstance(cell, str):
            cell = cv2.imread(cell, cv2.IMREAD_GRAYSCALE)
            if cell is None:
                raise ValueError(f"Image at path {df.iloc[i, 0]} could not be loaded.")

        # Ensure cell is a valid NumPy array
        if not isinstance(cell, np.ndarray):
            raise ValueError(f"Cell at row {i} is not a valid image.")

        top, bottom, left, right = 25, 9, 15, 10
        height, width = cell.shape[:2]
        cell = cell[top:height-bottom, left:width-right]
        cell = cv2.threshold(cell, 100, 255, cv2.THRESH_BINARY)[1]
        cell = cv2.erode(cell, (3, 3), iterations=4)
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_AREA = 50  # Minimum area threshold for valid digits
        digit_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
        digit_contours = sorted(digit_contours, key=lambda c: cv2.boundingRect(c)[0])
        
        for j, contour in enumerate(digit_contours):
            x, y, w, h = cv2.boundingRect(contour)  # Bounding box
            digit = cell[y:y+h, x:x+w]             # Crop the digit region
            digit = cv2.GaussianBlur(digit, (3, 3), 0)
            digit = cv2.threshold(digit, 100, 255, cv2.THRESH_BINARY)[1]
            prediction = predictGeneral(digit, digitsModel)
            code += prediction
        
        # Optionally display the processed image and code
        
        # Update DataFrame
        df.iloc[i, 0] = code
        code = ""
        
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
                
                if(j==3):
                    predection = predictGeneral(cell, digitsModel)
                    #number = ord(predection) - ord('a') + 1 
                    number = ord(predection) - ord('0') 
                    df.iloc[i, j] = number
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


