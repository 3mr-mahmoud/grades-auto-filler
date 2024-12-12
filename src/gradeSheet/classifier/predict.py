from train import train
from ..extractCells import extractCells
import numpy as np

digitsModel = train("../../../Dataset/Training Set/digits_dataset")
symbolsModel = train("../../../Dataset/Training Set/symbols_dataset")

def predict(image):
    cellImage, rows, cols = extractCells(image)
    res = [rows][cols]
    for row in range(rows):
        for col in range(3, cols):
            cell = cellImage[row][col]
            if cell is not None:
                # Predict digit and symbol
                digit_confidence = digitsModel.predict_proba(cell)  # If model supports predict_proba() for confidence scores
                symbol_confidence = symbolsModel.predict_proba(cell)

                # Get the predicted digit and symbol with the highest probability
                digit = np.argmax(digit_confidence)
                symbol = np.argmax(symbol_confidence)

                digit_prob = max(digit_confidence)
                symbol_prob = max(symbol_confidence)
                if digit_prob > symbol_prob:
                    res[row][col] = digit
                    


                
                

