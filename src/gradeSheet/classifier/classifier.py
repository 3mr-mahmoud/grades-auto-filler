from gradeSheet.classifier.predict import predictGeneral

def processCells(df, digitsModel, symbolsModel):
    rows = df.shape[0]
    cols = df.shape[1]
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


