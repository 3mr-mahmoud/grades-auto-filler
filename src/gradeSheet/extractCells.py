from utils.paper_extraction import *
from utils.commonfunctions import *
import cv2
import pandas as pd


def houghLines(img, type):
    lines = cv.HoughLinesP(img.astype(np.uint8), 0.5, np.pi/180, 100,
                           minLineLength=0.20*min(img.shape[0], img.shape[1]), maxLineGap=25)

    hough_lines_out = np.zeros(img.shape)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (type == "vertical"):
            cv.line(hough_lines_out, (x1, 0),
                    (x2, img.shape[0]), (255, 255, 255), 1)
        else:
            cv.line(hough_lines_out, (0, y1),
                    (img.shape[1], y2), (255, 255, 255), 1)
    return hough_lines_out


def threshold_intersections(pixels,ignoreCornerPoint= False, widthThresholdLow=15 , widthThresholdHigh=1400, heightThresholdLow=8, heightThresholdHigh=120):
    yMap = {}
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            if pixels[i][j] != 0:
                if i in yMap:
                    yMap[i].append((j, i))
                else:
                    yMap[i] = []
                    yMap[i].append((j, i))

    yPixels = list(yMap.keys())
    rows = []   

    # find the first row y coord
    firstRowIndex = 0
    for i in range(len(yPixels)):
        if (heightThresholdLow < (yPixels[i+1] - yPixels[i]) and (yPixels[i+1] - yPixels[i]) < heightThresholdHigh):
            if(ignoreCornerPoint and yPixels[i] < 40):
                continue
            rows.append(yMap[yPixels[i]])
            firstRowIndex = i
            break
    firstRowIndex = 1 if firstRowIndex == 0 else firstRowIndex
    for i in range(firstRowIndex, len(yPixels)-1):
        # diff in y should match the threshold to consider it a new row
        if (heightThresholdLow < (yPixels[i] - yPixels[i-1]) and (yPixels[i] - yPixels[i-1]) < heightThresholdHigh):
            rows.append(yMap[yPixels[i]])

    # Now we need to threshold the columns i.e the x values

    allIntersections = []
    index = 0
    widthThreshold = pixels.shape[1] // 25
    print("widthThreshold", widthThreshold)
    for row in rows:
        invertDirection = False
        elements = []
        for i in range(len(row)):
            # diff in x should be greater than epsilon to consider it a new column
            if i == len(row)-1 or invertDirection:
                if ( widthThresholdLow < (row[i][0] - row[i-1][0]) and (row[i][0] - row[i-1][0]) < widthThresholdHigh):
                    if (ignoreCornerPoint and pixels.shape[1] > row[i][0] and row[i][0] > pixels.shape[1]-20):
                        continue
                    elements.append(row[i])
            else:
                if (widthThresholdLow < (row[i + 1][0] - row[i][0]) and (row[i + 1][0] - row[i][0]) < widthThresholdHigh):
                    if (ignoreCornerPoint and ((0 < row[i][0] and row[i][0] < 20) or row[i + 1][0] > pixels.shape[1]-125)):
                        continue
                    elements.append(row[i])
                    invertDirection = True
        if len(elements) > 0:
            allIntersections.append(elements)
            index += 1
        
    return allIntersections


def extract_cells(paper):
    
    grayPaper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

    _,binary = cv2.threshold(grayPaper,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = 255 - binary



    kernelLength = np.array(binary).shape[1] // 10

    # Vertical Kernel (1 x kernelLength)
    verticalKernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernelLength))
    verticalKernelErode = cv.getStructuringElement(cv.MORPH_RECT, (1, int(kernelLength*0.8)))

    # Horizontal Kernel (kernelLength x 1)
    horizontalKernel = cv.getStructuringElement(
        cv.MORPH_RECT, (kernelLength, 1))
    horizontalKernelErode = cv.getStructuringElement(
        cv.MORPH_RECT, (int(kernelLength*0.8), 1))

    # Apply erosion then dilation to detect vertical lines using the vertical kernel
    erodedImg = cv.erode(binary, verticalKernelErode, iterations=1)
    verticalLinesImg = cv.dilate(erodedImg, verticalKernel, iterations=1)
    show_images([verticalLinesImg], ["verticalLinesImg after opening"])
    verticalLinesImg = houghLines(verticalLinesImg, "vertical")

    # Apply erosion then dilation to detect horizontal lines using the horizontal kernel
    erodedImg = cv.erode(binary, horizontalKernelErode, iterations=2)
    horizontalLinesImg = cv.dilate(erodedImg, horizontalKernel, iterations=3)
    show_images([horizontalLinesImg], ["horizontalLinesImg after opening"])
    horizontalLinesImg = houghLines(horizontalLinesImg, "horizontal")



    # Combine the two images to get the final lines image
    intersections = cv.bitwise_and(verticalLinesImg, horizontalLinesImg)
     # Apply a threshold to detect white areas
    show_images([verticalLinesImg, horizontalLinesImg, intersections], [
         "verticalLinesImg",
         "horizontalLinesImg",
         "intersections",
         ])
    

    _, thresholded = cv2.threshold(grayPaper, 160, 255, cv2.THRESH_BINARY)
    # Calculate the number of white pixels
    white_pixels = np.sum(thresholded == 255)
    total_pixels = thresholded.size
    white_percentage = (white_pixels / total_pixels) * 100

    ignoreCornerPoint = white_percentage > 82
    intersections = threshold_intersections(intersections,ignoreCornerPoint=ignoreCornerPoint)

    rows = len(intersections) - 1
    cols = len(intersections[0]) - 1
    df = pd.DataFrame(index=range(rows), columns=range(cols))
    
    for i in range(rows):
        for j in range(len(intersections[i]) - 1):
            if(i >= rows or j >= cols):
                continue
            x1, y1 = intersections[i][j]
            x2, _ = intersections[i][j + 1]
            

            _, y2 = intersections[i + 1][j]
            w = x2 - x1
            h = y2 - y1
            x = x1
            y = y1

            cell_img = grayPaper[y:y + h, x:x + w]
            cell_img = cv2.resize(cell_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _ , cell_img = cv2.threshold(cell_img,200,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
            df.iloc[i, j] = cell_img

    return df, rows, cols