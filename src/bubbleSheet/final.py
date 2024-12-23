import cv2
import numpy as np
import pandas as pd
import os
from utils.paper_extraction import *
from utils.commonfunctions import *

def is_approx_circle(contour, epsilon=0.02, circularity_threshold=0.8):
    """
    Checks if a contour approximates a circle based on aspect ratio and circularity.

    Parameters:
    - contour: The contour to check.
    - epsilon: The approximation accuracy (smaller values give more accurate approximations).
    - circularity_threshold: A value above which the contour is considered circular.

    Returns:
    - True if the contour is approximately a circle, False otherwise.
    """
    # Approximate the contour to a polygon (using epsilon)
    approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)

    # Get the bounding box of the contour to check aspect ratio
    x, y, w, h = cv2.boundingRect(approx)

    # Calculate the aspect ratio (circle-like contours should have width == height)
    aspect_ratio = float(w) / h

    # Calculate the area and perimeter of the contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate the circularity (higher values indicate more circular shapes)
    if perimeter == 0:
        return False
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Check if the aspect ratio is close to 1 (circle-like) and if the circularity is high
    if 0.8 <= aspect_ratio <= 1.2 and circularity >= circularity_threshold:
        return True
    
    return False

def process_image_for_cropping(gray_img, blurred_img, edged_img):#1
    """
    Perform adaptive thresholding, find contours, and crop the image based on the largest valid contour.

    Parameters:
        gray_img (numpy.ndarray): Grayscale image to process.
        blurred_img (numpy.ndarray): Blurred version of the grayscale image for contour overlay.
        edged_img (numpy.ndarray): Edge-detected image for further processing.
        is_approx_circle (callable): Function to check if a contour is approximately circular.

    Returns:
        dict: A dictionary containing:
            - "contour_overlay": Image with all contours drawn.
            - "white_img_contours": Image with filled contours.
            - "cropped_img": Cropped image based on the first valid contour.
            - "contours": List of detected contours.
    """
    # Step 1: Calculate the threshold using Otsu's method
    img_cpy = gray_img.copy()
    otsu_thresh, _ = cv2.threshold(img_cpy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Adjust the threshold value
    adjusted_thresh = otsu_thresh + 25  # Adjust the threshold by increasing by 25

    # Step 3: Apply the adjusted threshold
    _, img_cpy = cv2.threshold(img_cpy, adjusted_thresh, 255, cv2.THRESH_BINARY)

    # Perform dilation and erosion
    dilated_img = cv2.dilate(img_cpy, (2, 2), iterations=5)
    dilated_img = cv2.erode(dilated_img, (2, 2), iterations=2)
    dilated_img = cv2.dilate(edged_img, (2, 2), iterations=1)

    # Find contours in the processed image
    contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("Number of contours found:", len(contours))

    # Create an image to draw contours on (colored version)
    contour_overlay = cv2.cvtColor(blurred_img, cv2.COLOR_GRAY2BGR)  # Convert to BGR
    cv2.drawContours(contour_overlay, contours, -1, (0, 255, 0), 8)  # Draw green contours

    # Create a white image for filling contours
    white_img_contours = np.ones_like(img_cpy) * 255

    # Sort contours by area
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Fill contours on the white image
    for contour in sorted_contours:
        if is_approx_circle(contour):
            cv2.fillPoly(white_img_contours, pts=[contour], color=0)  # Black filled contours

    # Initialize cropped image
    cropped_img = contour_overlay

    # Process contours for cropping (based on bounding boxes)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Check if contour is valid for cropping
        if w > 0 and h > 0:
            cropped_img = blurred_img[y:y + h, x:x + w]  # Crop using the bounding box
            break  # Stop after cropping the first valid contour
    kernel = np.ones((15,15))
    if (img_cpy.shape[1] < 750):
        kernel = np.ones((6,6))
    
    dilatedImg = cv2.dilate(white_img_contours, kernel, iterations=1)

    erodedImg2 = cv2.erode(dilatedImg, kernel, iterations=2)

    erodedImg = cv2.erode(erodedImg2, kernel, iterations=4)
        
    return white_img_contours,erodedImg2,erodedImg,img_cpy
    
def sharpen_image(image):
    # Threshold the image to binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Morphological operations to fill gaps and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask
    mask = np.zeros_like(cleaned_image)

    # Create a colored copy of the original image for drawing contours
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Loop through all contours
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the bounding box needs adjustments
        # Refine the x-coordinate manually
        new_x = x  # Starting from the bounding box x

        # Iterate over each column to refine the bounding box
        for col in range(x, x + w):
            column = cleaned_image[y:y+h, col]
            white_pixels = np.sum(column == 255)

            # Adjust based on the number of white pixels in the column
            if white_pixels > (h // 2):  # Adjust this threshold as needed
                new_x = col
                break

        # Refine the y-coordinate manually (adjust the topmost point)
        new_y = y  # Starting from the bounding box y

        # Iterate over rows to find the topmost point of the contour
        for row in range(y, y + h):
            row_pixels = cleaned_image[row, x:x + w]
            white_pixels = np.sum(row_pixels == 255)

            # Adjust based on the number of white pixels in the row
            if white_pixels > (w // 2):  # Adjust this threshold as needed
                new_y = row
                break

        # Adjust the bounding box width and height if needed
        refined_w = x + w - new_x
        refined_h = y + h - new_y

        # Fill the refined rectangle on the mask
        cv2.rectangle(mask, (new_x, new_y), (new_x + refined_w, new_y + refined_h), 255, thickness=cv2.FILLED)


    # Optional: Apply dilation to "sharpen" the edges
    sharpened_image = cv2.dilate(mask, kernel, iterations=1)

    return sharpened_image

def process_image_for_final_output(white_img_contours, erodedImg2, erodedImg, img_cpy):#2

    
    # Find contours from the eroded image
    contours, hierarchy = cv.findContours((255 - erodedImg).astype("uint8"), 
                                          cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    # List to store bounding box dimensions of valid contours
    dimensions_contours = []

    # Create a mask filled with white
    mask = np.ones_like(img_cpy, dtype="uint8") * 255

    # Process each contour
    for contour in contours:
        # Get bounding box for the contour
        (x, y, w, h) = cv.boundingRect(contour)
        dimensions_contours.append((x, y, w, h))

        # Fill valid contours on the mask with black (to preserve them)
        cv.drawContours(mask, [contour], -1, (0, 0, 0), thickness=cv.FILLED)

    # Sort dimensions of contours by x-coordinate
    dimensions_contours = sorted(dimensions_contours, key=lambda dimension: dimension[0])
    print("Dimensions of contours:", dimensions_contours)

    # Invert the mask so the valid contours are white and the rest is black
    inverted_mask = cv.bitwise_not(mask)

    # Process the inverted mask
    sharpened_image = sharpen_image(inverted_mask)

    # Generate the pattern using the white contours
    output_pattern = 255 - (255 - white_img_contours) * sharpened_image

    # Create the final output image
    output_image = cv.bitwise_and(img_cpy, img_cpy, mask=sharpened_image)
    output_to_modify = output_image.copy()    # to do take output_image when call function
    sharpened_image = sharpened_image * (255 - erodedImg2)

    # Set pixels in the output image to white where the process_image mask is zero
    output_image[sharpened_image == 0] = 255
    # show_images([output_image, output_pattern, output_to_modify])
    return output_image, output_pattern, dimensions_contours, output_to_modify

def crop_and_sort_images_with_contours(image, dimensions_contours):
    """
    Crop and sort images based on bounding box dimensions and apply contours.
    
    Parameters:
        image (numpy.ndarray): The image from which to crop.
        dimensions_contours (list): List of bounding box dimensions [(x, y, w, h), ...].
        
    Returns:
        tuple: Contains:
            - sorted_images: List of sorted cropped images with contours.
            - ID_contour: The first cropped image (contour with smallest y).
    """
    cropped_images = []

    # Loop through each bounding box in the dimensions_contours list
    for dimension in dimensions_contours:
        x, y, w, h = dimension
        cropped_image = image[y:y+h, x:x+w]

        # Store images with their y and x coordinates for sorting
        cropped_images.append((y, x, cropped_image))

    # Sort by y-coordinate (top to bottom)
    sorted_cropped_images = sorted(cropped_images, key=lambda item: item[0])

    # Get the first contour (the one with the smallest y) to show it separately
    ID_contour = sorted_cropped_images[0][2] if sorted_cropped_images else None

    # Ignore the first contour (the one with the smallest y)
    sorted_cropped_images = sorted_cropped_images[1:]

    # Sort the remaining cropped images by the x-coordinate (left to right)
    sorted_cropped_images = sorted(sorted_cropped_images, key=lambda item: item[1])

    # Extract the sorted images (ignoring the y and x coordinates)
    sorted_images = [image for _, _, image in sorted_cropped_images]

    return sorted_images, ID_contour

def crop_and_sort_images(output_image, out_pattern, out_to_modify, dimensions_contours):#3
    """
    Crop and sort images from multiple sources (output_image, out_pattern, out_to_modify) and apply contours.
    
    Parameters:
        output_image (numpy.ndarray): The processed image from which to crop.
        out_pattern (numpy.ndarray): The pattern image to crop.
        out_to_modify (numpy.ndarray): The image to modify and crop.
        dimensions_contours (list): List of bounding box dimensions [(x, y, w, h), ...].
    
    Returns:
        dict: Contains sorted cropped images with contours from output_image, out_pattern, and out_to_modify.
    """
    # Process the images and sort them with contours
    sorted_images, ID_contour = crop_and_sort_images_with_contours(output_image, dimensions_contours)
    sorted_images_to_modify, ID_contour_to_modify = crop_and_sort_images_with_contours(out_to_modify, dimensions_contours)
    sorted_images_patterns, ID_contour_pattern = crop_and_sort_images_with_contours(out_pattern, dimensions_contours)
    return   sorted_images_to_modify, sorted_images_patterns, ID_contour, ID_contour_to_modify, ID_contour_pattern
    
def detect_bubbles_watershed(image):
    """
    Detect bubbles, calculate grid size, and complete the pattern.
    
    Args:
        image: Input image as a numpy array
    Returns:
        tuple: (num_rows, num_cols, centers, visualization)
    """
    if image is None:
        raise ValueError("Input image is None")
    
    # Initial preprocessing
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Get initial centers and radii
    initial_centers = []
    initial_radii = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue
            
        (x, y), radius = cv2.minEnclosingCircle(contour)
        initial_centers.append((int(x), int(y)))
        initial_radii.append(radius)
    
    if not initial_centers:
        return 0, 0, [], np.zeros_like(image)
    
    centers = np.array(initial_centers)
    avg_radius = np.mean(initial_radii)
    
    # Sort centers by y-coordinate
    centers = centers[centers[:, 1].argsort()]
    
    # Split into rows based on y-coordinate clustering
    rows = []
    current_row = [centers[0]]
    row_threshold = avg_radius * 1.5
    
    for center in centers[1:]:
        if abs(center[1] - current_row[0][1]) < row_threshold:
            current_row.append(center)
        else:
            rows.append(np.array(current_row))
            current_row = [center]
    rows.append(np.array(current_row))
    
    # Sort centers within each row by x-coordinate
    rows = [row[row[:, 0].argsort()] for row in rows]
    
    # Calculate number of rows and columns
    num_rows = len(rows)
    num_cols = max(len(row) for row in rows)  # Use the longest row for num_cols
    
    print(f"Detected grid size: {num_rows} rows x {num_cols} columns")
    
    # Calculate average x-spacing
    x_spacings = []
    for row in rows:
        if len(row) > 1:
            x_spacings.extend(np.diff(row[:, 0]))
    avg_x_spacing = np.mean(x_spacings) if x_spacings else avg_radius * 2
    
    # Complete each row to match num_cols
    completed_centers = []
    for row_idx, row in enumerate(rows):
        row_y = np.mean(row[:, 1])
        
        if len(row) > 0:
            min_x = np.min(row[:, 0])
            expected_x = np.linspace(min_x, min_x + (num_cols - 1) * avg_x_spacing, num_cols)
            
            for x_pos in expected_x:
                # Check if there's already a circle nearby
                too_close = any(
                    np.sqrt((c[0] - x_pos)**2 + (c[1] - row_y)**2) < avg_radius
                    for c in row
                )
                
                if not too_close:
                    completed_centers.append((int(x_pos), int(row_y)))
                else:
                    # Add the existing nearby circle
                    closest = min(row, key=lambda c: abs(c[0] - x_pos))
                    completed_centers.append((int(closest[0]), int(closest[1])))
    
    centers = np.array(completed_centers)
    
    # Create visualization
    visualization = cv2.cvtColor(opening.copy(), cv2.COLOR_GRAY2BGR)
    # Draw detected circles in white
    for center in centers:
        cv2.circle(visualization, (int(center[0]), int(center[1])), int(avg_radius), (255, 255, 255), -1)
    
    # show_images([visualization], ["images="])
    print(f"Original circles: {len(initial_centers)}")
    print(f"Total circles after completion: {len(centers)}")
    return num_rows, num_cols, centers, 255-visualization

def process_images_with_bubbles( ID_contour_pattern,ID_contour_to_modify, croped_images_sorted_patterns, croped_images_sorted_to_modify):#4
    """
    Process the images to detect and modify the bubbles using watershed algorithm.
    
    Parameters:
        out_pattern (numpy.ndarray): The pattern image.
        out_to_modify (numpy.ndarray): The image to modify.
        croped_images_sorted_patterns (list): List of cropped images from the pattern.
        croped_images_sorted_to_modify (list): List of cropped images to modify.
        
    Returns:
        tuple: Contains:
            - updated_images: List of updated images after processing.
            - patterns_completed: List of processed patterns.
            - dimensions: List of dimensions (rows, columns, centers).
    """
    # Apply the watershed detection on the pattern
    ID_rows, ID_cols, ID_centers, pattern_completed = detect_bubbles_watershed(ID_contour_pattern)
    
    # Convert `pattern_completed` to a single-channel grayscale image if necessary
    if len(pattern_completed.shape) == 3 and pattern_completed.shape[2] == 3:
        pattern_completed = cv2.cvtColor(pattern_completed, cv2.COLOR_BGR2GRAY)

    # Modify the pattern and create ID_contour
    ID_contour = 255 - ((255 - pattern_completed) * (255 - ID_contour_to_modify))

    # Initialize lists to store processed patterns and their dimensions
    dimensions_questions = []
    patterns_completed = []

    for i in range(len(croped_images_sorted_patterns)):
        # Call the function for each cropped image
        num_rows, num_cols, centers, pattern_completed = detect_bubbles_watershed(croped_images_sorted_patterns[i])

        # Convert pattern_completed to grayscale if needed
        if len(pattern_completed.shape) == 3 and pattern_completed.shape[2] == 3:
            pattern_completed = cv2.cvtColor(pattern_completed, cv2.COLOR_BGR2GRAY)

        patterns_completed.append(pattern_completed)
        dimensions_questions.append((num_rows, num_cols, centers))
        # print(num_rows, num_cols, len(centers))
    print("cols",dimensions_questions[0][1])    
    num_cols_list = [item[1] for item in dimensions_questions]
    max_cols = max(num_cols_list)

    filtered_images = []
    filtered_num_cols = []
    filtered_centers = []
    filtered_num_rows=[]
    croped_images=[]
    dimensions_quest=[]
    for (row,cols ,center ),image,croped_image in zip(dimensions_questions, patterns_completed,croped_images_sorted_to_modify):
        if  cols==max_cols :
            filtered_images.append(image)
            filtered_num_cols.append(cols)
            filtered_centers.append(center)
            filtered_num_rows.append(row)
            croped_images.append(croped_image)
            dimensions_quest.append((row,cols ,center))
    patterns_completed=filtered_images
    dimensions_questions=dimensions_quest
    show_images(patterns_completed)
    num_cols=filtered_num_cols 
    centers=filtered_centers
    num_rows=filtered_num_rows
    croped_images_sorted_to_modify=croped_images
    croped_images_sorted=[]
    for i in range(len(patterns_completed)):
        croped_images_sorted.append( 255 - ((255 - patterns_completed[i]) * (255 - croped_images[i])))

    return croped_images_sorted,dimensions_questions,ID_rows,ID_cols,ID_contour,ID_centers

def find_id(image, rows, cols, centers):#5
    # show_images([image])
    # Sort centers from top left to bottom right
    sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]))
    min_y = sorted_centers[0][1]
    max_y = sorted_centers[-1][1]
    row_height = (max_y - min_y) // (2 * (rows - 1))
    
    # Split centers into rows
    splitted_rows = []
    ID_code = ''
    for i in range(rows):
        splitted_row = sorted_centers[i * cols:(i + 1) * cols]
        splitted_rows.append(splitted_row)
    
    # Sort each row by the x-coordinate
    for i in range(len(splitted_rows)):
        splitted_rows[i] = sorted(splitted_rows[i], key=lambda c: c[0])
    
    # Loop through rows and calculate scores
    for i in range(len(splitted_rows)):
        scores = []
        for j in range(len(splitted_rows[i])):
            x, y = splitted_rows[i][j]
            # Create a region of interest (ROI) around each center
            bubble = (image[y - row_height : y + row_height, x - row_height : x + row_height]) // 255

            # Count how many pixels are black (assuming the bubble is darker than the background)
            scores.append(np.sum(bubble == 0))
            # print(np.min((bubble)))
        
        # Find the index of the bubble with the highest score (most black pixels)
        max_index = scores.index(max(scores))
        # Map the index to the appropriate character (e.g., numbers or letters)
        ID_code += str(max_index)
    
    return ID_code

def find_answers(images, dimensions, multi_select_threshold=0.88):#6
    """
    Detect answers in bubble sheets, ensuring questions with multiple selections are marked as empty.

    Args:
        images (list): List of preprocessed images (grayscale).
        dimensions (list): List of tuples (rows, cols, centers) for each image.
        multi_select_threshold (float): Ratio of black pixels to the maximum score to consider a bubble selected.

    Returns:
        list: List of answers for all questions, where each answer is a list with a single selected option or empty if multiple are selected.
    """
    question_index = 0
    img_index = 0
    answers = []
    show_images(images)
    # Iterate through each image
    for image in images:
        # Get rows, cols, and centers for the current image
        rows, cols, centers = dimensions[img_index]
        img_index += 1

        # Sort centers from top-left to bottom-right
        sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]))

        # Calculate bounding box ranges
        min_y = sorted_centers[0][1]
        max_y = sorted_centers[-1][1]
        min_x = sorted(sorted_centers, key=lambda c: c[0])[0][0]
        max_x = sorted(sorted_centers, key=lambda c: c[0])[-1][0]
        if(rows==1):
            row_height =  min_y 
            col_width = (max_x - min_x) // (2 * (cols - 1))
        else:
            row_height = (max_y - min_y) // (2 * (rows - 1))
            col_width = (max_x - min_x) // (2 * (cols - 1))

        # Split centers into rows
        splitted_rows = [
            sorted(sorted_centers[i * cols:(i + 1) * cols], key=lambda c: c[0])
            for i in range(rows)
        ]

        # Process each row
        for row in splitted_rows:
            black_scores = np.array([])
            selected_options = []
            
            # Process each bubble in the row
            for x, y in row:
                # Define region of interest (ROI) for the bubble
                x_start = max(x - col_width, 0)
                x_end = min(x + col_width, image.shape[1])
                y_start = max(y - row_height, 0)
                y_end = min(y + row_height, image.shape[0])

                bubble = image[y_start:y_end, x_start:x_end]

                # Perform processing if the bubble is not empty
                if bubble.size > 0:
                    
                    # Calculate the number of black pixels
                    black_pixel_count = np.sum((bubble)//255 == 0)
                    black_scores = np.append(black_scores, black_pixel_count)

            # Determine the selected answers
            if black_scores.size > 0:
                max_black_pixels = np.max(black_scores)
                threshold = multi_select_threshold * max_black_pixels

                # Select bubbles with black pixel counts above the threshold
                for idx, score in enumerate(black_scores):
                    if score >= threshold:
                        selected_options.append(chr(ord('A') + idx))  # Convert index to option (A, B, etc.)

            # Check for multiple selections
            if len(selected_options) > 1:
                selected_options = ['']  # Mark as empty if multiple options are selected

            answers.append(selected_options)
            question_index += 1
            print(f"Question {question_index}: {selected_options}")

    return answers


def process_bubble_sheet_gui(image_file, answers_file):
    # Read the uploaded image and answer key
    file_bytes = image_file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    answers = [line.strip().upper() for line in answers_file.read().decode().splitlines()]

    try:
        # Load and preprocess the image
        extracted_img = extract_paper(image)
        gray_img = cv2.cvtColor(extracted_img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edged_img = cv2.Canny(blurred_img, 20, 50)
        
        white_img_contours, erodedImg2, erodedImg, img_cpy = process_image_for_cropping(
            gray_img, blurred_img, edged_img
        )

        # Process the image to find contours and patterns
        output_image, out_pattern, dimensions_contours, output_to_modify = process_image_for_final_output(
            white_img_contours, erodedImg2, erodedImg, img_cpy
        )

        # Crop and sort images based on contours
        sorted_images_to_modify, sorted_images_patterns, ID_contour, ID_contour_to_modify, ID_contour_pattern = crop_and_sort_images(
            output_image, out_pattern, output_to_modify, dimensions_contours
        )

        # Process images with bubbles
        cropped_images_sorted, dimensions_contours, ID_rows, ID_cols, ID_contour, ID_centers = process_images_with_bubbles(
            ID_contour_pattern, ID_contour_to_modify, sorted_images_patterns, sorted_images_to_modify
        )

        # Detect ID and answers
        ID_code = find_id(ID_contour, ID_rows, ID_cols, ID_centers)
        answers_detected = find_answers(cropped_images_sorted, dimensions_contours)

        # Prepare results for display
        results = {"ID": ID_code}
        for i, expected_answer in enumerate(answers):
            question_column = f"Q{i+1}"
            detected_answer = answers_detected[i][0].upper() if answers_detected[i] else ''
            results[question_column] = 1 if expected_answer == detected_answer else 0

        # Create a DataFrame for display and saving
        df = pd.DataFrame([results])
        return df

    except Exception as e:
        return str(e)
