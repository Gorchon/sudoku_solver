import cv2
import skimage
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pyautogui
def main():
    # Make sure alt-tabbing switches to the browser where sudoku.com is open
    pyautogui.hotkey("alt", "tab", interval=0.1)
    # Take a screenshot of the screen and find the sudoku
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    preprocessed = preprocess(screenshot)
    square_contour = find_sudoku_contour(preprocessed)
    cropped_grid = crop_grid(screenshot, square_contour)
    cv2.imshow("sudoku", cropped_grid)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
# -- Image preprocessing
def preprocess(screenshot):
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def find_sudoku_contour(preprocessed):
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for contour in contours:
        if is_square(contour):
            squares.append(contour)
    squares = sorted(squares, key=cv2.contourArea, reverse=True)
    if len(squares) == 0:
        return None
    return squares[0]

def is_square(contour):
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    _, _, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    return len(approx) == 4 and abs(aspect_ratio - 1) < 0.1

def crop_grid(screenshot, square):
    x, y, w, h = cv2.boundingRect(square)
    cropped = screenshot[y:y+h, x:x+w]
    return cropped





if __name__ == '__main__':
    main()