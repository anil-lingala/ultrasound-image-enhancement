import cv2

def load_and_convert(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def apply_equalization(gray):
    return cv2.equalizeHist(gray)

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def detect_edges(gray):
    return cv2.Canny(gray, 100, 200)

def median_filter(gray):
    return cv2.medianBlur(gray, 5)