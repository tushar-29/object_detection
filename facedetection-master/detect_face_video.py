import cv2
from twilio.rest import Client
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

account_sid = 'AC01bc9878d0709388a27a172a98d26c2f'
auth_token = '0b0c7d49a339673f170902c54924bf52'
client = Client(account_sid, auth_token)
MESSAGE = "---UNKNOWN PERSON DETECTED---"
FROM = '+15075797964'
TO = '+919064354496'
TO2 = '+918521188063'


D_FOLDER = 'detected_face'
SUB_FOLDER = 'faces'

if not os.path.isdir(D_FOLDER):
    os.mkdir(D_FOLDER)

path = os.path.join(D_FOLDER, SUB_FOLDER)
if not os.path.isdir(path):
    os.mkdir(path)


webcam = cv2.VideoCapture(0)
count = 1
msg = False

while True:
    _, img = webcam.read()
    cv2.imwrite('% s/orginal_person_% s.png' % (path, count), img)



    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('% s/gray_scale_person_% s.png' % (path, count), gray)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imwrite('% s/box_person_% s.png' % (path, count), img)

        # if not msg:
        #     message = client.messages.create(body=MESSAGE, from_=FROM, to=TO)
        #     print(message.status)
        #     msg = True

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


webcam.release()
cv2.destroyAllWindows()

exit(1)
import cv2


def detect_objects(img, cascade_classifier, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    Applies the Haar cascade classifier on the given image and detects the objects present in it.

    Args:
    - img (numpy.ndarray): The input image on which the Haar cascade classifier will be applied.
    - cascade_classifier (cv2.CascadeClassifier): The pre-trained Haar cascade classifier.
    - scale_factor (float): The scale factor by which the image will be resized.
    - min_neighbors (int): The minimum number of neighbors that each detected object should have.
    - min_size (tuple): The minimum size of the object that can be detected.

    Returns:
    - objects (list): A list of rectangles, each representing an object detected in the image.
    """

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect objects in the image using the Haar cascade classifier
    objects = cascade_classifier.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

    return objects


def sliding_window(image, stepSize, windowSize, cascade_classifier):
    """
    Applies a sliding window over the input image and detects the objects present in each window using the Haar cascade classifier.

    Args:
    - image (numpy.ndarray): The input image on which the sliding window will be applied.
    - stepSize (int): The step size for the sliding window.
    - windowSize (tuple): The size of the sliding window.
    - cascade_classifier (cv2.CascadeClassifier): The pre-trained Haar cascade classifier.

    Returns:
    - objects (list): A list of rectangles, each representing an object detected in the image.
    """

    # Initialize the list of objects
    objects = []

    # Loop over the image and apply the sliding window
    for y in range(0, image.shape[0] - windowSize[1], stepSize):
        for x in range(0, image.shape[1] - windowSize[0], stepSize):

            # Extract the window from the image
            window = image[y:y + windowSize[1], x:x + windowSize[0]]

            # Detect objects in the window using the Haar cascade classifier
            objects_in_window = detect_objects(window, cascade_classifier)

            # Append the detected objects to the list
            for (ox, oy, ow, oh) in objects_in_window:
                objects.append((x + ox, y + oy, ow, oh))

    return objects



import cv2

# Load the input image
img = cv2.imread('input_image.jpg')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply the Haar cascade classifier
cascade_classifier = cv2.CascadeClassifier('haar_cascade_classifier.xml')
objects = cascade_classifier.detectMultiScale(gray_img)

# Extract the edges of the detected objects
edges = cv2.Canny(gray_img, threshold1=50, threshold2=200)

# Extract the edge features of the detected objects
edge_features = []
for (x, y, w, h) in objects:
    object_edges = edges[y:y+h, x:x+w]
    edge_features.append(object_edges)

# Print the matrix of edge features of the detected objects
print(edge_features)





# -------------------------------
def sliding_window(image, stepSize, windowSize, cascade_classifier):
    """
    Applies a sliding window over the input image, detects the objects present in each window using the Haar cascade classifier, and extracts their line features.

    Args:
    - image (numpy.ndarray): The input image on which the sliding window will be applied.
    - stepSize (int): The step size for the sliding window.
    - windowSize (tuple): The size of the sliding window.
    - cascade_classifier (cv2.CascadeClassifier): The pre-trained Haar cascade classifier.

    Returns:
    - line_features (list): A list of numpy arrays, each representing the line features of an object detected in the image.
    """

    # Initialize the list of line features
    line_features = []

    # Loop over the image and apply the sliding window
    for y in range(0, image.shape[0] - windowSize[1], stepSize):
        for x in range(0, image.shape[1] - windowSize[0], stepSize):

            # Extract the window from the image
            window = image[y:y + windowSize[1], x:x + windowSize[0]]

            # Detect objects in the window using the Haar cascade classifier
            objects_in_window = detect_objects(window, cascade_classifier)

            # Extract the line features of the detected objects
            for (ox, oy, ow, oh) in objects_in_window:
                object_lines = cv2.Canny(window[oy:oy + oh, ox:ox + ow], threshold1=50, threshold2=200, apertureSize=3)
                line_features.append(object_lines)

    return line_features


# Load the input image
img = cv2.imread('input_image.jpg')

# Load the pre-trained Haar cascade classifier for line detection
cascade_classifier = cv2.CascadeClassifier('haarcascade_mcs_upperbody.xml')

# Set the window size and step size for the sliding window
windowSize = (100, 100)
stepSize = 10

# Apply the sliding window method to extract the line features of the detected objects
line_features = sliding_window(img, stepSize, windowSize, cascade_classifier)

# Print the matrix of line features of the detected objects
print(line_features)