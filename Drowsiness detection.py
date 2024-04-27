import cv2
import dlib
from tkinter import Tk, Label, Canvas
from PIL import Image, ImageTk
import numpy as np
from scipy.spatial import distance as dist

# Constants
EAR_THRESH = 0.25  # EAR threshold for drowsiness
EAR_CONSEC_FRAMES = 20  # Number of consecutive frames the EAR must be below the threshold to trigger drowsiness

# Initialize face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Admin\Desktop\models\shape_predictor_68_face_landmarks.dat")

# Function to calculate EAR
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # Return the eye aspect ratio
    return ear

# Initialize Tkinter window
root = Tk()
root.title("Drowsiness Detection")

# Create canvas for displaying video frame
canvas = Canvas(root, width=640, height=480)
canvas.pack(side="left")

# Create label for drowsiness status
drowsiness_label = Label(root, text="Drowsiness: Awake")
drowsiness_label.pack(side="right")

# Initialize frame counter and drowsiness flag
frame_counter = 0
is_drowsy = False

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = cv2.VideoCapture(0).read()  # 0 for webcam

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    # Process each detected face
    for face in faces:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, face)
        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        # Extract the left and right eye coordinates
        left_eye = shape[36:42]
        right_eye = shape[42:48]

        # Calculate the EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Check if the EAR is below the threshold
        if ear < EAR_THRESH:
            frame_counter += 1
            if frame_counter >= EAR_CONSEC_FRAMES:
                is_drowsy = True
                drowsiness_label.config(text="Drowsiness: Asleep")
        else:
            frame_counter = 0
            is_drowsy = False
            drowsiness_label.config(text="Drowsiness: Awake")

    # Update the GUI with the frame and drowsiness status
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.imgtk = imgtk
    canvas.config(width=img.width, height=img.height)
    canvas.create_image(0, 0, anchor='nw', image=imgtk)
    root.update()

# Close the Tkinter window
root.mainloop()
