from streamlit_webrtc import webrtc_streamer
import numpy as np
import streamlit as st
import mediapipe as mp
import cv2
import av
from dbr import *


class VideoProcessor:


    def __init__(self) -> None:
        self.width_pixels = 640
        self.height_pixels = 480
        self.distance_to_object = 300
        self.sensor_height_y = 3
        self.sensor_height_x = 4
        self.focal_length_camera = 3
        self.logo_width = 260
        self.logo_height = 65
        self.logo = 'logo_wit.png'
        self.length_cup_cm = 0
        self.width_cup_cm = 0
        self.height_cup_cm = 0
        self.text_QR = ""
        self.option = "Cup"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Initialize dynamsoft barcode reader, with free license code
        BarcodeReader.init_license(
            "t0068fQAAAHI/rXvj1Bb8Y7N1eyBkrcYMFl76F1uFyQW/d+tPuswp/Gv1UrxgC9FXCi2rH2KFXgPc2gjNQiQ8VKcJCWgkeoA="
        )
        reader = BarcodeReader()
        # Initialize empty string for QR-code text processing.
        self.splitted = ""
        # Initialize MediaPipe drawing utils to draw the 3D bounding box around the object(s).
        mp_drawing = mp.solutions.drawing_utils
        # Initialize MediaPipe objectron to find the object and calculate the 3D bounding box.
        mp_objectron = mp.solutions.objectron

        # resize D'atalier logo for the videoframe.
        logo = cv2.imread(self.logo)
        logo = cv2.resize(logo, (self.logo_width, self.logo_height),
                          interpolation=cv2.INTER_CUBIC)  # 528,114

        # Convert image to grey and create mask by thresholding.
        img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        logo = cv2.cvtColor(logo, cv2.COLOR_RGB2BGR)

        with mp_objectron.Objectron(
                model_name=str(self.option),
                static_image_mode=True,
                max_num_objects=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.99, ) as objectron:

            # Detect and decode QR-code from video frame
            results_QR = reader.decode_buffer(img)

            # If there is a QR code detected parse the text_result of the QR code
            if results_QR != None:
                for text_result in results_QR:
                    tr = text_result.barcode_text

                    # Input text for each QR code is 2 columns of text and 5 rows.
                    # Merk Merk
                    # Materiaal Materiaal
                    # Verkoopprijs Verkoopprijs
                    # Gewicht Gewicht
                    # Soort Soort

                    # Each row is seperated by a \t, therefore replace the \t by an empty space to create one row.
                    self.splitted = tr.replace('\t', ' ')
                    # Each  previous row is seperated by a \n, therefore split it on \n to get the 5 individual rows.
                    self.splitted = self.splitted.split('\n')


            # Color image from bgr to rgb
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process image with MediaPipe objectron
            results = objectron.process(image)

            # Find region of interest for the logo
            roi = image[-self.logo_height - 400:-400,
                  -self.logo_width - 20:-20]
            roi[np.where(mask)] = 0
            roi += logo

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # If objectron finds an object in the frame corresponding to the given model name (e.g. cup)
            # Loop over all detected objects
            if results.detected_objects:

                for detected_object in results.detected_objects:
                    print('OBJECT DETECTED')
                    # Draw 3D landmarks on the object (See image above for the landmark numbers)
                    mp_drawing.draw_landmarks(
                        image, detected_object.landmarks_2d,
                        mp_objectron.BOX_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(88, 49, 14),
                                               thickness=8,
                                               circle_radius=5))

                    # Calculate height of the cup
                    # The height of the cup is determined by first taking the absolute difference between the height landmarks
                    # Then multiply the relative height by the pixel_height
                    # Lastly, calculate the heigh in cm by using the following formula
                    # height = (distance to object * heigh of object in pixels * physical sensor height of camera in mm) /
                    # (focal length of camera in mm * height of object in pixels) /10 (mm to cm conversion)

                    height_cup = abs(
                        detected_object.landmarks_2d.landmark[2].y -
                        detected_object.landmarks_2d.landmark[4].y)
                    height_cup_pixels = height_cup * self.height_pixels
                    self.height_cup_cm = (
                                                 (self.distance_to_object * height_cup_pixels *
                                                  self.sensor_height_y) /
                                                 (self.focal_length_camera *
                                                  self.height_pixels)) / 10

                    # Calculate the amount of pixels per cm on the y-axis
                    pixels_per_metric_y = height_cup_pixels / self.height_cup_cm

                    # Calculate length of the cup (Likewise to the height calculation)
                    length_cup = abs(
                        detected_object.landmarks_2d.landmark[2].x -
                        detected_object.landmarks_2d.landmark[6].x)
                    length_cup_pixels = length_cup * self.width_pixels
                    self.length_cup_cm = (
                                                 (self.distance_to_object * length_cup_pixels *
                                                  self.sensor_height_x) /
                                                 (self.focal_length_camera *
                                                  self.width_pixels)) / 10

                    # Calculate width of the cup (Calculate the width of an object by dividing the width in pixels by
                    # the pixels per metric in the y-axis calculated in the height.)
                    width_cup = abs(
                        detected_object.landmarks_2d.landmark[2].y -
                        detected_object.landmarks_2d.landmark[1].y)
                    self.width_cup_cm = (
                                                (width_cup * self.height_pixels) /
                                                pixels_per_metric_y) + 1.5  # +3 for correction

            image[440:, :] = [
                88, 49, 14
            ]  # Create blue box at bottom of the videocapture
            image[300:, 0:180] = [
                88, 49, 14
            ]  # Create blue box at the left bottom corner for QR-code information

            # Coordinates for the measurements (length,width,height) in the frame
            CenterCoordinates = (145, int((image.shape[0] / 2) + 460))

            # length of splitted is greater than 1 if there's a QR code detected, then put the text of the QR code
            # in the left bottom corner.
            # splitted[0] is the first row.
            if (len(self.splitted) > 1):
                cv2.putText(image, self.splitted[0], (int(self.width_pixels * 0.01), int(self.height_pixels * 0.65)), 2,
                            0.5,
                            (255, 255, 255), 1)
                cv2.putText(image, self.splitted[1],
                            (int(self.width_pixels * 0.01), int(self.height_pixels * 0.65) + 30),
                            2, 0.5,
                            (255, 255, 255), 1)
                cv2.putText(image, self.splitted[2],
                            (int(self.width_pixels * 0.01), int(self.height_pixels * 0.65) + 60),
                            2, 0.5,
                            (255, 255, 255), 1)
                cv2.putText(image, self.splitted[3],
                            (int(self.width_pixels * 0.01), int(self.height_pixels * 0.65) + 90),
                            2, 0.5,
                            (255, 255, 255), 1)
                cv2.putText(image, self.splitted[4],
                            (int(self.width_pixels * 0.01), int(self.height_pixels * 0.65) + 120),
                            2, 0.5,
                            (255, 255, 255), 1)

            # Add length, width, height text in the middle bottom of the frame
            cv2.putText(
                image, "Lengte: " + "{:.1f}".format((self.length_cup_cm)) +
                       ' cm' + "  " + "Breedte: " + "{:.1f}".format(
                    (self.width_cup_cm)) + ' cm'
                                           "  " + "Hoogte: " + "{:.1f}".format(
                    (self.height_cup_cm)) + ' cm', (10,465), 2,
                0.7, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")





#streamlit run webrtc_3Dobject_size_detection.py
# streamlit run C:\Users\niels\PycharmProjects\streamlit_webrtc\venv\webrtc_3Dobject_size_detection.py
ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,rtc_configuration={ "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if ctx.video_processor:
    ctx.video_processor.option = st.radio(
        'You can change the object to detect and measure here.',
        ('Cup', 'Chair', 'Shoe', 'Camera'))
