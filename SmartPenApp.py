from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from kivy.uix.stacklayout import StackLayout

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle

import cv2
import pyimagesearch
from pyimagesearch.getperspectivetransform.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import itertools
import math
import time

"""
TouchImage class is used for the dtection of the mouse movements and clicks. 
"""
class TouchImage(Image):

    def on_touch_down(self, touch):
        print(touch)
    def on_touch_move(self, touch):
        print(touch)
    def on_touch_up(self, touch):
        print("RELEASED!",touch)

class SmartPenApp(App):

    def build(self):
        self.img1=TouchImage(size_hint = (1, 0.66))

        # Define some global variables

        self.mode = "None"
        self.corners = []
        self.penHeaderPoint = np.float32([[[20.0, 30.0]]])

        self.old_frame_once = True
        self.old_frame = None
        self.old_gray = None

        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                              qualityLevel=0.01,
                              minDistance=2,
                              blockSize=5)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.pdf_points = []

        ###
        # User Interface Layout of the Application.
        layout = StackLayout()
        self.modeLabel = Label(text='Mode', size_hint=(0.3,0.33), font_size='20sp')
        layout.add_widget(self.modeLabel)
        self.logLabel = Label(text='Log', size_hint=(0.7,0.33), font_size='25sp')
        layout.add_widget(self.logLabel)
        startButton = Button(text='Start Page Detection', size_hint=(0.25, 0.1))
        startButton.bind(on_press = self.startButtonCallBack)
        layout.add_widget(startButton)
        doneButton = Button(text='Detect Pen header', size_hint=(0.25, 0.1))
        doneButton.bind(on_press = self.doneButtonCallBack)
        layout.add_widget(doneButton)
        startDrawingButton = Button(text='Start Drawing.', size_hint=(0.25, 0.1))
        startDrawingButton.bind(on_press = self.startDrawingButtonCallBack)
        layout.add_widget(startDrawingButton)
        saveImageButton = Button(text='Save image', size_hint=(0.25, 0.1))
        saveImageButton.bind(on_press = self.saveImageButtonCallBack)
        layout.add_widget(saveImageButton)
        layout.add_widget(self.img1)

        #Video capturing from the webcam using opencv
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        Clock.schedule_once(self.set_background, 0)

        return layout

    """
        Changing the Image of the App screen background.
    """
    def set_background(self, *args):
        self.root_window.bind(size=self.do_resize)
        with self.root_window.canvas.before:
            self.bg = Rectangle(source='background.jpg', pos=(0, 0), size=(self.root_window.size))

    def do_resize(self, *args):
        self.bg.size = self.root_window.size

    def startButtonCallBack(self, event):
        self.mode = "START"

    def saveImageButtonCallBack(self, event):
        self.mode = "SAVEIMAGE"

    def doneButtonCallBack(self, event):
        self.mode = "PENHEADER"

    def startDrawingButtonCallBack(self, event):
        self.mode = "DRAW"

    def check_rows_are_in_sequence(self, rows, starting_row, num):
        for i in range(starting_row, starting_row + num):
            if rows[i] + 1 != rows[i + 1]:
                return False
        return True

    def find_point(self, rows, num):
        for i in range(0, len(rows[0]) - num):
            cond = True
            for row in rows:
                for j in range(i, i + num):
                    if row[j] == False:
                        cond = False
            if cond:
                return i
        return -1

    """
        correct_values function is used to search for the Pen header again after getting lost. 
    """
    def correct_value(self):
        while(True):
            ret, frame = self.capture.read()

            trans_image = four_point_transform(frame, self.corners.reshape(4, 2))

            ## convert The imag to hsv
            hsv_n = cv2.cvtColor(trans_image, cv2.COLOR_BGR2HSV)

            mask1_n = cv2.inRange(hsv_n, (0, 180, 180), (10, 255, 255));
            # Generating the final mask to detect red color
            mask_n = mask1_n  # +mask2

            imask_n = mask_n > 0
            green_n = np.zeros_like(trans_image, np.uint8)
            green_n[imask_n] = trans_image[imask_n]

            # Searching for Pen header color algorithm
            num_rows = 1
            num_columns = 1
            rows = np.where(np.any(imask_n == True, axis=1))[0]
            good_points = []
            for i in range(0, len(rows) - num_rows, 1):
                if self.check_rows_are_in_sequence(rows, i, num_rows):
                    j = self.find_point(imask_n[rows[i]: rows[i + num_rows]], num_columns)
                    if j != -1:
                        good_points.append((rows[i], j))
            if len(good_points) > 0:
                x, y = good_points[0]
                return np.float32([[[x, y]]])

    """
        This function is being called continually to update the perform the task based on the current mode.
    """
    def update(self, dt):

        ret, frame = self.capture.read()
        display_image = frame
        if self.mode == "START":
            # find the page in the web camera view.
            setattr(self.modeLabel, 'text', "Step 1: \nPage Detection")
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ratio = image.shape[0] / 500.0
            orig = image.copy()
            image = imutils.resize(image, height = 500)

            # Convert the image to grayscale, blur it, and find edges in the image.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(gray, 10, 50)

            # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
            cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

            # loop over the contours to find the paper contour
            screenCnt = None
            for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.1 * peri, True)

                # if the approximated contour has four points, then we can assume it's the paper.
                if len(approx) == 4:
                    self.mode = "None"
                    screenCnt = approx
                    self.corners = approx
                    break
            # show the contour (outline) of the piece of paper
            if screenCnt is None:
                setattr(self.logLabel, 'text', "Page can't be detected! \nPlease place it correctly and press Start again!")
                self.mode = "None"
            else:
                setattr(self.logLabel, 'text', "Page was Found! \nIf it's correct press detect pen header button. \nIf not, adjust the paper and try again!")
                self.mode = "None"
                print("STEP 2: Find contours of paper")
                cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
                plt.imshow(image)
                plt.show()
                display_image = image
                # apply the four point transform to obtain a top-down
                # view of the original image
                warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
                warped_BGR= cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)

        elif self.mode == "PENHEADER":
            # Find the header of the pen.
            setattr(self.modeLabel, 'text', "Step 2: \nPen header Detection")

            transformed_image = four_point_transform(frame, self.corners.reshape(4, 2))
            pdf_image = np.zeros_like(transformed_image)
            ## convert to hsv
            hsv = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))  # 36, 70
            # Red Mask

            mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255));
            mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255));

            # Range for lower red
            #mask1 = cv2.inRange(hsv, (0, 180, 190), (10, 255, 255))
            # Range for upper range
            #mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
            # Generating the final mask to detect red color
            mask = mask1 +mask2

            imask = mask > 0
            green = np.zeros_like(transformed_image, np.uint8)
            green[imask] = transformed_image[imask]

            # searching for pen header in the image algorithm.
            num_rows = 1
            num_columns = 1

            rows = np.where(np.any(imask == True, axis=1))[0]
            print(rows)
            good_points = []
            print(good_points)
            for i in range(0, len(rows) - num_rows, 1):
                if self.check_rows_are_in_sequence(rows, i, num_rows):
                    j = self.find_point(imask[rows[i]: rows[i + num_rows]], num_columns)
                    if j != -1:
                        good_points.append((rows[i], j))
            print(good_points)
            if len(good_points) > 0:
                x, y = good_points[0]
                print("X: ", x)
                print("Y: ", y)
                self.penHeaderPoint = np.float32([[[y,x]]])
                self.mode = "None"
                setattr(self.logLabel, 'text', "Pen header found successfully! \nIf it's correct, please press Start Drawing. \nIf not, try again!")
                img = cv2.circle(transformed_image,(y, x), 5, [100, 255, 255], -1)
                plt.imshow(img)
                plt.show()
            else:
                self.mode = "None"
                setattr(self.logLabel, 'text', "Can't find pen header!\nPlease place it correctly and try again!")
        elif self.mode == "DRAW":
            # Process the user hand movements and drawings.
            setattr(self.modeLabel, 'text', "Step 3: \nDrawing")
            setattr(self.logLabel, 'text', "Drawing began! \nWhen you are done, press Save Image button!")

            t_frame = four_point_transform(frame, self.corners.reshape(4, 2))
            if (self.old_frame_once):
                print("Found page and Pen header")
                self.old_frame_once = False
                self.old_frame = t_frame
                self.old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
                self.mask = np.zeros_like(self.old_frame)


            frame_gray = cv2.cvtColor(t_frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.penHeaderPoint, None, **self.lk_params)
            if p1 is None:
                print("Header", self.penHeaderPoint)
                p1 = self.correct_value()
                print(self.penHeaderPoint)
            d = math.sqrt((p1[0][0][0] - self.penHeaderPoint[0][0][0]) ** 2 + (p1[0][0][1] - self.penHeaderPoint[0][0][1]) ** 2)
            if err[0][0] > 50 or d > 30:
                correcting = True
                print("Error: ", err[0][0], " D: ", d)
                p1 = self.correct_value()
                print(p1)

            # Select good points
            good_new = p1[0]

            good_old = self.penHeaderPoint[0]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.pdf_points.append(old)
                self.mask = cv2.line(self.mask, (a, b), (c, d), [100, 255, 255], 2)
                frame = cv2.circle(frame, (a, b), 5, [100, 255, 255], -1)

            img = self.mask.copy()
            img[np.where((img == [0, 0, 0]).all(axis=2))] = t_frame[np.where((img == [0, 0, 0]).all(axis=2))]
            display_image = img

            # Now update the previous frame and previous points
            self.old_gray = frame_gray.copy()
            self.penHeaderPoint = good_new.reshape(-1, 1, 2)

        elif self.mode == "SAVEIMAGE":
            setattr(self.modeLabel, 'text', "Step 4: \nSaving the image!")
            setattr(self.logLabel, 'text',"Image saved!")

            # create a new blank image and draw on it.
            img = np.zeros([500, 500, 3], dtype=np.uint8)
            img.fill(255)  # or img[:] = 255
            for i in range(0, len(self.pdf_points)-2):
                a, b = self.pdf_points[i].ravel()
                c, d = self.pdf_points[i+1].ravel()
                img = cv2.line(img, (a, b), (c, d), [100, 255, 255], 2)
            plt.imshow(img)
            plt.show()
            self.mode = "None"
        # change the image in the camera screen.
        buf1 = cv2.flip(display_image, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(display_image.shape[1], display_image.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1

if __name__ == '__main__':
    SmartPenApp().run()