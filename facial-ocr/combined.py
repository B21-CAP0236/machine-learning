from threading import Thread
from difflib import SequenceMatcher

from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt

import pytesseract
import face_recognition
import cv2 as cv


class ThreadWithReturnValue(Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None
    ):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


# Sequence Matcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def im2rgb(image) -> list:
    return cv.cvtColor(image, cv.COLOR_BGR2RGB) 

# Get grayscale image
def get_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def img2rotate(image) -> list:
    return cv.rotate(image, cv.cv2.ROTATE_90_CLOCKWISE)

def isPictureMatch(image, image_from, x1, y1, x2, y2) -> bool:
    """
    Check if picture in KTP (grab using exact coord)
    is match with the one captured from the camera

    Param:
        image = Image from camera
        x1 = first x coord of picture in KTP
        y1 = first y coord of picture in KTP
        x2 = second x coord of picture in KTP
        y2 = second y coord of picture in KTP
    """

    baseimg = im2rgb(img2rotate(face_recognition.load_image_file(image_from)))[x1:y1, x2:y2]
    encodedim1 = face_recognition.face_encodings(baseimg)[0]

    try:
        encodedim2 = face_recognition.face_encodings(im2rgb(image))[0]
    except IndexError:
        return (False, "Index error, Authentication Failed")

    return face_recognition.compare_faces([encodedim1], encodedim2, tolerance=0.5)[0]


def getKtpData(image, x1, y1, x2, y2):
    # Load image
    data = img2rotate(cv.imread(image))[x1:y1, x2:y2]

    # Image Processing
    gray_data = get_grayscale(data)

    # Extraction
    data_result = pytesseract.image_to_string(gray_data, lang="ind")

    return "".join([x for x in data_result if x.isalnum()])


def isFaceMatch(image, x1, y1, x2, y2, show=True, signal=None):
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    threadList = []
    delay = 25
    threshold = 3

    print("[+] Starting camera")
    while len(threadList) < 5:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if delay == 0:
            delay = 25

        if delay == 25:
            print(f"[+] Matching {len(threadList)}th image")
            t = ThreadWithReturnValue(
                target=isPictureMatch,
                args=(
                    frame,
                    image,
                    x1,
                    y1,
                    x2,
                    y2,
                ),
            )
            t.start()
            threadList.append(t)

        frame = cv.circle(frame, (int(width / 2), int(height / 2)), 250, (0, 255, 0), 2)
        
        if show:
            cv.imshow("frame", frame)

            if cv.waitKey(1) == ord("q"):
                break
        else:
            rgbImage = im2rgb(frame)
            h, w, ch = rgbImage.shape
            convertToQtFormat = QImage(rgbImage.data, w, h, ch * w, QImage.Format_RGB888)
            frame = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)

            signal.emit(len(threadList), frame)

        delay -= 1

    cap.release()
    cv.destroyAllWindows()

    # Wait for all the threads to finish
    # and get all return values from the function
    ret = [t.join() for t in threadList]

    if len([x for x in ret if x]) >= threshold:
        return True
    else:
        return False


if __name__ == "__main__":
    getKtpData(210, 270, 0, 520)
    isFaceMatch(215, 590, 500, 800)
