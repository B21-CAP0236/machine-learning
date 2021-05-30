from threading import Thread
from difflib import SequenceMatcher
import face_recognition
import cv2 as cv

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

# Sequence Matcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# get grayscale image
def get_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

def im2rgb(image) -> list:
    return cv.cvtColor(image, cv.COLOR_BGR2RGB) 


def isPictureMatch(image) -> bool:
    #Picture of user baseline comparison
    #the scanned ktp name format should be here with correct image format, the coordinate already correct
    #based on the captured scanner position, tested on several picture taken using builded scanner by fakhri
    baseimg = im2rgb(face_recognition.load_image_file("edi_2.jpeg"))[670:1620,2300:3100]
    encodedim1 = face_recognition.face_encodings(baseimg)[0]
    
    try:
        encodedim2 = face_recognition.face_encodings(im2rgb(image))[0]
    except IndexError:
        return (False, "Index error, Authentication Failed")

    return face_recognition.compare_faces([encodedim1], encodedim2, tolerance=0.5)[0] 

def KTP_NIK():
    #Load image
    nik = cv.imread("edi_2.jpeg")[600:800 ,860:2230]

    #Image Processing
    gray_nik = get_grayscale(nik)
    thresh_nik = thresholding(gray_nik)
    remove = remove_noise(thresh_nik)
    remove = remove_noise(remove)

    # Extraction
    nik_result = pytesseract.image_to_string(remove, lang="ind")
    result_nik = []
    for i in nik_result:
        if i=='\n' or i=='\x0c':
            pass
        else:
            result_nik.append(i)
    result_nik = "".join(result_nik)
    result_nik = result_nik.replace(" ", "")
    print(nik_result)

    # Matching
    # There will be a dataframe variable here that contains NIK
    # For i in db['NIK']:
        # if similar(nik_result, '<QUERIED_NIK_FROM_DB_[i]') > 0.9 :
            # print("Passed NIK Authentication!")
        # else :
            # pass

def faceVid():
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
            t = ThreadWithReturnValue(target=isPictureMatch, args=(frame,))
            t.start()
            threadList.append(t)

        frame = cv.circle(frame, (int(width/2), int(height/2)), 250, (0, 255, 0), 2)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

        delay -= 1

    cap.release()
    cv.destroyAllWindows()

    # Wait for all the threads finished
    # and get all return values from the function
    ret = [t.join() for t in threadList]
    print(f'{ret}')

    if len([x for x in ret if x]) >= 3:
        print("Authentication passed successfully !")
    else:
        print("Sorry, authentication failed, you are not match with the picture !")

if __name__ == "__main__":
    KTP_NIK()
    faceVid()
