# **anANTara** machine learning

Machine learning for anANTara project are consisting of OCR module and Face Recognition (and Similarity) module that will be deployed and used in the vending machine created in https://github.com/B21-CAP0236/embedded-system.

This repository will act as submodule in the embedded-system repository that will then be integrated with the components inside the vending machine.

We didnt manage to create our own version of the model because there are only some limited amount of time (and that we thought its too hard to create 2 custom inference model for this project) and also these models that comes in one package with the `pytesseract` and `face_recognition` library are already good, so we only wrap it in a module and also combine it with some preprocessing for our own specific purpose.

## Feature(s)

These are features or modules that available for machine learning part :

1. OCR Module

   We use [pytesseract](https://github.com/madmaze/pytesseract) in combination with some image processing to grab the text out of the ID Card (KTP) of the bansos recipients.
   
   We then compare the captured text with the NIK from the database using SequenceMatcher from the difflib package to verify if the NIK is correct or not.
   
   The first version of the OCR are available in the [OCR folder](./OCR).
   
2. Face Recognition and Similarity Module

   We are using [face_recognition](https://github.com/ageitgey/face_recognition) module also in combination with some image processing to grab photo part of the ID Card (KTP) and check if it match with the ones captured from the webcam in the vending machine.
   
   We actually developed 2 version for the module:
   
   - The capture version so it only capture one single image from the webcam and match it with the one from ID Card and its available in [facial-capt-img folder](./facial-capt-img)
   - The video version that will use threading to take several images and automatically match every single captured images with the one from ID Card and its available in [facial-video folder](./facial-video)  

3. Combined Module

   We combine those 2 module (OCR and Face Recognition and Similarity) to be served as a one compact submodule for the embedded system part, its available in the [facial-ocr folder](./facial-ocr).
   
   In this module we actually removed the multithreading for the face recognition because of limited memory in the embedded system (raspberry pi) and its too laggy to run even just more than one thread in a time.

## Development

### Tech Stack

- pytesseract
- face_recognition
- SequenceMatcher
- opencv2

to install the required dependencies simply type at command prompt/terminal:
```pip3 install -r requirements.txt```

### How to Contribute

These are steps that need to be done for development :
- Fork this repository
- Create issue in this repository about what problem you want to fix / what feature you want to add
- Start the development in your own repository by first creating branch that are unique to the development (problem to fix / feature to add)
- Open pull request to this repository and ask maintainer (anantara-machinelearning-team) that consist of [@ihza](https://github.com/zaza-ipynb), [@fakhri](https://github.com/fakhrip), and [@triska](https://github.com/Triskarum01) to review the PR
- Wait for the review approval and merge if approved

## Deployment

The machine learning will be deployed in the embedded system and served as a submodule in the [embedded-system repository](https://github.com/B21-CAP0236/embedded-system)
