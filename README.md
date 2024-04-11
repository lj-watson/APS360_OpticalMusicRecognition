# Sheet Music Recognition

## Description

This project uses multiple alogirthms and a deep learning model to transform one line digital printed sheet music into audio format.

## Requirements

All libraries are in requirements.txt: `pip install -r requirements.txt`

## Running

Drop the score in png format into 'release' folder, naming it input.png. Execute the program using `python runner.py`. MIDI file will appear in 'audio' folder.

## Method

The input image is fed through a staff line removal file, which replaces the black staff lines with white lines and then reconstructs the symbols. This new image is fed into a region proposal file that uses selective search. Each symbol this algorithm finds in the image is output into a folder as a .png image, and is then given sequentially to the deep learning model to classify. The model, a Convolutional Neural Network, returns a string of identified symbols in their order of appearance. To retreive the pitch, the staff lines are reconstructed by locating the rows on the image with the most pixels. A pitch for each symbol from the region proposal is recorded in a new string. The symbol and pitch strings are combined to generate a MIDI file, which uses some basic logic to check for things like key, clef, ornaments, and time signature. 

## Example

[input][https://github.com/lj-watson/APS360_OpticalMusicRecognition/blob/master/example/input.png]
