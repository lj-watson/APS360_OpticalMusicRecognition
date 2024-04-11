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

The input image
![input](https://github.com/lj-watson/APS360_OpticalMusicRecognition/blob/master/example/input.png)

Staff lines removed
![sr](https://github.com/lj-watson/APS360_OpticalMusicRecognition/blob/master/example/staff_removed.png)

Symbol regions bounded
![region](https://github.com/lj-watson/APS360_OpticalMusicRecognition/blob/master/example/region_proposals.png)

Staff lines reconstructed
![slr](https://github.com/lj-watson/APS360_OpticalMusicRecognition/blob/master/example/staff_reconstructed.png)

Pitch lines found
![pitch_lines](https://github.com/lj-watson/APS360_OpticalMusicRecognition/blob/master/example/pitch_lines.png)

Model output and pitch algorithm output:
"<G-Clef:Sharp:Sharp:4-4-Time:Quarter-Rest:Quarter-Note:Eighth-Note:Eighth-Note:Sharp:Eighth-Note:Eighth-Note:Quarter-Rest:Quarter-Note:Quarter-Rest:Eighth-Note:Eighth-Note:Eighth-Note:Eighth-Note:Eighth-Note:Eighth-Note:Quarter-Note:Quarter-Rest:Quarter-Note:Quarter-Rest:Quarter-Note:Quarter-Rest>
"<C4:D5:A4:F4:G4:A4:F4:A4:E4:G4:C4:G4:C5:G4:A4:F4:E4:D4:A4:A4:A4:G4:F4:G4:A4:G4>"

MIDI file generated from this data can be found in 'example/score.mid'!
