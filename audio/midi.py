'''
@Given a string of symbols and octaves, conver to a midi audio file
'''

from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
import json
import sys

# List of key signatures using Major notation which can also represent natural minors
sharp_key_signatures = {
    0: "C",
    1: "G",
    2: "D",
    3: "A",
    4: "E",
    5: "B",
    6: "F#",
    7: "C#"
}

flat_key_signatures = {
    0: "C",
    1: "F",
    2: "Bb",
    3: "Eb",
    4: "Ab",
    5: "Db",
    6: "Gb",
    7: "Cb"
}

# Mappings to notes and durations integer equiavelent
# TODO: fill this in
note_mapping = {
    'A0': 21,
    'A#0': 22, 'Bb0': 22,
    'B0': 23,
    'C1': 24,
    'C#1': 25, 'Db1': 25,
    'D1': 26,
    'D#1': 27, 'Eb1': 27,
    'E1': 28,
    'F1': 29,
    'F#1': 30, 'Gb1': 30,
    'G1': 31,
    'G#1': 32, 'Ab1': 32,
    'A1': 33,
    'A#1': 34, 'Bb1': 34,
    'B1': 35,
    'C2': 36,
    'C#2': 37, 'Db2': 37,
    'D2': 38,
    'D#2': 39, 'Eb2': 39,
    'E2': 40,
    'F2': 41,
    'F#2': 42, 'Gb2': 42,
    'G2': 43,
    'G#2': 44, 'Ab2': 44,
    'A2': 45,
    'A#2': 46, 'Bb2': 46,
    'B2': 47,
    'C3': 48,
    'C#3': 49, 'Db3': 49,
    'D3': 50,
    'D#3': 51, 'Eb3': 51,
    'E3': 52,
    'F3': 53,
    'F#3': 54, 'Gb3': 54,
    'G3': 55,
    'G#3': 56, 'Ab3': 56,
    'A3': 57,
    'A#3': 58, 'Bb3': 58,
    'B3': 59,
    'C4': 60,
    'C#4': 61, 'Db4': 61,
    'D4': 62,
    'D#4': 63, 'Eb4': 63,
    'E4': 64,
    'F4': 65,
    'F#4': 66, 'Gb4': 66,
    'G4': 67,
    'G#4': 68, 'Ab4': 68,
    'A4': 69,
    'A#4': 70, 'Bb4': 70,
    'B4': 71,
    'C5': 72,
    'C#5': 73, 'Db5': 73,
    'D5': 74,
    'D#5': 75, 'Eb5': 75,
    'E5': 76,
    'F5': 77,
    'F#5': 78, 'Gb5': 78,
    'G5': 79,
    'G#5': 80, 'Ab5': 80,
    'A5': 81,
    'A#5': 82, 'Bb5': 82,
    'B5': 83,
    'C6': 84,
    'C#6': 85, 'Db6': 85,
    'D6': 86,
    'D#6': 87, 'Eb6': 87,
    'E6': 88,
    'F6': 89,
    'F#6': 90, 'Gb6': 90,
    'G6': 91,
    'G#6': 92, 'Ab6': 92,
    'A6': 93,
    'A#6': 94, 'Bb6': 94,
    'B6': 95,
    'C7': 96,
    'C#7': 97, 'Db7': 97,
    'D7': 98,
    'D#7': 99, 'Eb7': 99,
    'E7': 100,
    'F7': 101,
    'F#7': 102, 'Gb7': 102,
    'G7': 103,
    'G#7': 104, 'Ab7': 104,
    'A7': 105,
    'A#7': 106, 'Bb7': 106,
    'B7': 107,
    'C8': 108
}

# Duration set to 120 BPM
duration_mapping = {
    'Quarter-Note': 480,
    'Quarter-Rest': 480,
    'Half-Note': 960,
    'Half-Rest': 960,
    'Whole-Note': 1920,
    'Whole-Rest': 1920,
    'Eighth-Note': 240,
    'Eighth-Rest': 240,
    'Sixteenth-Note': 120,
    'Sixteenth-Rest': 120,
    'Thirty-Two-Note': 60,
    'Thirty-Two-Rest': 60,
    'Sixty-Four-Note': 30,
    'Sixty-Four-Rest': 30
}

# Get the symbols and octaves string
with open("octaves.json", 'r') as file:
    data = json.load(file)

octaves = data['octaves']

with open("symbols.json", 'r') as file:
    data = json.load(file)

symbols = data['classification']

# Sets a temporary list of symbols 
symbols_string = symbols

# Dictionary of all valid notes
valid_notes = ["Whole_Note",
                 "Half-Note",
                 "Quarter-Note",
                 "Eighth-Note",
                 "Sixteeth-Note",
                 "Thirty-Two-Note",
                 "Sixty-Four-Note"]

# Dictionary of all valid rests
valid_rests = ["Whole_Rest",
               "Half-Rest",
               "Quarter-Rest",
               "Eighth-Rest",
               "Sixteeth-Rest",
               "Thirty-Two-Rest",
               "Sixty-Four-Rest"]

# List of all valid notes and rests respectively 
notes = [symbol.split(":")[0] for symbol in symbols_string.strip("<>").split(":") if symbol.split(":")[0] in valid_notes]
rests = [symbol.split(":")[0] for symbol in symbols_string.strip("<>").split(":") if symbol.split(":")[0] in valid_rests]

# Separates the symbols and octave pitches
octave_items = octaves.strip("<>").split(":")
symbol_items = symbols.strip("<>").split(":")

# Make sure we have the same number of symbols and octaves
if len(octave_items) != len(symbol_items):
    print("Error: number of symbols and octaves do not match!")
    sys.exit()

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

# Set to grand piano
track.append(Message('program_change', program=0))
# Set default tempo
track.append(MetaMessage('set_tempo', tempo=bpm2tempo(120)))

# Get time signature
# If the Time signature appears after a note then there might be an error
# so just set default time of 4-4 in that case
time_sig = "4-4-Time"
time_indx = -1
for indx, symbol in enumerate(symbol_items):
    if "Time" in symbol:
        time_indx = indx
        time_sig = symbol
        break

first_note_indx = -1;
for indx, symbol in enumerate(symbol_items):
    if "Note" in symbol:
        if indx < time_indx:
            print("Warning: time signature appears after first note, setting to default time")
            time_sig = "4-4-Time"
            first_note_indx = indx
            break
        elif indx > time_indx:
            if time_indx == -1:
                first_note_indx = indx
            break

# Set the time signature
time_sig_parts = time_sig.split("-")
track.append(MetaMessage('time_signature', numerator=int(time_sig_parts[0]), denominator=int(time_sig_parts[1])))

# Get key from all the accidentals before the time signature
# In case of errors we assume the key to be C major


# Check if we have a clef element, if not assume G clef
clef = 'G-Clef'
clef_indx = -1
for indx, symbol in enumerate(symbol_items):
    if "Clef" in symbol:
        clef = symbol
        clef_indx = indx
        break
    elif indx > 2: # Probably an error at this point
        break

# Check the time signature appears after the clef
key_stopping = time_indx
if time_indx == -1:
    key_stopping = first_note_indx
key_starting = clef_indx
if clef_indx == -1:
    key_starting == 0
if key_starting >= key_stopping:
    print("Error: time signature appears before clef")
    sys.exit()

# TODO: improve this instead of counting number of flats/sharps
# Count the number of flats or sharps appearing
# If there are both flats and sharps then just assume the key to be C major
num_sharps = symbol_items[key_starting:key_stopping].count('Sharp')
num_flats = symbol_items[key_starting:key_stopping].count('Flat')

if num_sharps == 0:
    key = flat_key_signatures.get(num_flats, "UNKNOWN")
else:
    key = sharp_key_signatures.get(num_sharps, "UNKNOWN")

if key == "UNKNOWN":
    print("Warning: Too many sharps/flats! Setting key signature to C")
    key = "C"

# Adjusts the pitches depending on the key signature 
adjustments = {}
if key == "G":
    adjustments = {'F': 'F#'}
elif key == "D":
    adjustments = {'F': 'F#', 'C': 'C#'}
elif key == "A":
    adjustments = {'F': 'F#', 'C': 'C#', 'G': 'G#'}
elif key == "E":
    adjustments = {'F': 'F#', 'C': 'C#', 'G': 'G#', 'D': 'D#'}
elif key == "B":
    adjustments = {'F': 'F#', 'C': 'C#', 'G': 'G#', 'D': 'D#', 'A': 'A#'}
elif key == "F#":
    adjustments = {'F': 'F#', 'C': 'C#', 'G': 'G#', 'D': 'D#', 'A': 'A#', 'E': 'F'}
elif key == "C#":
    adjustments = {'F': 'F#', 'C': 'C#', 'G': 'G#', 'D': 'D#', 'A': 'A#', 'E': 'F', 'B': 'C'}
elif key == "F":
    adjustments = {'B': "Bb"}
elif key == "Bb":
    adjustments = {'B': "Bb", 'E': 'Eb'}
elif key == "Eb":
    adjustments = {'B': "Bb", 'E': 'Eb', 'A': 'Ab'}
elif key == "Ab":
    adjustments = {'B': "Bb", 'E': 'Eb', 'A': 'Ab', 'D': 'Db'}
elif key == "Db":
    adjustments = {'B': "Bb", 'E': 'Eb', 'A': 'Ab', 'D': 'Db', 'G': 'Gb'}
elif key == "Gb":
    adjustments = {'B': "Bb", 'E': 'Eb', 'A': 'Ab', 'D': 'Db', 'G': 'Gb', 'C': 'B'}
elif key == "Cb":
    adjustments = {'B': "Bb", 'E': 'Eb', 'A': 'Ab', 'D': 'Db', 'G': 'Gb', 'C': 'B', 'F': 'E'}

# Makes the adjustment for all key signatures other than C-Major or A-minor
pitch_items = []
if key is not "C":
    for note in octave_items:
        root_note = note[0]
        if root_note in adjustments:
            adjusted_note = adjustments[root_note] + note[1:]
        else:
            adjusted_note = note
        pitch_items.append(adjusted_note)
else:
    pitch_items = octave_items

# Finally append key
track.append(MetaMessage("key_signature", key=key))

# TODO: finish going through and adding Midi events
# Initialize the MIDI ticks 
cumulative_ticks = 0

# Process notes, rests, and durations
for note, duration_sym in zip(pitch_items, symbol_items):
    # Get note and duration from mapping dictionaries         
    midi_note = note_mapping.get(note, None)
    ticks = duration_mapping.get(duration_sym, None)

    # Checks if both notes and duration are valid
    if midi_note is not None and ticks is not None and duration_sym in valid_notes:
        track.append(Message('note_on', note=midi_note, velocity=64, time=0))
        track.append(Message('note_off', note=midi_note, velocity=64, time=ticks))
        print(f"Note: {note}, {duration_sym}")
    elif duration_sym in valid_rests:
        track.append(Message('note_off', note=0, velocity=0, time=ticks))
        print(f"Rest: {duration_sym}")
    else: 
        # print(f"Invalid note or duration: {note}, {duration_sym}")
        print(f"Not a note: {note}, {duration_sym}")

track.append(MetaMessage('end_of_track'))
mid.save("score.mid")
