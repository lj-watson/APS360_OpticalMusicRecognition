'''
@Given a string of symbols and octaves, conver to a midi audio file
'''

from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
import json
import sys

# List of key signatures
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
    1: "Dm",
    2: "Gm",
    3: "Cm",
    4: "Fm",
    5: "Bbm",
    6: "Ebm",
    7: "Abm"
}

# Mappings to notes and durations integer equiavelent
# TODO: fill this in
note_mapping_gclef = {
    'C4': 60,
    'D4': 60,
    'E4': 60,
    'F4': 60,
    'G4': 60,
    'A4': 60,
    'B4': 60,
    'C5': 60,
    'D5': 60,
    'E5': 60,
    'F5': 60,
    'G5': 60
}
note_mapping_cclef = {
    'E2': 60,
    'F2': 60,
    'G2': 60,
    'A2': 60,
    'B2': 60,
    'C3': 60,
    'D3': 60,
    'E3': 60,
    'F3': 60,
    'G3': 60,
    'A3': 60,
    'B3': 60
}
duration_mapping = {
    'Quarter': 0,
    'Half': 0,
    'Whole': 0,
    'Eigth': 0,
    'Sixteenth': 0,
    'Thirty-Two': 0
}

# Get the symbols and octaves string
with open("octaves.json", 'r') as file:
    data = json.load(file)

octaves = data['octaves']

with open("symbols.json", 'r') as file:
    data = json.load(file)

symbols = data['classification']

# Make sure we have the same number of symbols and octaves
octave_items = octaves.strip("<>").split(":")
symbol_items = symbols.strip("<>").split(":")

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
if num_sharps != 0 and num_flats != 0:
    key = "C"
elif num_sharps == 0:
    key = flat_key_signatures.get(num_flats, "UNKNOWN")
else:
    key = sharp_key_signatures.get(num_sharps, "UNKNOWN")
if key == "UNKNOWN":
    print("Warning: too many sharps/flats! Setting key signature to C")
    key = "C"

# Finally append key
track.append(MetaMessage("key_signature", key=key))

# TODO: finish going through and adding Midi events
cumulative_ticks = 0
for note, duration in zip(octave_items, symbol_items):
    if note == 'rest':
        # Skip some length
        cumulative_ticks += 
    # Get note, duration from mapping
    midi_note = 
    ticks = 
    track.append(Message('note_on', note=midi_note, velocity=64, time=cumulative_ticks))
    cumulative_ticks = ticks
    track.append(Message('note_off', note=midi_note, velocity=64, time=0))

track.append(MetaMessage('end_of_track'))
mid.save("piano_track.mid")