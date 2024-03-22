'''
@brief Convert final midi file to wav file for playing
'''

# TODO: install sound font

from midi2audio import FluidSynth

FluidSynth().midi_to_audio('piano_track.mid', 'piano_track.wav')