"""
@brief List of classes to ignore in datasets, and mapping to unified class names

Last updated: 03/06/24
"""

ignored_classes = {
    
    "OpenOMR": [
        "crotchet",
        "minim",
        "quaver_br",
        "quaver_tr",
        "semiquaver_br",
        "semiquaver_tr"
    ],

    "Audiveris": [
        "articStaccatissimoAbove",
        "brace",
        "coda",
        "codaSquare",
        "dynamicMP",
        "dynamicPiano",
        "flag8thDown",
        "flag8thUp",
        "flag16thUp",
        "flag32ndUp",
        "flag128thDown",
        "graceNoteAcciaccaturaStemUp",
        "graceNoteAppoggiaturaStemUp",
        "ledger"
        "noteheadBlack",
        "noteheadBlackSmall",
        "noteheadHalf",
        "rest128th",
        "segno",
        "stem",
        "timeSigCommon",
        "tuplet3"
    ],

    "Printed": [
        "2-2-Time", 
        "9-8-Time", 
        "C-Clef",
        "Common-Time", 
        "Double-Flat", 
        "Double-Sharp", 
        "Eighth-Grace-Note", 
        "Multiple-Hlaf-Notes", 
        "Onehundred-Twenty-Eight-Note", 
    ],

    "Rebelo1": [
        "C-Clef",
        "Common-Time", 
        "Eighth-Grace-Note", 
        "Multiple-Eighth-Notes", 
        "Multiple-Half-Notes", 
        "Multiple-Quarter-Notes", 
        "Multiple-Sixteenth-Notes", 
        "Staccatissimo", 
        "Tie-Slur"
    ],

    "Rebelo2": [
        "1-8-Time", 
        "4-2-Time", 
        "4-8-Time", 
        "5-8-Time", 
        "6-4-Time", 
        "7-4-Time", 
        "8-8-Time", 
        "9-8-Time", 
        "Breve", 
        "C-Clef", 
        "Chord", 
        "Common-Time", 
        "Double-Whole-Rest", 
        "Glissando", 
        "Multiple-Eighth-Notes", 
        "Multiple-Sixteenth-Notes", 
        "Staccatissimo", 
        "Stopped",
        "Tie-Slur", 
        "Tuplet", 
        "Turn"
    ]
}

mapped_classes = {

    "OpenOMR": {
        "bass": "F-Clef",
        "demisemiquaver_line": "Beam",
        "flat": "Flat",
        "natural": "Natural",
        "quaver_line": "Beam",
        "semibreve": "Whole-Note",
        "semiquaver_line": "Beam",
        "sharp": "Sharp",
        "treble": "G-Clef"
    },

    "Audiveris": {
        "accidentalFlat": "Flat",
        "accidentalNatural": "Natural",
        "accidentalSharp": "Sharp",
        "articStaccatoAbove": "Dot",
        "articTenutoBelow": "Tenuto",
        "augmentationDot": "Dot",
        "barlineDouble": "Barline",
        "barlineSingle": "Barline",
        "fClef": "F-Clef",
        "fClefChange": "F-Clef",
        "gClef": "G-Clef",
        "gClefChange": "G-Clef",
        "keyFlat": "Flat",
        "keySharp": "Sharp",
        "noteheadWhole": "Whole-Note",
        "rest8th": "Eighth-Rest",
        "rest16th": "Sixteenth-Rest",
        "rest32nd": "Thirty-Two-Rest",
        "rest64th": "Sixty-Four-Rest",
        "restHalf": "Whole-Half-Rest",
        "restQuarter": "Quarter-Rest",
        "restWhole": "Whole-Half-Rest",
        "timeSig4over4": "4-4-Time",
        "timeSigCutCommon": "Cut-Time",
    },

    "Printed": {

    },

    "Rebelo1": {

    },

    "Rebelo2": {

    },
}