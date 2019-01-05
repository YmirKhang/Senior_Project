#%%
from music21 import converter, instrument, note, chord
notes = [
]

#%%
midi = converter.parse('./clean_midi/Iron Maiden/Fear of the Dark.mid')
notes_to_parse = None
keySignature = None
timeSignature = None
parts = instrument.partitionByInstrument(midi)
print(midi.flat.analyze('ambitus').noteEnd,midi.flat.analyze('ambitus').noteStart)
if parts: # file has instrument parts
    notes_to_parse = max(parts.parts, key=lambda p: p.__len__()).recurse()
else: # file has notes in a flat structure
    notes_to_parse = midi.flat.notes
for element in notes_to_parse:
    if isinstance(element, note.Note):
        notes.append(str(element.pitch))
    elif isinstance(element, chord.Chord):
        notes.append('.'.join(str(n) for n in element.normalOrder))
        chordexamp = element
#%%

