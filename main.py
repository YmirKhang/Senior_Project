from music21 import converter, instrument, note, chord
import pandas as pd
import numpy as np
from math import ceil

num_threads = 8

note_dict = {
        'C':0,
        'D':2,
        'E':4,
        'F':5,
        'G':7,
        'A':9,
        'B':11
    }

#To extract genres from a list of genres
def genre_extractor(row, lookup_genre):
    for genre in row['genres']:
        if lookup_genre in genre:
            return 1
    return 0

#Helper function
def string_to_list(row):
    return [s.strip()[1:-1] for s in row['genres'][1:-1].split(",")]

# Function to get numerical note value from text
def get_notes(text):
    val = (int(text[-1]) - 1) * 12
    val = val + note_dict[text[0]]
    if text[1] == '-':
        val = val -1
    elif text[1] == '#':
        val = val + 1
    while val > 72:
        val = val - 12
    return val

def midi_to_input(artist, song, key, mode):
    midi = converter.parse('./clean_midi/' + artist+'/'+ song + '.mid')
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    print(song)
    
    if parts: 
        # get the track with the most notes from the instruments
        notes_to_parse = max(parts.parts, key=lambda p: p.__len__()).flat.notes
    else: 
        #single instrument
        notes_to_parse = midi.flat.notes
    transposed = transpose(notes_to_parse, key, mode, notes_to_parse.analyze('ambitus').noteStart, notes_to_parse.analyze('ambitus').noteEnd)
    duration = notes_to_parse.duration.quarterLength
    notes = np.zeros((ceil(duration*4) + 1, 74))
    for element in transposed:
        if isinstance(element, note.Note):
            timestep = int(round(element.offset*4)) 
            notes[timestep, get_notes(element.pitch.nameWithOctave)] = element.volume.velocityScalar
            notes[timestep, 73] = max(notes[timestep, 73], element.duration.quarterLength)
        elif isinstance(element, chord.Chord):
            timestep = int(round(element.offset*4)) 
            for part in element:
                notes[timestep, get_notes(part.pitch.nameWithOctave)] = part.volume.velocityScalar
            notes[timestep, 73] = max(notes[timestep, 73], element.duration.quarterLength)     
    return notes

def transpose(note_stream, key, mode,spectrumStart,spectrumEnd):
    start = get_notes(spectrumStart.nameWithOctave)
    if mode == 0:
        key = -1* ((21 - key) % 12)
    if (key < 6) and (key<=start):
        offset = -1 * key 
    else:
        offset = 12 - key
    print(offset)
    return note_stream.transpose(offset)

def lstm_input_from_df(df):
    df['LSTM_input'] = df.apply(lambda row: midi_to_input(row.artist_name, row.song_name, row.key), axis = 1)

#%%
if __name__ == "__main__":
    try:
        songs = pd.read_pickle("./statistics_with_genres.pkl")
    except FileNotFoundError:
        songs = pd.read_csv('./statistics.csv')
        songs['genres'] = songs.apply(lambda row: string_to_list(row),axis=1)
        genre_list = ['pop','rock','jazz','soul','classical','electronic','dance','metal','disco', 'funk']
    
        for genre in genre_list:
            songs['is_' + genre] = songs.apply(lambda row: genre_extractor(row, genre), axis = 1)
    
        songs.to_pickle("./statistics_with_genres.pkl")
    
    #%%
    try:
        songs_as_input = pd.read_pickle("./statistics_as_input.pkl")
    except FileNotFoundError:
        songs_per_thread = int(len(songs)/num_threads)
        df_list = []
        thread_list = []
        for i in range(num_threads):
            if i == num_threads-1:
                df = songs.iloc[i*songs_per_thread:]
            else:
                df = songs.iloc[i*songs_per_thread: (i+1)*songs_per_thread]
            df_list.append(df)
            thread_list.append(Thread(target=lstm_input_from_df, args=(df,)))
                               
        for i in range(num_threads):
            thread_list[i].start()
                            
        for i in range(num_threads):
            thread_list[i].join()
        songs_as_input = pd.concat(df_list, ignore_index = True)
        songs_as_input.to_pickle("./statistics_as_input.pkl")
                           
    print("Finished extracting song input matrices")     
        