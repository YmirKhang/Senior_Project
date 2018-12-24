from os.path import join, isfile
from os import listdir

artistpath = './clean_midi'+ "/"+ "Ludwig van Beethoven"
raw_local_song_names = ["".join(f.split(sep=".")[:-1]) for f in listdir(artistpath) if isfile(join(artistpath, f))]
local_song_names = []
for song in raw_local_song_names:
    if song[-1].isdigit() and song[-2] == ".":
        continue
    else:
        local_song_names.append(song)
local_song_names.sort()
print("number of songs: ", len(local_song_names))
for song in local_song_names:
	print(song)
	
