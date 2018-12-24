import http
import requests
import sys
from string import Template
from os import listdir
from os.path import isfile,join
import csv
from pandas import DataFrame
from threading import Thread, Lock
import time

from rx import Observable, Observer

song_df = DataFrame(columns=['artist_name', 'song_name', 'genres', 'danceability',  'energy',  'key',  'loudness',  'mode',  'speechiness',  'acousticness',  'instrumentalness',  'liveness',  'valence',  'tempo', 'time_signature'])
df_lock = Lock()

def fetch_task(self,song):
    songlist = self.getSongids(song)['tracks']['items']
    if len(songlist) > 0:
        index = None
        for idx, track in enumerate(songlist):
            for artist in track['artists']:
                if artist['id'] == self.id:
                    index = idx
                    break
        if index is not None :
            stats = self.getSongStatistics(songlist[index]['id'])
            statistic = {
                'artist_name': self.name,
                'song_name': song,
                'genres': self.genres,
                'danceability': stats['danceability'],
                'energy': stats['energy'],
                'key': stats['key'],
                'loudness': stats['loudness'],
                'mode': stats['mode'],
                'speechiness': stats['speechiness'],
                'acousticness': stats['acousticness'],
                'instrumentalness': stats['instrumentalness'],
                'liveness': stats['liveness'],
                'valence': stats['valence'],
                'tempo': stats['tempo'],
                'time_signature': stats['time_signature']
            }
            df_lock.acquire()
            global song_df
            song_df = song_df.append(statistic, ignore_index=True)
            df_lock.release()


token = "BQAjk2_uvE4PPW150SDUgqHi7EVNwthMe1pDF2BxI4luJOjfBeRdLz0E15lpzDXmnCI1L_tawBwTwVXfG4YHOgQUL_3ea7wgFD_CGC4D_lgsPM5YnuJ_265k5nQ8HqWX2kd3z1jqTb9gFlSMoRFVrnqKLRbXMZewou1949Mw4AGKIKFd3Vio2_PFaTFEkzEkoROZECHRX2Fx3fXrVJOD9C2aQjDUXIkLzcIVFQDS9LyDUO8hqeLAiZqpLDbQ2Cc-QuAnwDS4UA9mkOvjM-hf"
midi_dir = "./clean_midi"
not_found=0


class Artist():
    def __init__(self,name):
        self.name = name
        self.spotifyname= None
        self.id = None
        self.potential_ids=[]
        self.genres=[]
        self.local_song_names=[]
        self.local_mathced_song_list=[]
        self.local_unmatched_song_list=[]
        print("looking for artist: " + self.name)
        self.getLocalSongNames()

    def getLocalSongNames(self):
        artistpath = midi_dir+ "/"+ self.name
        #Fixed join with empty string clause
        raw_local_song_names = [".".join(f.split(sep=".")[:-1]) for f in listdir(artistpath) if isfile(join(artistpath, f))]
        self.local_song_names = []
        for song in raw_local_song_names:
            if song[-1].isdigit() and song[-2] == ".":
                continue
            else:
                self.local_song_names.append(song)
        self.local_song_names.sort()
        print("number of songs: ", len(self.local_song_names))

    def getArtist(self):
        while True:
            request_url = Template("https://api.spotify.com/v1/search?q=$artist&type=artist").substitute(artist=self.name)
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token
            }
            response = requests.get(request_url, headers=headers)
            if response.status_code==429:
                time.sleep(int(response.headers['Retry-After']))
                continue
            fetched_artist_names = response.json()['artists']['items']
            if (len(fetched_artist_names) > 0):
                self.spotifyname = fetched_artist_names[0]['name']
                self.id = fetched_artist_names[0]['id']
                self.genres = fetched_artist_names[0]['genres']
            else:
                self.spotifyname = "unmatched artist"
                global not_found
                not_found = not_found + 1
            print("Local name ", self.name, " spot name: ", self.spotifyname)
            return response

    def getSongids(self, songname):
        while True:
            request_url = Template("https://api.spotify.com/v1/search?q=$song&type=track").substitute(song=songname)
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token
            }
            response = requests.get(request_url,headers=headers)
            if response.status_code==429:
                time.sleep(int(response.headers['Retry-After']))
                continue
            return response.json()

    def getSongStatistics(self,id):
        while True:
            request_url = Template("https://api.spotify.com/v1/audio-features/$song_id").substitute(song_id=id)
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token
            }
            response = requests.get(request_url, headers=headers)
            if response.status_code==429:
                time.sleep(int(response.headers['Retry-After']))
                continue
            response = response.json()
            return response

    def getMatchedSongs(self):
        if self.id is None:
            return
        threads=[]
        for song in self.local_song_names:
            threads.append(Thread(target=fetch_task,args=(self,song)))
        for i in range(len(self.local_song_names)):
            threads[i].start()
        for i in range(len(self.local_song_names)):
            threads[i].join()

artistList=[]
onlyfiles = [f for f in listdir(midi_dir) if not isfile(join(midi_dir, f))]
onlyfiles.sort()

for filename in onlyfiles:
    artist = Artist(filename)
    artistList.append(artist)

for artist in artistList:
    artist.getArtist()
    artist.getMatchedSongs()
    song_df.to_csv('statistics.csv',sep=',')

print(not_found, " artists unmatched")
