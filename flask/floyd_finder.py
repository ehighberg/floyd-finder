#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:48:25 2019

@author: Errol Highberg
"""


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import pickle

import essentia.standard as es
from sklearn.preprocessing import StandardScaler

from flask import Flask, request, render_template, redirect, url_for, flash, session


# Library configs
sns.set()
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams.update({'font.size': 20})


# Global configs
sample_rate = 44100
samples_per_frame = 4096
hop_length = 2048

window_type = 'blackmanharris92'

peak_params = {'magnitudeThreshold':0.00001,
               'minFrequency':40,
               'maxFrequency':5000,
               'maxPeaks':60}

HPCP_params = {'size':36, 
                'bandPreset':False,
                'minFrequency':40,
                'maxFrequency':5000,
                'nonLinear':True,   
                'harmonics':8,
                'windowSize':1.33} 

key_params = {'profileType':'shaath',  
             'numHarmonics':8,         
             'pcpSize':36,
             'slope':0.6,
             'usePolyphony':True,
             'useThreeChords':True}

secs_per_frame = samples_per_frame / sample_rate


# Feature Extraction
def gen_frames(filepath):
    """Cuts audio into many frames"""
    # Convert file to mono raw audio
    audio = es.MonoLoader(filename=filepath, sampleRate=sample_rate)()
    
    # Cut audio into frames and expand them into windowed frames for better processing
    frame_gen = es.FrameGenerator(audio, frameSize=samples_per_frame, hopSize=hop_length)
    frames = np.array([es.Windowing(size=samples_per_frame, type=window_type)(frame)
                            for frame in frame_gen])
    
    return frames


def get_spectral_info(frame):
    """Gets spectrum frequencies and their magnitudes for a single frame"""
    spectrum = es.Spectrum(size=samples_per_frame)(frame)
    freqs, mags = es.SpectralPeaks(**peak_params)(spectrum)
    mags = es.SpectralWhitening()(spectrum, freqs, mags)
    return spectrum, freqs, mags


def gen_frame_HPCP(spectral_info):
    """Generates HPCP for a single frame"""
    spectrum, freqs, mags = spectral_info
    return es.HPCP(**HPCP_params)(freqs, mags)


def gen_global_HPCP(frames):
    """Stitches together frame-based HPCPs into full HPCP"""
    return np.array([gen_frame_HPCP(get_spectral_info(frame)) for frame in frames])


def get_key_attrs(hpcp):
    """Gets key and scale info from HPCP"""
    return es.Key(**key_params)(np.mean(hpcp, axis=0))


def get_beats(filepath):
    """Gets beat locations by sample number, as well as global BPM"""
    audio = es.MonoLoader(filename=filepath, sampleRate=sample_rate)()
    return es.RhythmExtractor2013(method='multifeature')(audio)


def gen_beat_HPCP(ticks, hpcp):
    """Converts HPCP from one entry per frame to one entry per beat.
    Each entry is the average of the per-frame values in the corresponding beat interval."""
    
    # 2x adjustment needed to adjust for windowing making each frame twice as large
    frame_ticks = 2 * ticks / secs_per_frame
    
    def average_HPCP_frames(start_frame, end_frame, hpcp):
        return np.mean(hpcp[int(start_frame): int(end_frame)], axis=0)
    
    adjusted_hpcp = []
    prev_frame = 0
    for frame_num in frame_ticks:
        adjusted_hpcp.append(average_HPCP_frames(prev_frame, frame_num, hpcp))
        prev_frame = frame_num
    
    return np.array(adjusted_hpcp)


def calc_means(hpcp):
    """Arithmetic mean, geometric mean, and median for all semitones in HPCP"""
    ameans = np.mean(hpcp, axis=0)
    medians = np.median(hpcp, axis=0)
    gmeans = stats.gmean(hpcp, axis=0)
    
    return ameans, gmeans, medians


def song_dict(filepath, semitone_offset=0, save_hpcp=False):
    """Collects all information from a single song"""
    
    # Gather all of the info
    frames = gen_frames(filepath)
    global_hpcp = gen_global_HPCP(frames)
    
    global_hpcp = np.roll(global_hpcp, semitone_offset, axis=1)
    
    key, scale, strength, rel_strength = get_key_attrs(global_hpcp)
    bpm, ticks, confidence, estimates, intervals = get_beats(filepath)
    adj_hpcp = gen_beat_HPCP(ticks, global_hpcp)
    ameans, gmeans, medians = calc_means(adj_hpcp)
    all_means = (ameans, gmeans, medians)
    
    # Format info and put into dict
    mean_names = ['amean', 'gmean', 'median']
    
    song_stats = dict()
    song_stats['title'] = filepath.split(sep='/')[-1][:-4]
    
    if len(filepath.split(sep='/')) > 1:
        song_stats['album'] = filepath.split(sep='/')[-2]
    else:
        song_stats['album'] = 'unknown'
        
    song_stats['filepath'] = filepath[:-4]
    song_stats['semitone_offset'] = semitone_offset
    song_stats['bpm'] = bpm
    song_stats['key'] = key
    song_stats['scale'] = scale
    song_stats['total_beats'] = int(adj_hpcp.shape[0])
    
    # Add means and median for each semitone
    for mean_num in range(len(mean_names)):
        for semitone in range(adj_hpcp.shape[1]):
            mean_type = mean_names[mean_num]
            song_stats['semitone_%s_%d' % (mean_type, semitone)] = all_means[mean_num][semitone]
    if save_hpcp:        
        song_stats['hpcp'] = adj_hpcp
    
    return song_stats


# Load Pickles
def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


all_songs = load_from_pickle('all_songs.pkl')
features = load_from_pickle('features_list.pkl')
lr = load_from_pickle('lr.pkl')


# Setup scaler for prediction
scaler = StandardScaler()

X_train = all_songs[features]
scaler.fit_transform(X_train)
y_train = all_songs['title']


# For conversion of candidate song
key_dict = {'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5,
            'F#': 6, 'C#': 7, 'Ab': 8, 'Eb': 9, 'Bb': 10, 'F': 11}


# Extract features from live song and classify
def one_song_df(filepath):
    """Generates a 1-row DataFrame for single-song classification"""
    
    one_song_dict = dict()
    filename = filepath.split(sep='/')[-1]
    song_stats = song_dict(filepath)
    one_song_dict[filename[:-4]] = song_stats
    one_song_dict['dummy'] = song_stats
    
    one_df = pd.DataFrame.from_dict(one_song_dict, orient='index').drop('dummy')
    
    one_df['key'] = one_df['key'].map(key_dict)
    one_df = (pd.get_dummies(one_df['scale'])
                     .merge(one_df, left_index=True, right_index=True)
                     .drop(['scale', 'filepath'], axis=1))
    
    if 'minor' not in one_df.columns:
        one_df['minor'] = 0
    elif 'major' not in one_df.columns:
        one_df['major'] = 0
        
    one_df['unique_id'] = (one_df['title'] + '_'
                                    + one_df['album'] + '_'
                                    + str(one_df['semitone_offset']))
    
    return one_df


def predict_one(one_df):
    """Classifies a single live recording"""
    
    row = one_df.loc[one_df.index[0]]
    X_one = row[features]
    scaler.transform(np.array(X_one).reshape(1, -1))
    return lr.predict(np.array(X_one).reshape(1, -1))[0]


def predict_title(filepath):
    """Gives prediction for the file"""
    return predict_one(one_song_df(filepath))


def album_name(song_name):
    """Gives album name of a studio track"""
    return all_songs[all_songs['title'] == song_name].album.values[0]


# Flask Time!
# Config
app = Flask('predict')

app.config['upload_folder'] = './uploads'
app.config['SECRET_KEY'] = 'my super secret key'


# Pages
@app.route("/")
def hello():
    flash('Hello')
    return '''Welcome to Floyd Finder!
              <a href='/upload'>Get your prediction here!</a>
              '''


@app.route('/upload', methods=["POST", "GET"])
def upload_file():
    flash('UPLOAD')
    if request.method == 'POST':
        print('POST')
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['upload_folder'], filename))
            return redirect(url_for('hello'))
            # return redirect(url_for('show_prediction',
                                    # filename=filename))
    print('HTML')
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload .mp3 to be identified</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


# Wiki Links
wiki_prefix = 'http://en.wikipedia.org/wiki/'
wiki_suffix = {'A Momentary Lapse of Reason': 'A_Momentary_Lapse_of_Reason',
               'Animals': 'Animals_(Pink_Floyd_album)',
               'A Saucerful of Secrets': 'A_Saucerful_of_Secrets',
               'Atom Heart Mother': 'Atom_Heart_Mother',
               'Meddle': 'Meddle',
               'More': 'More_(soundtrack)',
               'Obscured by Clouds': 'Obscured_by_Clouds',
               'The Dark Side of the Moon': 'The_Dark_Side_of_the_Moon',
               'The Division Bell': 'The_Division_Bell',
               'The Final Cut': 'The_Final_Cut_(album)',
               'The Piper at the Gates of Dawn': 'The_Piper_at_the_Gates_of_Dawn',
               'The Wall': 'The_Wall',
               'Ummagumma': 'Ummagumma',
               'Wish You Were Here': 'Wish_You_Were_Here_(Pink_Floyd_album)'
               }

@app.route('/predict?filename=<filename>')
def show_prediction(filename):
    prediction = predict_title(os.path.join(app.config['upload_folder'], filename))
    album = album_name(prediction)
    wiki_link = wiki_prefix + wiki_suffix[album]
    return render_template('prediction.html',
                           prediction=prediction,
                           album=album,
                           wiki_link=wiki_link)


app.run(debug=True)
