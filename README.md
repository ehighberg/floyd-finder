# Floyd Finder
If you've collected music from all over the internet, I'm sure you've run into this problem: the filenames are, sometimes, complete garbage. For example, PinkFloydThePerfectDay062777Disc101.mp3; that doesn't help you figure out what song this is! So I set out to automatically deduce the correct track names, and along the way the project morphed into it's current state.
<br>
At this time Floyd Finder can be used as a Flask app, via upload of mp3 files from your computer for identification. Supports 131 unique tracks from 15 studio albums, from Piper to The Division Bell.
<br>
[Presentation slides here.](https://docs.google.com/presentation/d/1EyX_12hw2EtFkLVwCLaCSKUenuOmA0_nrXgHI6hQQl4/edit?usp=sharing)

## Getting Started
Starting from an Anaconda installation, you'll need to install [Essentia](https://essentia.upf.edu/documentation/). Then navigate to the flask folder, and run `floyd_finder.py` via terminal. Navigate to the address displayed in the terminal with your browser, and follow the links and onscreen directions. Enjoy having a little assistance in maintaining your music library.

## Built With
[Essentia](https://essentia.upf.edu/documentation/) - C++ library with Python bindings, for feature extraction from audio.
