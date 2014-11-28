#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: distances.py
# date: Thu November 27 17:45 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""distances:

"""

from __future__ import division

import os.path as path


import numpy as np
import scipy.spatial.distance as ssd
from scikits.audiolab import wavread
from sklearn.preprocessing import StandardScaler
import toml

import spectral

def load_config(filename):
    if not path.exists(filename):
        print 'no such file: {0}'.format(filename)
        exit()
    with open(filename) as fid:
        config = toml.loads(fid.read())
    return config

def extract_features_single(filename, config):
    sig, fs, _ = wavread(filename)
    expected_fs = config['features']['preprocessing']['samplerate']
    if fs != expected_fs:
        if config['features']['preprocessing']['resample']:
            try:
                import scikits.samplerate
            except ImportError:
                print 'cannot resample because scikits.samplerate is not ' \
                    'installed. Either resample all audio files externally or ' \
                    'install it.'
            sig = scikits.samplerate.resample(sig, fs / expected_fs,
                                              'sinc_best')
        else:
            print 'samplerate error in file {2}: expected {0}, got {1}.\n' \
                'Try to resample all audio files to the samplerate specified' \
                ' in the config file. If you can\'t resample yourself, set the' \
                ' value of "reset" to "true" in the configuration file.'.format(
                    fs, expected_fs, filename)

    nfilt = config['features']['spectral'].get('filterbanks', 40)
    ncep = config['features']['spectral'].get('nceps', 13)
    do_dct = config['features']['spectral'].get('dct', True)
    lowerf = config['features']['spectral'].get('lowerf', 120)
    upperf = config['features']['spectral'].get('upperf', 6900)
    alpha = config['features']['preprocessing'].get('preemph', 0.97)
    fs = config['features']['spectral'].get('samplerate', 16000)
    frate = config['features']['spectral'].get('framerate', 100)
    wlen = config['features']['spectral'].get('winlen', 0.025)
    nfft = config['features']['spectral'].get('nfft', 512)
    compression = config['features']['spectral'].get('compression', 'log')
    do_deltas = config['features']['spectral'].get('deltas', True)
    do_deltasdeltas = config['features']['spectral'].get('deltas', True)

    encoder = spectral.Spectral(nfilt=nfilt,
                                ncep=ncep,
                                do_dct=do_dct,
                                lowerf=lowerf,
                                upperf=upperf,
                                alpha=alpha,
                                fs=fs,
                                frate=frate,
                                wlen=wlen,
                                nfft=nfft,
                                compression=compression,
                                do_deltas=do_deltas,
                                do_deltasdeltas=do_deltasdeltas)
    return encoder.transform(sig)

def extract_features(files, config):
    X = None
    for f in files:
        spec = extract_features_single(f, config)
        if X is None:
            X = spec
        else:
            X = np.vstack((X, spec))
    mean_norm = config['features']['spectral']['mean_normalize']
    std_norm = config['features']['spectral']['std_normalize']
    if mean_norm or std_norm:
        X = StandardScaler(with_mean=mean_norm, with_std=std_norm).fit_transform(X)
    return X



if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='distances.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Plot distribution of distances between feature frames.',
            epilog="""Example usage:

$ python distances.py AUDIOFILES --config=distances_cfg.toml --plot=myplot.png

extracts spectral features from the audiofiles  according to the configuration
in distances_cfg.toml and plots them.

""")
        parser.add_argument('audiofiles', 'AUDIOFILES',
                            nargs='+',
                            help='input phone file')
        parser.add_argument('-p', '--plot',
                            action='store',
                            dest='plot',
                            default=None,
                            help='plot to this file')
        parser.add_argument('-s', '--save',
                            action='store',
                            dest='save',
                            default=None,
                            help='save distances to this file')
        parser.add_argument('-c', '--config',
                            action='store',
                            dest='config',
                            default='distances_cfg.toml',
                            help='location of configuration file')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='talk more')

        return vars(parser.parse_args())

    args = parse_args()
    audiofiles = args['audiofiles']
    verbose = args['verbose']
    config = load_config(args['config'])
    plot = args['plot']
    save = args['save']

    nsamples = config['distances']['nsamples']
    dfunc = config['distances']['distance']
    if dfunc == 'euclidean':
        dfunc = ssd.euclidean
    elif dfunc == 'cosine':
        dfunc = ssd.cosine
    else:
        print 'invalid distance function, using euclidean'
        dfunc = ssd.euclidean

    lag = config['distances']['lag']

    frames = extract_features(audiofiles, config)

    # random frames
    ixs = np.random.choice(np.arange(frames.shape[0]), nsamples*2, replace=False)
    zipped = np.dstack((frames[ixs[:nsamples],:], frames[ixs[nsamples:],:]))
    random_dists = np.from_iter((dfunc(x[:,0], x[:,1]) for x in zipped),
                                dtype=np.double)

    # near frames
    ixs = np.random.choice(np.arange(frames.shape[0]-lag), nsamples, replace=False)
    zipped = np.dstack((frames[ixs,:], frames[ixs+lag,:]))
    near_dists = np.fromiter((dfunc(x[:,0], x[:,1]) for x in zipped),
                             dtype=np.double)

    if not plot is None:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('white')
        plt.figure()
        sns.distplot(random_dists, label='random')
        sns.distplot(near_dists, label='near')
        plt.xlim(0,)
        plt.ylim(0,)
        plt.legend(loc='best')
        plt.savefig(plot)

    if not save is None:
        import cPickle as pickle
        with open(save, 'wb') as fid:
            pickle.dump((random_dists, near_dists), fid, -1)
