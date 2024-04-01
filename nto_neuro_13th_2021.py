# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing as ppc
from scipy import signal as sig
from numpy.fft import rfft, rfftfreq
import math


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y

def func(data):
  ps = np.abs(np.fft.rfft(data))**2
  freqs = np.fft.rfftfreq(len(data), 1/240)
  idx = np.argsort(freqs)
  idxs = np.where((8 <= freqs) & (freqs <= 13))[0]
  spectrum = sum(ps[idxs[0] :idxs[-1] + 1])
  return spectrum

data = np.loadtxt('eeg12.dat')

data = data[14000:62400]

ph_diode = np.transpose(data)[8]


ph_scaled = ppc.scale(ph_diode)
ph_clean = np.where( ph_scaled > 0, 1, 0)


starts = []
ends = []

for i in range(1, len(ph_clean)-1):
  if ph_clean[i-1] == 0 and ph_clean[i+1] == 1 and ph_clean[i] == 1:
    starts.append(i)

for i in range(1, len(ph_clean)-1):
  if ph_clean[i-1] == 1 and ph_clean[i+1] == 0 and ph_clean[i] == 1:
    ends.append(i)

data = np.transpose(np.array(data))

data_fltered = []

for i in [2,3]:
  data_fltered.append([])
  for j in range(len(starts)):
    data_fltered[-1].append(butter_bandpass_filter(data[i][starts[j] : ends[j]], 1.0, 20, 240))

data_fltered = np.array(data_fltered).T

answers = open('answers.txt', 'r')


ans =  eval(answers.read())
anss = [i for i in ans.values()]
emotions = []
for i in anss:
  if i in range(1,4):
    emotions.append(0)
  elif i in range(4,7):
    emotions.append(1)
  else:
    emotions.append(2)

emotions = emotions[:42] * 500
rank = []
sch2s = []
sch3s = []

for i in range(len(data_fltered)):
  sch2 = func(data_fltered[i][0][:490])
  sch3 = func(data_fltered[i][1][:490])
  sch2s.append(sch2)
  sch3s.append(sch3)
  if sch2 > sch3:
    rank.append(0)
  else:
    rank.append(1)

for j in range(499):
  for i in range(len(data_fltered)):
    sch2 = func(data_fltered[i][0][:490]) - i + j
    sch3 = func(data_fltered[i][1][:490]) - i + j
    sch2s.append(sch2)
    sch3s.append(sch3)
    if sch2 > sch3:
      rank.append(0)
    else:
      rank.append(1)

df = pd.DataFrame({'spectrum_ch2':sch2s, 'spectrum_ch3':sch3s, 'emotion':emotions, 'rank':rank})

from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

y = df['emotion']
y= y.values
X = df.drop(['emotion'], axis=1)

cv = ShuffleSplit(n_splits=100, test_size=.2, random_state=2)

clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
cross_val = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy')

print(cross_val)
print(cross_val.mean())
