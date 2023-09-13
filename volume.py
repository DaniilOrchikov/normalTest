from math import sqrt

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

y, sr = librosa.load('20230911_215908.wav', sr=None)
amplitude = librosa.amplitude_to_db(y)

audio_volume = [abs(amplitude[i]) for i in range(len(amplitude[sr:20 * sr]))]
print(sr)
print(len(audio_volume))

mu = np.mean(audio_volume)
sigma = sqrt(sum((i - mu) ** 2 for i in audio_volume) / (len(audio_volume) - 1))
print(mu, sigma)

plt.hist(audio_volume, bins=9)
# plt.show()

x = [audio_volume[i] for i in range(len(audio_volume)) if i % 100 == 0]
x.sort()

y = norm.pdf(x, mu, sigma)
y *= 200000 / max(y)

plt.plot(x, y, color='red')
plt.show()
