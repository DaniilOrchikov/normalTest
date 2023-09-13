from math import sqrt
import librosa
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load('9270cbb9c02ea04.mp3', sr=4000)
amplitude = librosa.amplitude_to_db(y)

audio_volume = abs(amplitude)[sr * 60:sr * 2 * 60]
print(sr)
print(len(audio_volume))

mu = sum(audio_volume) / len(audio_volume)
sigma = sqrt(sum((i - mu) ** 2 for i in audio_volume) / (len(audio_volume) - 1))
print(mu, sigma)
ms = 0
for i in audio_volume:
    ms += i - mu
print(ms * 10000000)

plt.hist(audio_volume, bins=20, density=True)

x = np.linspace(mu - 3 * sigma, mu + 3 * sigma)
y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

plt.plot(x, y, color='red')

plt.show()

# with open('txt.txt', 'w') as f:
#     for i in range(len(audio_volume)):
#         print(f'{i}\t{audio_volume[i]}\t{audio_volume[i] - mu}\t{(audio_volume[i] - mu) ** 2}', file=f)
