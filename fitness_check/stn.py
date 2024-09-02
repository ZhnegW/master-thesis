import numpy as np

class STN:
    def __init__(self, snr_db):
        self.snr_db = snr_db

    def generate_snr(self, signal):
        power = signal ** 2
        signal_average_power = np.mean(power)
        signal_averagepower_db = 10 * np.log10(signal_average_power)
        noise_db = signal_averagepower_db - self.snr_db
        noise_watts = 10 ** (noise_db / 10)
        # Generate white noise
        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(signal))
        noise_signal = signal + noise

        return noise_signal

class Ssn(object):
    def __init__(self, snr_db):
        self.snr_db = snr_db

    def generate_snr(self, signal):
        power = signal ** 2
        signal_average_power = np.mean(power)
        signal_averagepower_db = 10 * np.log10(signal_average_power)
        noise_db = signal_averagepower_db - self.snr_db
        noise_watts = 10 ** (noise_db / 10)
        # Generate white noise
        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(signal))
        signal_fft = np.fft.fft(signal)
        noise_signal_spectral = signal_fft + noise
        noise_signal = np.fft.ifft(noise_signal_spectral)


        return noise_signal

