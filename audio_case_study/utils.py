import torchaudio
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import six.moves.urllib.parse
import six.moves.urllib.request
import torch
import librosa
import numpy as np

class Loader():
    def __init__(self,
            x_res: int = 256,
            y_res: int = 256,
            n_fft: int = 2048,
            hop_length: int = 512,
            top_db: int = 80,
            n_iter: int = 32,
            target_sample_rate = 22050,
            num_samples = 22050):
        """
        Args:
            target_sample_rate (int) : the target sample rate of the audio
            num_samples (int) : the desired length of the audio file

        """
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.x_res = x_res
        self.y_res = y_res
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.top_db = top_db
        self.n_iter = n_iter
        
    
    def load_audio(self, path):
        """
        Load an audio file.

        Args:
            path (str) : path to the audio file to be loaded.
        Returns:
            signal (torch.tensor) : the audio file as a tensor
        """
        signal, sr = torchaudio.load(path, normalize=True)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
    
        return signal[0]

    def audio_to_image(self, audio):
        """
        convert an audio file to mel spectrogram image using same logic as training scheme.
        """
        S = librosa.feature.melspectrogram(
            y=audio.numpy(), sr=self.target_sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.y_res)
        log_S = librosa.power_to_db(S, ref=np.max, top_db=self.top_db)
        bytedata = (((log_S + self.top_db) * 255 / self.top_db).clip(0, 255) + 0.5).astype(np.uint8)

        return torch.Tensor(bytedata).permute(1, 0)

    def image_to_audio(self, image, original_audio=None):
        log_S = image.permute(1, 0).numpy().astype("float") * self.top_db / 255 - self.top_db
        S = librosa.db_to_power(log_S)
        audio = librosa.feature.inverse.mel_to_audio(S, sr=self.target_sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_iter=self.n_iter)

        if original_audio is not None:
            if original_audio.shape[0] != audio.shape[0]:
                end = min(original_audio.shape[0], audio.shape[0])
            else:
                end = original_audio.shape[0]
        scale = original_audio[:end]/audio[:end] if original_audio is not None else 1

        return torch.Tensor(audio)*scale
            
        
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.x_res * self.hop_length - 1:
            signal = signal[:, :self.x_res * self.hop_length -1]
        return signal
        
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.x_res * self.hop_length -1:
            num_missing_samples = self.x_res * self.hop_length - length_signal - 1
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
        
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
        
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = signal[0, :]
            signal = signal.unsqueeze(0)
        return signal

def scrape(url, params=None, user_agent=None):
    '''
    Scrape a URL optionally with parameters.
    This is effectively a wrapper around urllib2.urlopen.
    '''

    headers = {}

    if user_agent:
        headers['User-Agent'] = user_agent

    data = params and six.moves.urllib.parse.urlencode(params) or None
    req = six.moves.urllib.request.Request(url, data=data, headers=headers)
    f = six.moves.urllib.request.urlopen(req)

    text = f.read()
    f.close()

    return text

def get_embed_data(url):
    """
    (helper func.)
    Scrape a shakespeare play for sentences.
    
    Parameters
    ----------
    url : str
        The url to scrape. Current set up is designed for shakespeare.mit
    """
    html = scrape(url)
    
    soup = BeautifulSoup(html)
    
    sentences = []
    
    for a in soup.findAll("a"):
        b = str(a)
        if "speech" not in b and "href" not in b:
            b = remove_tags(b)
            if b[0].isupper and b[-1] in ["?", ";", "."]:
                sentences.append(b)
            
    return sentences

def remove_tags(html_string):
    """ 
    (helper func.)
    Remove html tags from scraped data
    """
    return re.sub(r'<.*?>', '', html_string)