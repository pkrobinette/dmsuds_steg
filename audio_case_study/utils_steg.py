"""
Steganography utils

Currently Included:
lsb text-into-audio from tensor
lsb text-into-audio from wav file
lsb audio-into-audio from tensor
dwt text-into-audio from tensor
dwt audio-into-audio from tensor
dct audio-into-audio from tensor

"""

import torch
import pywt
import scipy.fftpack as fftpack
import numpy as np
import math

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#              Least significant bit (LSB) methods
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def lsb_embed_text_into_audio(audio_signal, msg, normalized=True):
    """
    Embed a text message into an audio file using the least significant bit method.

    Args:
        audio_path (str): the tensor audio file. 
        msg (str): the secret message to embed.
        normalized (bool): if true, the audio is normalized; else, not normalized.

    Returns: 
        None. The container audio file which contains the secret message is saved to output_path.
    """
    #
    # Un-normalize the audio signal if needed
    # and make byte array
    #
    modified_audio_data = audio_signal
    if normalized:
        modified_audio_data = (audio_signal*(2**15)).to(torch.int16)
    #
    # Convert the message to binary and add a flag
    #
    binary_msg = ''.join(format(ord(i), '08b') for i in msg)
    binary_msg += '00000000'
    #
    # Embed the message
    #
    index = 0
    for bit in binary_msg:
        # Modify the least significant bit of the audio sample
        current_byte = modified_audio_data[index]
        modified_audio_data[index] = (current_byte & ~1) | int(bit)
        index += 1
        if index >= len(modified_audio_data):
            raise ValueError("Audio file is too short to hold the secret message.")

    if normalized:
        return modified_audio_data/(2**15)
        
    return modified_audio_data

def lsb_extract_text_from_audio(audio_signal, normalized=True):
    """
    Extract a secret message hidden in an audio signal with LSB.

    Args:
        audio_signal (torch.tensor): the tensor audio file.
        normalized (bool): if true, the data is normalized; else, not normalized.

    Returns: 
        message (str): message hidden in the audio signal
    """
    #
    # Un-normalize the audio signal if needed
    # and make byte array
    #
    modified_audio_data = audio_signal
    if normalized:
        modified_audio_data = (audio_signal*(2**15)).to(torch.int16)
    #
    # Extract bits from the least significant bits of the audio samples
    #
    extracted_bits = [int(modified_audio_data[i] & 1) for i in range(modified_audio_data.shape[0])]
    #
    # Group the bits into bytes and convert to characters
    #
    message = ''
    for i in range(0, len(extracted_bits), 8):
        byte_as_str = ''.join(str(bit) for bit in extracted_bits[i:i+8])
        if byte_as_str == '00000000':  # Check for the delimiter indicating the end of the message
            break
        message += chr(int(byte_as_str, 2))

    return message

def lsb_embed_audio_into_audio(container_audio_signal, msg_audio_signal, normalized=True):
    """
    Embed an audio message into another audio signal using the 2 least significant bits.
    Max secret size = (container_audio_signal//3)-1

    Args:
        container_audio_signal (torch.tensor): The tensor of the container audio signal.
        msg_audio_signal (torch.tensor): The tensor of the message audio signal to embed.
        normalized (bool): If true, both audios are normalized; else, not normalized.

    Returns: 
        torch.tensor: The modified container audio signal containing the embedded message.
    """
    #
    # Normalize or un-normalize the container audio signal if needed
    #
    if normalized:
        container_audio_data = (container_audio_signal * (2**15)).to(torch.int16)
        message_audio_data = (msg_audio_signal * (2**15)).to(torch.int16)
    else:
        container_audio_data = container_audio_signal
        message_audio_data = msg_audio_signal
    #
    # Shorten secret if need be
    #
    
    #
    # Convert the message audio signal to binary
    #
    binary_msg = [format(sample, '016b') for sample in message_audio_data]
    binary_msg.append("0"*16) # end flag!
    
    #
    # Embed the message using the 2 least significant bits
    #
    index = 0
    for i, bin in enumerate(binary_msg):
        # deal with negatives
        if bin[0] == "-":
            if len(bin) == 17:
                bin = "11" + bin[1:]
            else:
                bin = "110" + bin[1:]
        # for the flag
        elif i == len(binary_msg)-1:
            bin = "10" + bin 
        # deal with positives
        else:
            # add non-positive flag to front
            bin = "00" + bin
        for i in range(0, len(bin), 3):
            current_sample = container_audio_data[index]
            bits_to_embed = int(bin[i:i+3], 2)
            container_audio_data[index] = (current_sample & ~7) | bits_to_embed
            index += 1
            if index >= len(container_audio_data):
                raise ValueError("Container audio is too short to hold the secret message.")

    if normalized:
        return container_audio_data / (2**15)
        
    return container_audio_data


def lsb_extract_audio_from_audio(container_audio_signal, normalized=True):
    """
    Extract an audio message hidden in another audio signal using the 2 least significant bits.

    Args:
        container_audio_signal (torch.tensor): The tensor audio file of the container.
        normalized (bool): If true, the data is normalized; else, not normalized.

    Returns: 
        torch.tensor: The extracted message audio signal.
    """
    #
    # Normalize or un-normalize the container audio signal if needed
    #
    if normalized:
        container_audio_data = (container_audio_signal * (2**15)).to(torch.int16)
    else:
        container_audio_data = container_audio_signal
    #
    # Extract bits from the 2 least significant bits of the audio samples
    # and put in proper format
    #
    all_bits = [format(container_audio_data[i] & 7, '02b') for i in range(container_audio_data.shape[0])]
    extracted_bits = ''
    for bit in all_bits:
        while len(bit) < 3:
            bit = "0" + bit
        extracted_bits += bit
    #
    # Convert the bits back to audio samples
    #
    message_audio_data = []
    for i in range(0, len(extracted_bits), 18):
        # print(extracted_bits[i:i+9])
        if i + 18 > len(extracted_bits): break  # Ensure we don't go past the end
        # positive
        if extracted_bits[i:i+2] == "00":
            sample = max(-2**15, min(2**15-1, int(extracted_bits[i+2:i+18], 2)))
        # negative
        elif extracted_bits[i:i+2] == "11":
            sample = max(-2**15, min(2**15-2, int("-" + extracted_bits[i+2:i+18], 2)))
        else:
            break # we have reached the end flag
        message_audio_data.append(sample)
    #
    # return message_audio_data
    #
    message_audio_data = torch.tensor(message_audio_data, dtype=torch.int16)
    if normalized:
        return message_audio_data / (2**15)
        
    return message_audio_data

def lsb_embed_text_into_audio_wav(audio_path, msg, output_path):
    """
    Embed a text message into an audio file using the least significant bit method.

    Args:
        audio_path (str): the path to the cover audio file (.wav format).
        msg (str): the secret message to embed.
        output_path (str): the path to save the container audio file.

    Returns: 
        None. The container audio file which contains the secret message is saved to output_path.
    """
    #
    # Convert the message to binary
    #
    binary_msg = ''.join(format(ord(i), '08b') for i in msg)
    binary_msg += '00000000'  # add flag
    #
    # Open wave file
    #
    with wave.open(audio_path, 'rb') as cover_audio:
        frame_rate = cover_audio.getframerate()
        n_frames = cover_audio.getnframes()
        audio_data = cover_audio.readframes(n_frames)

    modified_audio_data = bytearray(audio_data)
    #
    # Embed the binary message into the least significant bits of the audio samples
    #
    index = 0
    for bit in binary_msg:
        # Modify the least significant bit of the audio sample
        current_byte = modified_audio_data[index]
        modified_audio_data[index] = (current_byte & ~1) | int(bit)
        print(current_byte, current_byte & ~1, bit, modified_audio_data[index])
        index += 1
        if index >= len(modified_audio_data):
            raise ValueError("Audio file is too short to hold the secret message.")
    #
    # Save the modified audio data to a new file
    #
    with wave.open(output_path, 'wb') as modified_audio:
        modified_audio.setparams(cover_audio.getparams())
        modified_audio.writeframes(modified_audio_data)

    return modified_audio_data
    
def lsb_extract_text_from_audio_wav(audio_path):
    """
    Extract a text message from an audio file using the least significant bit method.

    Args:
        audio_path (str): the path to the audio file (.wav format) that contains the secret message.

    Returns: 
        str: the extracted secret message.
    """
    #
    # Open the wav file
    #
    with wave.open(audio_path, 'rb') as audio_file:
        frame_rate = audio_file.getframerate()
        n_frames = audio_file.getnframes()
        audio_data = audio_file.readframes(n_frames)
    
    audio_data_bytes = bytearray(audio_data)
    #
    # Extract bits from the least significant bits of the audio samples
    #
    extracted_bits = [audio_data_bytes[i] & 1 for i in range(len(audio_data_bytes))]
    #
    # Group the bits into bytes and convert to characters
    #
    message = ''
    for i in range(0, len(extracted_bits), 8):
        byte_as_str = ''.join(str(bit) for bit in extracted_bits[i:i+8])
        if byte_as_str == '00000000':  # Check for the delimiter indicating the end of the message
            break
        message += chr(int(byte_as_str, 2))

    return message
    

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#           Discrete Wavelete transform (DWT) methods
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def dwt_embed_text_into_audio(audio, message):
    """
    Embed a text message into an audio file using the discrete wavelet transform (DWT) method.

    Args:
        audio (torch.Tensor): the audio file loaded as a tensor
        msg (str): the secret message to embed.
        
    Returns:  
        torch.tensor: The modified container audio signal containing the embedded message.
    """
    #
    # Apply DWT and expand high frequency values (cD)
    #
    cA, cD = pywt.dwt(audio.numpy(), 'db2')
    cD_big = cD*(2**15)
    #
    # Turn message into binary message
    #
    binary_msg = ''.join(format(ord(i), '08b') for i in message)
    binary_msg += '00000000'
    #
    # Embed the message
    #
    for i, bit in enumerate(binary_msg):
        # Modify the least significant bit of the audio sample
        current_byte = int(cD_big[i])
        cD_big[i] = (current_byte & ~1) | int(bit)
        if i >= len(cD):
            raise ValueError("Audio file is too short to hold the secret message.")
        
    #
    # Adjust cD and apply IDWT
    #
    audio_create = pywt.idwt(cA, cD_big/2**15, 'db2')
    return torch.Tensor(audio_create[:audio.shape[0]])

def dwt_extract_text_from_audio(audio):
    """
    Extract a text message hidden in an audio signal using the discrete wavelet transform
    method (DWT).

    Args:
        audio (torch.tensor): The container audio.

    Returns: 
        str: The extracted text message.
    """
    #
    # Apply DWT and expand high frequency values (cD)
    #
    cA, cD = pywt.dwt(audio.numpy(), 'db2')
    cD_big = cD*(2**15)
    #
    # Extract bits from the least significant bits of the audio samples
    #
    extracted_bits = [int(round(cD_big[i])) & 1 for i in range(cD_big.shape[0])]
    #
    # Group the bits into bytes and convert to characters
    #
    message = ''
    for i in range(0, len(extracted_bits), 8):
        byte_as_str = ''.join(str(bit) for bit in extracted_bits[i:i+8])
        if byte_as_str == '00000000':  # Check for the delimiter indicating the end of the message
            break
        message += chr(int(byte_as_str, 2))

    return message

def dwt_embed_audio_into_audio(container_audio_signal, msg_audio_signal, normalized=True):
    """
    Embed an audio message into another audio signal using the 2 least significant bits.
    Max secret size = (container_audio_signal//3)-1

    Args:
        container_audio_signal (torch.tensor): The tensor of the container audio signal.
        msg_audio_signal (torch.tensor): The tensor of the message audio signal to embed.
        normalized (bool): If true, both audios are normalized; else, not normalized.

    Returns: 
        torch.tensor: The modified container audio signal containing the embedded message.
        torch.tensor: The shortened secret to compare later on. 
    """
    #
    # Apply DWT and expand high frequency values (cD)
    #
    
    cA, cD = pywt.dwt(container_audio_signal.numpy(), 'db2')
    if normalized:
        cD_big = cD*(2**15)
        message_audio_data = (msg_audio_signal * (2**15)).to(torch.int16)
    else:
        cD_big = cD
        message_audio_data = msg_audio_signal
    #
    # Make secret smaller if need be
    #
    if message_audio_data.shape[0] > cD_big.shape[0]//6-2:
        message_audio_data = message_audio_data[:cD_big.shape[0]//6-2]
        print("Shortening secret ...\n")
    #
    # Turn message into binary message
    #
    binary_msg = [format(sample, '016b') for sample in message_audio_data]
    binary_msg.append("0"*16) # end flag!
    #
    # Embed the message
    #
    index = 0
    for i, bin in enumerate(binary_msg):
        # deal with negatives
        if bin[0] == "-":
            if len(bin) == 17:
                bin = "11" + bin[1:]
            else:
                bin = "110" + bin[1:]
        # for the flag
        elif i == len(binary_msg)-1:
            bin = "10" + bin 
        # deal with positives
        else:
            # add non-positive flag to front
            bin = "00" + bin
        for i in range(0, len(bin), 3):
            current_sample = int(cD_big[index])
            bits_to_embed = int(bin[i:i+3], 2)
            cD_big[index] = (current_sample & ~7) | bits_to_embed
            index += 1
    #
    # Adjust cD and apply IDWT
    #
    if normalized:
        audio_create = pywt.idwt(cA, cD_big/2**15, 'db2')
    else:
        audio_create = pywt.idwt(cA, cD_big, 'db2')
        
    return torch.Tensor(audio_create[:container_audio_signal.shape[0]]), message_audio_data/(2**15)

def dwt_extract_audio_from_audio(audio, normalized=True):
    """
    Extract a text message hidden in an audio signal using the discrete wavelet transform
    method (DWT).

    Args:
        audio (torch.tensor): The container audio.

    Returns: 
        str: The extracted text message.
    """
    # lambda function for rounding --> Very important!
    rnd = lambda x: int(x) if abs(x) % abs(int(x)) < 0.99 else int(round(x))
    #
    # Apply DWT and expand high frequency values (cD)
    #
    cA, cD = pywt.dwt(audio.numpy(), 'db2')
    if normalized:
        cD_big = cD*(2**15)
    else:
        cD_big = cD
    #
    # Extract bits from the least significant bits of the audio samples
    #
    all_bits = [format(rnd(cD_big[i]) & 7, '02b') for i in range(cD_big.shape[0])]
    extracted_bits = ''
    for bit in all_bits:
        while len(bit) < 3:
            bit = "0" + bit
        extracted_bits += bit
    #
    # Convert the bits back to audio samples
    #
    message_audio_data = []
    for i in range(0, len(extracted_bits), 18):
        if i + 18 > len(extracted_bits): break  # Ensure we don't go past the end
        # positive
        if extracted_bits[i:i+2] == "00":
            sample = max(-2**15, min(2**15-1, int(extracted_bits[i+2:i+18], 2)))
        # negative
        elif extracted_bits[i:i+2] == "11":
            sample = max(-2**15, min(2**15-2, int("-" + extracted_bits[i+2:i+18], 2)))
        else:
            break # we have reached the end flag
        message_audio_data.append(sample)
    #
    # return message_audio_data
    #
    message_audio_data = torch.tensor(message_audio_data, dtype=torch.int16)
    if normalized:
        return message_audio_data / (2**15)
        
    return message_audio_data

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#           Discrete Cosign Transform (DCT) methods
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def dct_embed_text_into_audio(container_audio_signal, message, normalized=True):
    """
    Embed a text secret into an audio signal using the east significant bits with DCT.

    Args:
        container_audio_signal (torch.Tensor): The tensor of the container audio signal.
        message (str): The text secret to embed in the cover audio.
        normalized (bool): If true, both audios are normalized; else, not normalized.

    Returns:
        torch.Tensor: The modified container audio signal with the embedded message.
    """
    #
    # Perform DCT on the container signal
    #
    container_dct = fftpack.dct(container_audio_signal.numpy(), norm='ortho')
    if normalized:
        container_dct *= (2**15)
    #
    # Turn message into binary message
    #
    binary_msg = ''.join(format(ord(i), '08b') for i in message)
    binary_msg += '00000000'
    #
    # Embed the message
    #
    for i, bit in enumerate(binary_msg):
        current_byte = int(container_dct[i])
        container_dct[i] = (current_byte & ~1) | int(bit)
        if i >= len(container_dct):
            raise ValueError("Audio file is too short to hold the secret message.")
    #
    # Apply inverse DCT
    #
    modified_audio = fftpack.idct(container_dct, norm='ortho')
    
    if normalized:
        modified_audio /= (2**15)
    
    return torch.tensor(modified_audio[:len(container_audio_signal)], dtype=torch.float)
    

def dct_extract_text_from_audio(audio, normalized=True):
    """
    Extract a text secret from an audio signal using the discrete cosine transform (DCT) method.

    Args:
        audio (torch.Tensor): The container audio signal.

    Returns:
        str: The extracted text secret
    """
    # lambda function for rounding --> Very important!
    rnd = lambda x: int(x) if abs(x) % abs(int(x)) < 0.99 else int(round(x))
    #
    # Apply DCT
    #
    audio_dct = fftpack.dct(audio.numpy(), norm='ortho')
    if normalized:
        audio_dct *= (2**15)
    #
    # Extract bits from the transformed coefficients
    #
    extracted_bits = [rnd(audio_dct[i]) & 1 for i in range(audio_dct.shape[0])]
    #
    # Group the bits into bytes and convert to characters
    #
    message = ''
    for i in range(0, len(extracted_bits), 8):
        byte_as_str = ''.join(str(bit) for bit in extracted_bits[i:i+8])
        if byte_as_str == '00000000':  # Check for the delimiter indicating the end of the message
            break
        message += chr(int(byte_as_str, 2))

    return message


def dct_embed_audio_into_audio(container_audio_signal, msg_audio_signal, normalized=True):
    """
    Embed an audio message into another audio signal using the 2 least significant bits with DCT.
    Max secret size = (len(container_audio_signal)//6)-1

    Args:
        container_audio_signal (torch.tensor): The tensor of the container audio signal.
        msg_audio_signal (torch.tensor): The tensor of the message audio signal to embed.
        normalized (bool): If true, both audios are normalized; else, not normalized.

    Returns:
        torch.tensor: The modified container audio signal with the embedded message.
        torch.tensor: The shortened secret for later comparison.
    """
    #
    # Perform DCT on the container signal
    #
    container_dct = fftpack.dct(container_audio_signal.numpy(), norm='ortho')
    if normalized:
        container_dct *= (2**15)
        message_audio_data = (msg_audio_signal * (2**15)).to(torch.int16)
    else:
        message_audio_data = msg_audio_signal
    #
    # Ensure the message fits within the available space
    #
    if message_audio_data.shape[0] > len(container_dct)//6-2:
        message_audio_data = message_audio_data[:len(container_dct)//6-2]
        print("Shortening secret ...")
    #
    # Convert message to binary
    #
    binary_msg = [format(sample, '016b') for sample in message_audio_data]
    binary_msg.append("0"*16) # End flag
    #
    # Embed the message
    #
    index = 0
    for i, bin in enumerate(binary_msg):
        # deal with negatives
        if bin[0] == "-":
            if len(bin) == 17:
                bin = "11" + bin[1:]
            else:
                bin = "110" + bin[1:]
        # for the flag
        elif i == len(binary_msg)-1:
            bin = "10" + bin 
        # deal with positives
        else:
            # add non-positive flag to front
            bin = "00" + bin
        
        for j in range(0, len(bin), 3):
            current_sample = int(container_dct[index])
            bits_to_embed = int(bin[j:j+3], 2)
            container_dct[index] = (current_sample & ~7) | bits_to_embed
            index += 1
    #
    # Apply inverse DCT
    #
    modified_audio = fftpack.idct(container_dct, norm='ortho')
    
    if normalized:
        modified_audio /= (2**15)
    
    return torch.tensor(modified_audio[:len(container_audio_signal)], dtype=torch.float), message_audio_data / (2**15) if normalized else message_audio_data
    

def dct_extract_audio_from_audio(audio, normalized=True):
    """
    Extract an audio message hidden in an audio signal using the discrete cosine transform (DCT) method.

    Args:
        audio (torch.tensor): The container audio signal.

    Returns:
        torch.tensor: The extracted audio message.
    """
    # lambda function for rounding --> Very important!
    rnd = lambda x: int(x) if abs(x) % abs(int(x)) < 0.99 else int(round(x))
    #
    # Apply DCT
    #
    audio_dct = fftpack.dct(audio.numpy(), norm='ortho')
    if normalized:
        audio_dct *= (2**15)
    #
    # Extract bits from the transformed coefficients
    #
    all_bits = [format(rnd(audio_dct[i]) & 7, '02b') for i in range(audio_dct.shape[0])]
    extracted_bits = ''
    for bit in all_bits:
        while len(bit) < 3:
            bit = "0" + bit
        extracted_bits += bit
    #
    # Convert the bits back to audio samples
    #
    message_audio_data = []
    for i in range(0, len(extracted_bits), 18):
        if i + 18 > len(extracted_bits): break  # Ensure we don't go past the end
        # positive
        if extracted_bits[i:i+2] == "00":
            sample = max(-2**15, min(2**15-1, int(extracted_bits[i+2:i+18], 2)))
        # negative
        elif extracted_bits[i:i+2] == "11":
            sample = max(-2**15, min(2**15-2, int("-" + extracted_bits[i+2:i+18], 2)))
        else:
            break # we have reached the end flag
        message_audio_data.append(sample)
    #
    # Return
    #
    message_audio_data = torch.tensor(message_audio_data, dtype=torch.int16)
    
    if normalized:
        return message_audio_data / (2**15)
    
    return message_audio_data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#           Spread Spectrum Methods
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def ss_lsb_embed_text_into_audio(audio_signal, msg, normalized=True, key=12):
    """
    Embed a text message into an audio file using the spread spectrum least significant bit method.

    Args:
        audio_path (str): the tensor audio file. 
        msg (str): the secret message to embed.
        normalized (bool): if true, the audio is normalized; else, not normalized.
        key (int) : random number generator key

    Returns: 
        None. The container audio file which contains the secret message is saved to output_path.
    """
    np.random.seed(key)
    #
    # Un-normalize the audio signal if needed
    # and make byte array
    #
    modified_audio_data = audio_signal
    if normalized:
        modified_audio_data = (audio_signal*(2**15)).to(torch.int16)
    #
    # Convert the message to binary and add a flag
    #
    binary_msg = ''.join(format(ord(i), '08b') for i in msg)
    binary_msg += '00000000'
    #
    # Determine indices to embed message bits using pseudo-random sequence
    #
    l = min(len(binary_msg), len(modified_audio_data)) # make sure is not longer than actual audio file.
    indices = np.random.choice(range(len(modified_audio_data)), l, replace=False)
    #
    # Embed the message
    #
    for idx, bit in zip(indices, binary_msg):
        # Modify the least significant bit of the audio sample
        current_byte = modified_audio_data[idx]
        modified_audio_data[idx] = (current_byte & ~1) | int(bit)

    if normalized:
        return modified_audio_data/(2**15)
        
    return modified_audio_data

def ss_lsb_extract_text_from_audio(audio_signal, normalized=True, key=12):
    """
    Extract a secret message hidden in an audio signal with spread spectrum LSB.

    Args:
        audio_signal (torch.tensor): the tensor audio file.
        normalized (bool): if true, the data is normalized; else, not normalized.

    Returns: 
        message (str): message hidden in the audio signal
    """
    np.random.seed(key)
    #
    # Un-normalize the audio signal if needed
    # and make byte array
    #
    modified_audio_data = audio_signal
    if normalized:
        modified_audio_data = (audio_signal*(2**15)).to(torch.int16)
    #
    # Determine indices to embed message bits using pseudo-random sequence
    #
    indices = np.random.choice(range(len(modified_audio_data)), len(modified_audio_data), replace=False)
    #
    # Extract bits from the least significant bits of the audio samples
    #
    extracted_bits = [int(modified_audio_data[i] & 1) for i in indices]
    #
    # Group the bits into bytes and convert to characters
    #
    message = ''
    for i in range(0, len(extracted_bits), 8):
        byte_as_str = ''.join(str(bit) for bit in extracted_bits[i:i+8])
        if byte_as_str == '00000000':  # Check for the delimiter indicating the end of the message
            break
        message += chr(int(byte_as_str, 2))

    return message
    
def ss_dwt_embed_text_into_audio(container_audio_signal, message, alpha=0.05, key=12):
    """
    Embed a text secret into an audio signal using spread spectrum steganography in the dwt.

    Args:
        container_audio_signal (torch.Tensor): The tensor of the container audio signal.
        message (str): The text secret to embed.
        key (int): The seed for the pseudo-random sequence generator.
        alpha (float): st
        normalized (bool): If true, audio is normalized; else, not normalized.

    Returns:
        torch.Tensor: The modified container audio signal with the embedded message.
    """
    np.random.seed(key)
    #
    # Convert to dwt
    #
    cA, cD = pywt.dwt(container_audio_signal.numpy(), 'db2')
    #
    # Convert message into binary
    #
    binary_msg = ''.join(format(ord(i), '08b') for i in message) + '00000000'
    #
    # Determine indices to embed message bits using pseudo-random sequence
    #
    indices = np.random.choice(range(len(cD)), len(binary_msg), replace=False)
    #
    # Embed the message
    #
    for idx, bit in zip(indices, binary_msg):
        # Here we add a very slight alteration
        cD[idx] += (1 if bit == '1' else -1) * alpha  # Alteration magnitude is an example
    #
    # Apply inverse DWT
    #
    audio_create = pywt.idwt(cA, cD, 'db2')
    
    return torch.Tensor(audio_create[:container_audio_signal.shape[0]])

def ss_dwt_extract_text_from_audio(audio, key=12):
    """
    Extract a text secret from an audio signal using spread spectrum steganography.

    Args:
        audio (torch.Tensor): The container audio signal.
        key (int): The seed for the pseudo-random sequence generator.

    Returns:
        str: The extracted text secret.
    """
    #
    # set key
    #
    np.random.seed(key)
    #
    # DWT Transforom
    #
    cA, cD = pywt.dwt(audio.numpy(), 'db2')
    #
    # Select indices from key
    #
    indices = np.random.choice(range(len(cD)), len(cD), replace=False)
    #
    # Extract the bits
    #
    extracted_bits = ['1' if cD[idx] > 0 else '0' for idx in indices]
    #
    # Group the bits into bytes and convert to characters
    #
    message = ''
    for i in range(0, len(extracted_bits), 8):
        byte_as_str = ''.join(str(bit) for bit in extracted_bits[i:i+8])
        if byte_as_str == '00000000':  # Check for the delimiter indicating the end of the message
            break
        message += chr(int(byte_as_str, 2))

    return message


def ss_dwt_embed_audio_into_audio(container_audio_signal, message_audio_signal, alpha=0.05, key=12, normalized=True):
    """
    Embed an audio signal secret into another audio signal using spread spectrum steganography in the dwt.

    Args:
        container_audio_signal (torch.Tensor): The tensor of the container audio signal.
        message_audio_signal (str): The tensor of the secret audio signal.
        key (int): The seed for the pseudo-random sequence generator.
        alpha (float): strength
        normalized (bool): If true, audio is normalized; else, not normalized.

    Returns:
        torch.Tensor: The modified container audio signal with the embedded message.
        torch.Tensor: the shortened secret if the intended secret is too big.
    """
    np.random.seed(key)
    #
    # Convert to dwt
    #
    cA, cD = pywt.dwt(container_audio_signal.numpy(), 'db2')
    if normalized:
        message_audio_signal = (message_audio_signal*(2**15)).to(torch.int16)
    #
    # Make secret smaller if need be
    #
    if message_audio_signal.shape[0] > (cD.shape[0]//2)//18-1:
        message_audio_signal = message_audio_signal[:(cD.shape[0]//2)//18-1]
        print("Shortening secret ...\n")
    #
    # Turn message into binary message
    #
    binary_msg = [format(sample, '016b') for sample in message_audio_signal]
    binary_msg.append("0"*16)
    #
    # Determine indices to embed message bits using pseudo-random sequence
    #
    indices = np.random.choice(range(len(cD)), len(cD), replace=False)
    #
    # Embed the message
    #
    index = 0
    for i, bin in enumerate(binary_msg):
        # deal with negatives
        if bin[0] == "-":
            if len(bin) == 17:
                bin = "11" + bin[1:]
            else:
                bin = "110" + bin[1:]
        # for the flag
        elif i == len(binary_msg)-1:
            bin = "10" + bin 
        # deal with positives
        else:
            # add non-positive flag to front
            bin = "00" + bin
        #
        # encode
        #
        for k in range(len(bin)):
            cD[indices[index]] += (1 if bin[k] == '1' else -1) * alpha  # Alteration magnitude is an example
            index += 1
    #
    # Apply inverse DWT
    #
    audio_create = pywt.idwt(cA, cD, 'db2')
    
    return torch.Tensor(audio_create[:container_audio_signal.shape[0]]), message_audio_signal/(2**15)

def ss_dwt_extract_audio_from_audio(audio, key=12, normalized=True):
    """
    Extract an audio secret from an audio signal using spread spectrum steganography with dwt.

    Args:
        audio (torch.Tensor): The container audio signal.
        key (int): The seed for the pseudo-random sequence generator.

    Returns:
        torch.Tensor: The extracted audio secret
    """
    #
    # set key
    #
    np.random.seed(key)
    #
    # DWT Transforom
    #
    cA, cD = pywt.dwt(audio.numpy(), 'db2')
    #
    # Select indices from key
    #
    indices = np.random.choice(range(len(cD)), len(cD), replace=False)
    #
    # Extract the bits
    #
    extracted_bits = ['1' if cD[idx] > 0 else '0' for idx in indices]
    #
    # Group the bits into bytes and convert to characters
    #
    message_audio_data = []
    for i in range(0, len(extracted_bits), 18):
        if i + 18 > len(extracted_bits): break  # Ensure we don't go past the end
        signal = ''.join(extracted_bits[i:i+18])
        # positive
        if signal[:2] == "00":
            sample = max(-2**15, min(2**15-1, int(signal[2:], 2)))
        # negative
        elif signal[:2] == "11":
            sample = max(-2**15, min(2**15-2, int("-" + signal[2:], 2)))
        else:
            break # we have reached the end flag
        message_audio_data.append(sample)

    message_audio_data = torch.tensor(message_audio_data, dtype=torch.int16)
    if normalized:
        return message_audio_data / (2**15)

    return message_audio_data


def ss_dct_embed_text_into_audio(container_audio_signal, message, key, alpha, normalized=True):
    """
    Embed a text secret into an audio signal using spread spectrum steganography.

    Args:
        container_audio_signal (torch.Tensor): The tensor of the container audio signal.
        message (str): The text secret to embed.
        key (int): The seed for the pseudo-random sequence generator.
        normalized (bool): If true, audio is normalized; else, not normalized.

    Returns:
        torch.Tensor: The modified container audio signal with the embedded message.
    """
    np.random.seed(key)
    
    # Perform DCT
    container_dct = dct(container_audio_signal.numpy(), norm='ortho')
    # if normalized:
    #     container_dct *= (2**15)

    # Convert message into binary
    binary_msg = ''.join(format(ord(i), '08b') for i in message) + '00000000'
    
    # Determine indices to embed message bits using pseudo-random sequence
    indices = np.random.choice(range(len(container_dct)), len(binary_msg), replace=False)
    
    # Embed the message
    for idx, bit in zip(indices, binary_msg):
        # Here we add a very slight alteration
        container_dct[idx] += (1 if bit == '1' else -1) * alpha  # Alteration magnitude is an example
    
    # Apply inverse DCT
    modified_audio = idct(container_dct, norm='ortho')
    
    return torch.tensor(modified_audio[:len(container_audio_signal)], dtype=torch.float)

def ss_dct_extract_text_from_audio(audio, key, normalized=True):
    """
    Extract a text secret from an audio signal using spread spectrum steganography.

    Args:
        audio (torch.Tensor): The container audio signal.
        key (int): The seed for the pseudo-random sequence generator.

    Returns:
        str: The extracted text secret.
    """
    np.random.seed(key)
    
    # Apply DCT
    audio_dct = dct(audio.numpy(), norm='ortho')
    # if normalized:
    #     audio_dct *= (2**15)

    message_length = 2000 ## set as max value
    indices = np.random.choice(range(len(audio_dct)), len(audio_dct), replace=False)
    #
    # Extract the bits
    #
    extracted_bits = ['1' if audio_dct[idx] > 0 else '0' for idx in indices]
    #
    # Group the bits into bytes and convert to characters
    #
    message = ''
    for i in range(0, len(extracted_bits), 8):
        byte_as_str = ''.join(str(bit) for bit in extracted_bits[i:i+8])
        if byte_as_str == '00000000':  # Check for the delimiter indicating the end of the message
            break
        message += chr(int(byte_as_str, 2))

    return message

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#           Echo hiding -> very small hiding capacity. 
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _apply_echo(signal, delay, alpha=0.5):
    """
    Applies an echo effect of size alpha to an audio signal at distance delay.

    Args:
        signal (torch.Tensor) : the input audio.
        delay (int) : the delay for the echo in samples
        alpha (float) : The amplitude of the echo relative to the original (0 < alpha < 1).
        
    Returns:
        torch.Tensor: The signal with the echo effect applied.
    """
    #
    # Create the echo kernel
    # if delay = 4, -> k0 = [0,0,0,0,0.5]
    #
    k0 = torch.cat((torch.zeros(delay), torch.tensor([alpha])))
    #
    # Initialize the output signal tensor with zeros
    #
    output_signal = torch.zeros(signal.shape[0] + k0.shape[0] - 1)
    #
    # Apply the echo effect
    #
    for i in range(signal.shape[0]):
        output_signal[i:i+k0.shape[0]] += signal[i] * k0
    
    return output_signal[:signal.shape[0]]
    
def _hanning_window(L):
    """
    Generates a Hanning window using PyTorch.
    """
    if L == 1:
        return torch.tensor([1.0])
    elif L > 1:
        n = torch.arange(0, L)
        return 0.5 * (1 - torch.cos(2 * np.pi * n / (L - 1)))
    else:
        raise ValueError("L must be greater than zero")

def _mixer(L, bits, lower, upper, K=None):
    """Generates a smoothed signal from binary data using PyTorch."""
    if K is None or 2 * K > L:
        K = L // 4 - (L // 4) % 4
    else:
        K = K - K % 4
    
    # bits is already a binary sequence string like "0101010101010"
    N = len(bits)
    
    # Create the mixer signal
    encbit = torch.tensor([float(bit) for bit in bits], dtype=torch.float32)
    m_sig = encbit.repeat_interleave(L+20)
    
    # Apply Hanning window
    hann_win = _hanning_window(K)
    c = torch.nn.functional.conv1d(m_sig.view(1, 1, -1), hann_win.view(1, 1, -1), padding=K//2)
    wnorm = c[0, 0, K//2:-(K//2)+1] / torch.max(torch.abs(c))
    
    # Adjusting bounds
    w_sig = wnorm * (upper - lower) + lower
    m_sig = m_sig * (upper - lower) + lower
    
    return w_sig, m_sig

def echo_hide_embed_text_into_audio(cover, secret, d0=75, d1=200, L=1024, alpha=0.5):
    #
    # Convert secret text into binary
    #
    binary_msg = ''.join(format(ord(char), '08b') for char in secret)
    binary_msg += '00000000'  # End flag
    N = len(binary_msg)
    #
    # Get echo for zero bits and one bits
    #
    echo_zro = _apply_echo(cover, d0, alpha)
    echo_one = _apply_echo(cover, d1, alpha)
    mix_w, mix_m = _mixer(L, binary_msg, 0, 1, 256)
    #
    # Create container
    #
    embed_signal = cover[:N*L] + echo_zro[:N*L]*torch.abs(mix_m[:N*L]-1) + echo_one[:N*L]*mix_m[:N*L]
    container = torch.cat((embed_signal, cover[N*L:])) # combine the rest of the cover signal

    return container


def echo_hide_extract_text_from_audio(signal, d0, d1, L, len_msg=None):
    """
    Extracts a hidden message from an audio signal using real cepstrum analysis.

    Parameters:
    - signal: A PyTorch tensor representing the audio signal.
    - L: The length of each frame in the signal.
    - d0, d1: The delays used for encoding '0's and '1's, respectively.
    - len_msg: Optional; the length of the original hidden message.

    Returns:
    - The extracted hidden message as a string.
    """    
    N = signal.shape[0] // L

    extracted_bits = []
    for k in range(L, signal.shape[0], L):
        frame = signal[k-L:k]
        spectrum = torch.fft.fft(frame)
        log_spectrum = torch.log(torch.abs(spectrum))
        rceps = torch.fft.ifft(log_spectrum).real
        bit = '0' if rceps[d0] >= rceps[d1] else '1'
        extracted_bits.append(bit)
    #
    # Group the bits into bytes and convert to characters
    #
    message = ''
    for i in range(0, len(extracted_bits), 8):
        byte_as_str = ''.join(str(bit) for bit in extracted_bits[i:i+8])
        if byte_as_str == '00000000':  # Check for the delimiter indicating the end of the message
            break
        message += chr(int(byte_as_str, 2))

    return message

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#                        Phase coding --> small capacity. 
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def phase_code_embed_text_into_audio(audio, secret, normalized=True):
    """
    Embed a text secret into an audio signal using phase coding.

    Note: max secret size is scale of 1024 compared to audio. i.e. 
    max_length_of_secret = len(audio)/1024

    Args:
        audio (torch.Tensor): The tensor of the container audio signal.
        secret (str): The text secret to embed.
        normalized (bool): If true, audio is normalized; else, not normalized.

    Returns:
        torch.Tensor: The modified container audio signal with the embedded message.
    """
    #
    # un-normalize audio [-1, 1]
    #
    if normalized:
        audio_data = audio * (2**15)
    #
    # Get chunk size and number of chunks
    #
    if len(audio_data)/len(secret) < 1024:   # secret is too big
        print("\nShortening secret ...")
        secret = secret[:len(audio_data)//1024]
    secret = secret.ljust(100, '~')  # Add ending flag
    text_length = 8 * len(secret)  # Binary text secret length (8 bits per char)
    chunk_size = int(2 * 2 ** math.ceil(math.log2(2 * text_length)))
    num_of_chunks = int(math.ceil(audio_data.numel() / chunk_size))
    #
    # Resize and view operations
    #
    padded_size = num_of_chunks * chunk_size     # if audio is currently too short
    if audio_data.numel() < padded_size:
        audio_data = torch.cat((audio_data, torch.zeros(padded_size - audio_data.numel())))
    chunks = audio_data.view(num_of_chunks, chunk_size)  # reshape to tensor chunks
    #
    # Applying FFT
    #
    chunks_fft = torch.fft.fft(chunks)
    magnitudes = torch.abs(chunks_fft)
    phases = torch.angle(chunks_fft)
    phase_diff = torch.diff(phases, dim=0)
    #
    # Convert text to pi adjustments
    #
    text_in_pi = torch.tensor([(-1.0 if bit == '0' else 1.0) for char in secret for bit in format(ord(char), '08b')]) * (-math.pi / 2)
    mid_chunk = chunk_size // 2
    #
    # Compute phase matrix
    #
    phases[0, mid_chunk - text_length : mid_chunk] = text_in_pi
    phases[0, mid_chunk + 1 : mid_chunk + 1 + text_length] = -text_in_pi.flip(dims=[0])
    for i in range(1, phases.size(0)):
        phases[i] = phases[i - 1] + phase_diff[i - 1]
    #
    # Reverse the process
    #
    chunks_fft = magnitudes * torch.exp(1j * phases)
    audio_data = torch.fft.ifft(chunks_fft).real
    audio_data = audio_data.view(-1)
    
    return audio_data / (2**15) if normalized else audio_data

    

def phase_code_extract_text_from_audio(container, normalized=True):
    """
    Extract a text secret from an audio signal using phase coding.

    Args:
        container (torch.Tensor): The container audio signal.
        normalized (bool): If true, audio is normalized; else, not normalized.
        
    Returns:
        str: The extracted text secret.
    """
    #
    # Convert audio to numpy and unnormalize
    #
    if normalized:
        audio_data = container*(2**15)
    text_length = 800
    block_length = 2 * int(2 ** math.ceil(math.log2(2 * text_length)))
    block_mid = block_length // 2
    #
    # Get header info
    #
    code = audio_data[:block_length]
    #
    # Get the phase and convert it to binary
    #
    code_phases = torch.angle(torch.fft.fft(code))[block_mid - text_length:block_mid]
    bin = (code_phases < 0).to(torch.int16)
    #
    # Convert into characters
    #
    sq_map = 1 << torch.arange(7, -1, -1).short()
    code_int = [b.dot(sq_map).item() for b in bin.view((-1, 8))]
    #
    # Combine characters to original text and return
    #
    return "".join([chr(i) for i in code_int]).replace("~", "")