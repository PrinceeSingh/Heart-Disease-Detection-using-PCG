import torch
import librosa
print("Torch:", torch.__version__)
print("Librosa:", librosa.__version__)
print("CPU threads:", torch.get_num_threads())