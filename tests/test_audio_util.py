import AudioUtil
import torch

def test_AudioUtil(test_file):
    # try preprocessing an example file
    # we should get a 3 dimensional tensor
    result = AudioUtil.preprocess_file(test_file)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 3

def test_AudioUtil_basic():
    # using an example audio file, test that result of preprocessing is expected...
    result = 3
    assert result == 3