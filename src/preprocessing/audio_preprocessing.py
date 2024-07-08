#!/usr/bin/env python # [1]
"""
This script creates the dataframe containing music file metadata.
"""

# imports
# ------------------------------------------------------------------
import os
import glob
import pickle
import pandas as pd
# ------------------------------------------------------------------
# functions
# ------------------------------------------------------------------


def check_dict(py_dict, dict_key):
    """
    This function checks if a dictionary contains a key.
    if it contains the key, the value is returned,
    otherwise, a blank string is returned

    Args:
        py_dict: A python dictionary
        dict_key: A string indicating the key to be checked

    Returns:
        the value of the dictionary for the key if it exists
        otherwise, returns ''

    Raises:
        KeyError: Raises an exception if args are not proper format.
    """
    if dict_key in py_dict:
        val = py_dict[dict_key]
    else:
        val = ''
    return val


def find_matching_strings(string_list: list, target_string: str) -> list:
    """
    Check if a string contains a target

    Parameters:
    string_list (list of strings): List of strings to check against target
    string (str): the str that will be checked against the list

    Returns:
    return_type: list of matching strings

    Raises:
    None
    """
    # make case insensitive
    target_string_l = target_string.lower()
    strings_m = [s.lower()
                 for s in string_list
                 if s.lower() in target_string_l]
    return strings_m


def check_wav_files(wav_directory: str) -> bool:
    """
    Checks if there are any .wav files in the given directory
    and its subdirectories.

    Parameters:
    directory (str): The path to the directory to check.

    Returns:
    bool: True if there are .wav files in the directory or any of its
    subdirectories, False otherwise.

    Raises:
    ValueError: If the path is not a valid directory.
    """
    if not os.path.isdir(wav_directory):
        raise ValueError(f"The path {wav_directory} is not a valid directory.")
    for _, _, wav_files in os.walk(wav_directory):
        for file in wav_files:
            if file.endswith('.wav'):
                return True
    return False


def get_wav_files(wav_file_directory: str) -> str:
    """
    Gets the wav file in a directory

    Parameters:
    wav_file_directory, a directory

    Returns:
    wav_file, a string indicating the wav file

    Raises:
    None
    """
    wav_files = glob.glob(os.path.join(wav_file_directory, '**', '*.wav'),
                          recursive=True)
    # skip if more/less than 1 file found
    if len(wav_files) == 1:
        wav_file = wav_files[0]
    else:
        wav_file = None
    return wav_file


def check_metadata(file_path: str) -> bool:
    """
    Checks if there is a sound_metadata.pkl file in the directory.

    Parameters:
    directory (str): The path to the directory to check.

    Returns:
    bool: True if there is a sound_metadata.pkl in the directory or any of its
    subdirectories, False otherwise.

    Raises:
    ValueError: If the path is not a valid directory.
    """
    if not os.path.isdir(os.path.dirname(file_path)):
        raise ValueError(f"The path {file_path} is not a valid directory.")
    # check if file exists:
    metadata_exists = os.path.isfile(file_path)
    return metadata_exists


def create_music_record(dir_path: str, instrument_list: list) -> dict:
    """
    Create a record for a dataframe from a file path

    Args:
        dir_path: string containing freesound file path
        instrument_list: list contraining valid instrument names

    Returns:
        music_record, a dictionary containing file information

    Raises:
        KeyError: Raises an exception if args are not proper format.
    """
    # check if wav file is in directory
    has_wav_files = check_wav_files(dir_path)
    # get sound_metadata
    metadata_file = os.path.join(dir_path, 'sound_metadata.pkl')
    has_metadata = check_metadata(metadata_file)
    if has_wav_files is False:
        print('No wav files found')
        return None
    if has_metadata is False:
        print('No metadata file found')
        return None
    wav_file = get_wav_files(dir_path)
    with open(metadata_file, 'rb') as file:
        sound_metadata = pickle.load(file)
    # format metadata
    try:
        sound_name = sound_metadata['name']
        found_instruments = find_matching_strings(instrument_list,
                                                  sound_name)
        if len(found_instruments) == 1:
            sound_instr = found_instruments[0]
        else:
            # default to blank and investigate later
            sound_instr = ''
    except IndexError as e:
        print(e)
        sound_instr = ''
    music_record = {"relative_path": wav_file,
                    "channels": check_dict(sound_metadata, "channels"),
                    "filesize": check_dict(sound_metadata, "filesize"),
                    "bitrate": check_dict(sound_metadata, "bitrate"),
                    "bitdepth": check_dict(sound_metadata, "bitdepth"),
                    "duration": check_dict(sound_metadata, "duration"),
                    "samplerate": check_dict(sound_metadata, "samplerate"),
                    "instrument_name": sound_instr}
    return music_record


def create_music_set(dir_list: list[str]) -> pd.DataFrame:
    """
    Creates the dataframe containing information about music files

    Parameters:
    dir_list (list of strings): list of directories containing audio files and
    metadata

    Returns:
    music_df: description

    Raises:
    None
    """
    music_list = [create_music_record(dir_path=dir_,
                                      instrument_list=INSTRUMENT_LIST)
                  for dir_ in dir_list]
    # filter
    music_fltrd = [i for i in music_list if i is not None]
    music_df = pd.DataFrame(music_fltrd)
    return music_df
# ------------------------------------------------------------------


# define global variables
# ------------------------------------------------------------------
INSTRUMENT_LIST = ['Clarinet',
                   'Sax Alto',
                   'Flute',
                   'Violin',
                   'Trumpet',
                   'Cello',
                   'Sax Tenor',
                   'Piccolo',
                   'Sax Soprano',
                   'Sax Baritone',
                   'Oboe',
                   'Double Bass',
                   ]

# ------------------------------------------------------------------


if __name__ == "__main__":
    fs_dirs = []
    for root, dirs, files in os.walk("./freesound"):
        for directory in dirs:
            fs_dirs.append(os.path.join(root, directory))
    print(len(fs_dirs))
    music_file_df = create_music_set(fs_dirs)
    print(music_file_df.head())
