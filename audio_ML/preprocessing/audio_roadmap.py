import os
import glob
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from audio_ML.config import ml_config

# audio roadmap
class freesound_audio():
    def __init__(self,file_path: str, config: ml_config) -> None:
        """the freesound_audio class takes a file path and extracts information 
        into a dictionary"""
        self.file_path = file_path
        self.data_dir = os.path.dirname(file_path)
        self.metadata_path = os.path.join(self.data_dir,'sound_metadata.pkl')
        self.metadata_exists = os.path.isfile(self.metadata_path)
        self.sound_metadata = self.get_metadata()
        self.augment_metadata()
        

        # add config variables as attrs
        for attr in dir(config):
            if not attr.startswith("__"):  # Skip special/private attributes
                setattr(self, attr, getattr(config, attr))

        # future: throw error if instrument_list not in config
        self._parse_instrument()
        

    def __str__(self):
        return str(self.file_path)

    def __repr__(self) -> str:
        return "freesound_audio(file:{})".format(self.file_path)
    
    def get_metadata(self) -> dict:
        if self.metadata_exists:
            with open(self.metadata_path, 'rb') as file:
                self.sound_metadata = pickle.load(file)
        else:
            self.sound_metadata = {}
        return self.sound_metadata
    
    def augment_metadata(self) -> dict:
        if self.metadata_exists:
            self.sound_metadata['file_path'] = self.file_path
        return self.sound_metadata
    
    def _parse_instrument(self) -> str:
        """Given a file name, return a matching instrument
        Args:
            file_name(str): the file name which may contain an instrument
        Example usage: if the input is 
            overall quality of single note - trumpet - D#5.wav,
            result will be "trumpet" if trumpet is in the provided config
            Note: currently this assumes each file has only 1 label
                this won't work as written for multi-label"""
        
        for i in self.INSTRUMENT_LIST:
            if i.lower() in self.file_path.lower():
                self.sound_metadata['target_instrument'] = i
                return i
            self.sound_metadata['target_instrument'] = ''
        return ''

class music_roadmap():
    def __init__(self, data_dir: str, ml_config: ml_config) -> None:
        """The music_roadmap class builds a dataframe from a directory to 
           tell a torch dataloader where to find audio files
           
           Args: 
            data_dir(str) a string representing the data directory containing
                music files
            instrument_list(list[str]): a list containing all of the targets 
                for our classifier
           """
        if not os.path.isdir(data_dir):
            raise ValueError(f"The input is not a valid directory: {data_dir}")
        self.data_dir = data_dir
        self.ml_config = ml_config
        self.music_df = None
        self.processed_df = None
        self.records = 0
        pass
    
    def __str__(self):
        return str(self.records)

    def __repr__(self) -> str:
        return "music_roadmap(directory:{})".format(self.data_dir)
    
    def create_file_list(self,file_ext = ".wav") -> list:
        """Using the instance data directory, create a list of files of 
        the given type
        
        Args: file_ext(str) the file extension designating type of file"""

        self.file_list = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(file_ext):
                    self.file_list.append(os.path.join(root, file))
        return self.file_list

    def create_data_list(self) -> list:
        fl = self.file_list
        ml_config = self.ml_config
        # loop through files and get metadata
        self.data_list = [freesound_audio(file_,ml_config).sound_metadata for file_ in fl]
        return self.data_list
    
    def create_audio_df(self) -> pd.DataFrame:
        self.audio_df = pd.DataFrame(self.data_list)
        return self.audio_df

    def clean_audio_df(self) -> pd.DataFrame:
        """
        Cleans music dataframe by removing records where target is missing

        Args:
            audio_df (pd.DataFrame): initial df containing file info

        Returns:
            audio_df_clean (pd.DataFrame): processed df with no blank
            instrument_names

        Raises:
            empty dataframe: raises error if returned dataframe is empty
        """
        missing_recs = self.audio_df.loc[self.audio_df['target_instrument'] == '']
        n_missing = missing_recs.shape[0]
        n_rows = self.audio_df.shape[0]
        pct_missing = round(10*n_missing/n_rows,2)
        print(f"""Records missing target variable: {n_missing}.
            Removing  {pct_missing}% of records from our data""")
        audio_df_clean = self.audio_df.loc[self.audio_df['target_instrument'] != '']
        self.audio_df_clean = audio_df_clean
        try:
            assert not audio_df_clean.empty, "DataFrame is empty!"
            # log success
            print('dataframe has records')
        except AssertionError as e:
            # log error
            print(e)
        return self.audio_df_clean
    
    def add_target_to_df(self,var_name: str) -> pd.DataFrame:
        encoder = LabelEncoder()
        df = self.audio_df_clean
        df['target'] = encoder.fit_transform(df[var_name]).astype('int64')
        self.processed_df = df
        return self.processed_df
    
    def save_df(self) -> None:
        print('saving roadmap!')
        self.processed_df.to_csv('./data/interim/audio_roadmap.csv')
        pass

    def roadmap_diagnostics(self) -> None:
        """placeholder: ensure roadmap passes tests"""
        pass