import os
# audio roadmap
class music_roadmap():
    def __init__(self, data_dir: str) -> None:
        """The music_roadmap class builds a dataframe from a directory to 
           tell a torch dataloader where to find audio files"""
        if not os.path.isdir(data_dir):
            raise ValueError(f"The input is not a valid directory: {data_dir}")
        self.data_dir = data_dir
        self.music_df = None
        self.status = None
        pass
    
    def __str__(self):
        return str(self.status)

    def __repr__(self) -> str:
        return "music_roadmap(directory:{})".format(self.data_dir)
    
    def create_music_record(self):
        pass