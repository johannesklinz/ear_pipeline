import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import datetime
import os
from tqdm import tqdm
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
log_msg = ""
logging.basicConfig(format='%(message)s', level=logging.INFO)

class ForegroundDataset(Dataset):
    def __init__(
            self,
            audio_paths_full:list[str]
    ):
        """
        GLOBAL VARIABLES
        ------------------------------
        audio_paths_full:list[str]  = A list that contains all wav-files that should be processed.
        dataset:list[str]           = See audio_paths_full.
        melbins:int                 =
        target_length:int           =
        norm_mean:float             =
        norm_std:float              =
        sample_rate                 = Refers to the sampling rate (aka maximum frequency) that the pipeline
                                      can process.

        DESCRIPTION
        ------------------------------
        Initializer
        """
        self.dataset = audio_paths_full
        self.melbins = 128
        self.target_length = 3072
        self.norm_mean = -7.625
        self.norm_std = 2.36
        self.sample_rate = 16000

    def __len__(
            self
    ):
        """
        ARGUMENTS
        ------------------------------
        None

        RETURNS
        ------------------------------
        len(self.dataset):int       = Refers to the number of to be processed audio files.

        DESCRIPTION
        ------------------------------
        This function  returns the length of the dataset and is mandatory as per PyTorch's API.
        """
        return len(self.dataset)

    def __getitem__(
            self,
            item:int
    ):
        """
        ARGUMENTS
        ------------------------------
        item:int                    = Refers to the item-index that should be loaded from the dataset.

        RETURNS
        ------------------------------
        (infer:int,                 = infer(0, 1) indicates whether a file could be loaded appropriately.
        fbank:torch.Tensor)         = fbank is the filterbank that is processed by the FG algorithm.

        DESCRIPTION
        ------------------------------
        This function returns one specific item from the dataset and is mandatory as per PyTorch's API.
        """
        global log_msg

        try:
            waveform, sr = torchaudio.load(self.dataset[item])
            if sr != self.sample_rate:
                transform = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = transform(waveform)

            waveform = waveform - waveform.mean()
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.sample_rate, use_energy=False,
                                                    window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

            p = self.target_length - fbank.shape[0]
            if p>0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif p<0:
                start_id = np.random.randint(0, -p)
                fbank = fbank[start_id: start_id+self.target_length, :]
        
            fbank = (fbank-self.norm_mean)/(self.norm_std*2)
            fbank = torch.transpose(fbank, 0, 1).unsqueeze(0)

            return (1, fbank)
        except Exception as e:
            log_msg += f"\n\nFile {self.dataset[item].split('/')[-1]} could not be transcribed with the following error: {' '.join(str(e).split()[:20])}"
            fbank = torch.zeros(1, self.melbins, self.target_length)
            return (0, fbank)
        
class EarPipeline:

    def __init__(
            self,
    ):
        """
        GLOBAL VARIABLES
        ------------------------------
        None
        DESCRIPTION
        ------------------------------
        Initializer
        """
    
    def get_audio_paths(
            self,
            dir:str
    ):
        """
        ARGUMENTS
        ------------------------------
        dir:str                     = Points to a directory that contains wav-files to be processed.

        RETURNS
        ------------------------------
        audio_paths:list[str]       = A list that contains the complete paths for every audio file that
                                      was found within the respective directory.

        DESCRIPTION
        ------------------------------
        This function checks whether a user inputed directory indeed contains audio files and, if so, 
        summarizes them into a list.
        """
        global log_msg

        if not os.path.isdir(dir): 
            log_msg += f"\n\n{dir} does not point to a directory and is therefore skipped!"
            return []
        audio_paths = [f"{dir}/{file}" for file in os.listdir(dir) if file.endswith("wav")]
        if not len(audio_paths): 
            log_msg += f"\n\n{dir} does not contain any wav-files and is therefore skipped!"
            return [] 

        return audio_paths
    
    def run_foreground_inference(
            self, 
            audio_paths_full:list[str]):
        """
        ARGUMENTS
        ------------------------------
        audio_paths_full:list[str]  = A list that contains all wav-files that should be processed.

        RETURNS
        ------------------------------
        posteriors:list[float]      = A list that contains the posteriors that there is language.
        labels:list[float]          = A list that denotes the final label (0 = no speech, 1 = speech).
        
        DESCRIPTION
        ------------------------------
        This function applies the Foreground Speech Detection algorithm to the wav-files and saves its
        results accordingly.
        """
        global log_msg, device

        foreground_model = torch.load("models/foreground.pt", map_location=device).module
        foreground_model.eval()
        foreground_dataloader = DataLoader(ForegroundDataset(audio_paths_full), batch_size=32)

        smax = torch.nn.Softmax(dim=1)
        posteriors = []
        labels = []

        log_msg += f"\n\nFG Inference startet at {datetime.datetime.now().time()} for {len(audio_paths_full)} files."
        logging.info("FG Inference starts...")

        with torch.no_grad():
            for infer, audio_input in tqdm(foreground_dataloader):
                mask_idx = [i for i in range(len(infer)) if infer[i] == 1]
                audio_input = audio_input[mask_idx]
                audio_input = audio_input.to(device)
                audio_output = smax(foreground_model(audio_input))
                infer_posteriors = audio_output[:, 1].cpu().numpy()
                temp_posteriors = [np.nan]*len(infer)
                infer_labels = torch.argmax(audio_output, dim=1).cpu().numpy()
                temp_labels = [np.nan]*len(infer)
                for iloc, infer_posterior, infer_label in zip(mask_idx, infer_posteriors, infer_labels):
                    temp_posteriors[iloc] = infer_posterior
                    temp_labels[iloc] = infer_label
                posteriors = [*posteriors, *temp_posteriors]
                labels = [*labels, *temp_labels]

        log_msg += f"\n\nFG Inference ended at {datetime.datetime.now().time()}"

        return posteriors, labels

    def stack_save_results(
            self,
            audio_paths_full:list[str],
            posteriors:list[float],
            labels:list[float],
            output_save_path:str
    ):
        """
        ARGUMENTS
        ------------------------------
        audio_paths_full:list[str]  = A list that contains all wav-files that should be processed.
        posteriors:list[float]      = A list that contains the posteriors that there is language.
        labels:list[float]          = A list that denotes the final label (0 = no speech, 1 = speech).
        output_save_path:str            = Refers to the path under which the final Excel-Output shall be saved.

        RETURNS
        ------------------------------
        None
        
        DESCRIPTION
        ------------------------------
        This function combines all the extracted information and saves them to a user-defined location.
        """
        global log_msg

        audio_file_name = [file_path.split("/")[-1] for file_path in audio_paths_full]
        
        df = pd.DataFrame({
            "file_name": audio_file_name,
            "file_path": audio_paths_full,
            "fg_posterior": posteriors,
            "fg_label": labels,
        })

        df.to_excel(output_save_path)
        log_msg += f"\n\nThe EAR-Pipeline is finished and saved the output file to {output_save_path}!"
        
        

    def inference(
            self,
            audio_dirs:list[str],
            output_save_path:str,
            log_save_path:str
    ):
        """
        ARGUMENTS
        ------------------------------
        audio_dirs:list[str]        = Contains all the directories under which the programm looks for wav-files.
        output_save_path:str        = Refers to the path under which the final Excel-Output shall be saved.
        log_save_path:str           = Refers to the path under which the Log.txt shall be saved.

        RETURNS
        ------------------------------
        None
        :list[str]
        
        DESCRIPTION
        ------------------------------
        This function runs all the computations and data wrangling steps in the necessary order.
        """
        global log_msg

        cur_time = datetime.datetime.now()
        log_msg = f"The Ear-Pipeline Inference (only FG) started on {cur_time.date()} at {cur_time.time()}!"

        audio_paths_full = []
        for audio_dir in audio_dirs:
            audio_paths_full = [*audio_paths_full, *self.get_audio_paths(audio_dir)]
        posteriors, labels = self.run_foreground_inference(audio_paths_full)
        self.stack_save_results(audio_paths_full, posteriors, labels, output_save_path)
        with open(log_save_path, "w") as log_txt:
            log_txt.write(log_msg)

def main(
        audio_dirs:list[str],
        output_save_path:str,
        log_save_path:str
):
    ear_pipeline = EarPipeline()
    ear_pipeline.inference(audio_dirs, output_save_path, log_save_path)
        

if __name__ == "__main__":
    main(
        audio_dirs=[
            ""
        ],
        output_save_path="",
        log_save_path=""
    )

