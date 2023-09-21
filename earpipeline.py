import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import datetime
import os
import whisper
from tqdm import tqdm
import librosa
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
        whisper_languages:dict      = Contains all languages that Whisper is capable of transcribing.

        DESCRIPTION
        ------------------------------
        Initializer
        """
        self.whisper_languages = {
            "en": "english",
            "zh": "chinese",
            "de": "german",
            "es": "spanish",
            "ru": "russian",
            "ko": "korean",
            "fr": "french",
            "ja": "japanese",
            "pt": "portuguese",
            "tr": "turkish",
            "pl": "polish",
            "ca": "catalan",
            "nl": "dutch",
            "ar": "arabic",
            "sv": "swedish",
            "it": "italian",
            "id": "indonesian",
            "hi": "hindi",
            "fi": "finnish",
            "vi": "vietnamese",
            "he": "hebrew",
            "uk": "ukrainian",
            "el": "greek",
            "ms": "malay",
            "cs": "czech",
            "ro": "romanian",
            "da": "danish",
            "hu": "hungarian",
            "ta": "tamil",
            "no": "norwegian",
            "th": "thai",
            "ur": "urdu",
            "hr": "croatian",
            "bg": "bulgarian",
            "lt": "lithuanian",
            "la": "latin",
            "mi": "maori",
            "ml": "malayalam",
            "cy": "welsh",
            "sk": "slovak",
            "te": "telugu",
            "fa": "persian",
            "lv": "latvian",
            "bn": "bengali",
            "sr": "serbian",
            "az": "azerbaijani",
            "sl": "slovenian",
            "kn": "kannada",
            "et": "estonian",
            "mk": "macedonian",
            "br": "breton",
            "eu": "basque",
            "is": "icelandic",
            "hy": "armenian",
            "ne": "nepali",
            "mn": "mongolian",
            "bs": "bosnian",
            "kk": "kazakh",
            "sq": "albanian",
            "sw": "swahili",
            "gl": "galician",
            "mr": "marathi",
            "pa": "punjabi",
            "si": "sinhala",
            "km": "khmer",
            "sn": "shona",
            "yo": "yoruba",
            "so": "somali",
            "af": "afrikaans",
            "oc": "occitan",
            "ka": "georgian",
            "be": "belarusian",
            "tg": "tajik",
            "sd": "sindhi",
            "gu": "gujarati",
            "am": "amharic",
            "yi": "yiddish",
            "lo": "lao",
            "uz": "uzbek",
            "fo": "faroese",
            "ht": "haitian creole",
            "ps": "pashto",
            "tk": "turkmen",
            "nn": "nynorsk",
            "mt": "maltese",
            "sa": "sanskrit",
            "lb": "luxembourgish",
            "my": "myanmar",
            "bo": "tibetan",
            "tl": "tagalog",
            "mg": "malagasy",
            "as": "assamese",
            "tt": "tatar",
            "haw": "hawaiian",
            "ln": "lingala",
            "ha": "hausa",
            "ba": "bashkir",
            "jw": "javanese",
            "su": "sundanese",
            }

    
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
    
    def run_whisper_inference(
            self,
            audio_paths_full:list[str]
    ):
        """
        ARGUMENTS
        ------------------------------
        audio_paths_full:list[str]  = A list that contains all wav-files that should be processed.

        RETURNS
        ------------------------------
        languages:list[str]         = A list that denotes the detected language for every wav-file.
        segments_english:list[tuple]= A list that contains the single segments for every wav-file in English.
        number_segments:list[int]   = A list that denotes the number of segments for every wav-file.
        speech_duration:list[float] = A list that denotes the length of every wav-file.
        segments_foreign:list[tuple]= If a wav-file's original language is not English, this list contains the
                                      segments in the wav-file's original language.
        
        DESCRIPTION
        ------------------------------
        This function applies OpenAI's Whisper-Medium-model to all wav-files, automatically detects the languages
        and processes the wav-file accordingly.
        """
        global log_msg

        whisper_model = whisper.load_model("medium", download_root="models/")
        languages = []
        segments_english = []
        segments_foreign = []
        number_segments = []
        speech_duration = []

        log_msg += f"\n\nWhisper Inference startet at {datetime.datetime.now().time()} for {len(audio_paths_full)} files."
        logging.info("Whisper Inference starts...")

        for audio in tqdm(audio_paths_full):
            try:
                audio = whisper.load_audio(audio)
                result = whisper.transcribe(model=whisper_model, audio=audio, task="translate")
                languages.append(self.whisper_languages[result["language"]])
                segments_english.append([(segment["start"], segment["end"], segment["text"]) 
                                            for segment in result["segments"]] if result["segments"] else [])
                number_segments.append(len(result["segments"]))
                speech_duration.append(np.sum([segment["end"]-segment["start"] for segment in result["segments"]]))
                if languages[-1] != "english":
                    result = whisper.transcribe(model=whisper_model, audio=audio, task="transcribe")
                    segments_foreign.append([(segment["start"], segment["end"], segment["text"]) 
                                            for segment in result["segments"]] if result["segments"] else [])
                else:
                    segments_foreign.append([])
            except Exception as e:
                log_msg += f"\n\nFile {audio.split('/')[-1]} could not be transcribed with the following error: {' '.join(str(e).split()[:20])}"
                languages.append("")
                segments_english.append([])
                number_segments.append(np.nan)
                speech_duration.append(np.nan)
                segments_foreign.append([])
        
        log_msg += f"\n\nWhisper Inference ended at {datetime.datetime.now().time()}"
        
        return languages, segments_english, number_segments, speech_duration, segments_foreign
    
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

    def get_audio_length(
            self,
            audio_paths_full:list[str]
    ):
        """
        ARGUMENTS
        ------------------------------
        audio_paths_full:list[str]  = A list that contains all wav-files that should be processed.


        RETURNS
        ------------------------------
        audio_lengths:list[float]   = A list that contains the length of every wav-file or np.nan if the given
                                      wav-file cannot be read.
        
        DESCRIPTION
        ------------------------------
        This is a helper function that extracts the length for every  wav-file.
        """
        global log_msg

        audio_lengths = []
        for audio_file in audio_paths_full:
            try:
                temp_dur = librosa.get_duration(path=audio_file)
                audio_lengths.append(temp_dur)
            except Exception as e:
                log_msg += f"\n\nLibrosa cannot open {audio_file.split('/')[-1]} to determine sample length with the following error: {' '.join(str(e).split()[:20])}"
                audio_lengths.append(np.nan)
        
        return audio_lengths
    
    def unstack_segments(
            self,
            transcript_segments:list[tuple],
            timestamps:bool
    ):
        """
        ARGUMENTS
        ------------------------------
        transcript_segments         = A list that contains all the segments for a given transcript as tuples. 
        :list[tuple]
        timestamps:bool             = Denotes whether the unstacked unstacked segment should contain timestamps
                                      or not.

        RETURNS
        ------------------------------
        unstack_segments            = A list that contains the unstacked transcripts as strings with timestamps
                                      for all the single segments.
        :list[str]
        
        DESCRIPTION
        ------------------------------
        This function is solely for data preparation and unstacks the segments so that they can be better printed
        to an Excel-file later.
        """
        unstack_segments = []
        for transcript_segment in transcript_segments:
            temp_str_transcript_segment = ""
            if timestamps:
                for segment in transcript_segment:
                    temp_str_transcript_segment += f"[{segment[0]} -> {segment[1]}]\n{segment[2]}\n\n"
            else:
                for segment in transcript_segment:
                    temp_str_transcript_segment += f"{segment[2]}\n\n"
            unstack_segments.append(temp_str_transcript_segment.rstrip())
        
        return unstack_segments
    
    def create_whisper_label(
            self,
            segments_english:list[str],
            posteriors:list[float]
    ):
        """
        ARGUMENTS
        ------------------------------
        segments_english:list[tuple]= A list that contains the single segments for every wav-file in English.
        posteriors:list[float]      = A list that contains the posteriors that there is language.

        RETURNS
        ------------------------------
        whisper_label:list[float]   = A list that indicates whether Whisper detected language (1), not detected
                                      language (0) or was not able to open the given wav-file (np.nan).
        
        DESCRIPTION
        ------------------------------
        This is a helper function that extracts the Whisper label value for every processed wav-file.
        """
        whisper_label = []
        for iloc, transcript in enumerate(segments_english):
            if not len(transcript) and np.isnan(posteriors[iloc]):
                whisper_label.append(np.nan)
            elif not len(transcript):
                whisper_label.append(0)
            else:
                whisper_label.append(1)
        
        return whisper_label

    def stack_save_results(
            self,
            audio_paths_full:list[str],
            languages:list[str],
            segments_english:list[tuple],
            number_segments,
            speech_duration,
            segments_foreign:list[tuple],
            posteriors:list[float],
            labels:list[float],
            output_save_path:str
    ):
        """
        ARGUMENTS
        ------------------------------
        audio_paths_full:list[str]  = A list that contains all wav-files that should be processed.
        languages:list[str]         = A list that denotes the detected language for every wav-file.
        segments_english:list[tuple]= A list that contains the single segments for every wav-file in English.
        number_segments:list[int]   = A list that denotes the number of segments for every wav-file.
        speech_duration:list[float] = A list that denotes the length of every wav-file.
        segments_foreign:list[tuple]= If a wav-file's original language is not English, this list contains the
                                      segments in the wav-file's original language.
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
        audio_lengths = self.get_audio_length(audio_paths_full)
        speech_fraction = [np.nan if np.isnan(length) else (duration/length) 
                           for duration, length in zip(speech_duration, audio_lengths)]
        segments_english_timestamps = self.unstack_segments(segments_english, True)
        segments_foreign_timestamps = self.unstack_segments(segments_foreign, True)
        segments_english = self.unstack_segments(segments_english, False)
        segments_foreign = self.unstack_segments(segments_foreign, False)
        whisper_label = self.create_whisper_label(segments_english, posteriors)

        
        df = pd.DataFrame({
            "file_name": audio_file_name,
            "file_path": audio_paths_full,
            "fg_posterior": posteriors,
            "fg_label": labels,
            "whisper_label": whisper_label,
            "number_of_segments": number_segments,
            "spech_duration": speech_duration,
            "speech_fraction": speech_fraction,
            "detected_language": languages,
            "transcript_english": segments_english,
            "transcript_english_segments": segments_english_timestamps,
            "transcript_foreign": segments_foreign,
            "transcript_foreign_segments": segments_foreign_timestamps,
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
        log_msg = f"The Ear-Pipeline Inference started on {cur_time.date()} at {cur_time.time()}!"

        audio_paths_full = []
        for audio_dir in audio_dirs:
            audio_paths_full = [*audio_paths_full, *self.get_audio_paths(audio_dir)]
        languages, segments_english, number_segments, speech_duration, segments_foreign = self.run_whisper_inference(audio_paths_full)
        posteriors, labels = self.run_foreground_inference(audio_paths_full)
        self.stack_save_results(audio_paths_full, languages, segments_english, number_segments,speech_duration, 
                                segments_foreign, posteriors, labels, output_save_path)
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

        ],
        output_save_path="",
        log_save_path=""
    )

