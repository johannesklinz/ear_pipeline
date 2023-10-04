# EAR Pipeline
This pipeline combines the likes of OpenAI's [Whisper](https://github.com/openai/whisper) and UCS' [Foreground Speech Detection](https://github.com/usc-sail/egocentric-fg-speech-detection). In doing so, it supports 30s audio files - in theory, Whisper also supports longer files but FG is limited to this threshold as of now.
## Setup Instruction
1. Install [Python 3.9.9](https://www.python.org/downloads/release/python-399/).
2. Install [Git](https://git-scm.com/downloads).
3. Install [FFmpeg](https://ffmpeg.org/download.html) and add it to your PATH.
4. Clone this GitHub repository:<br>
`git clone https://github.com/johannesklinz/ear_pipeline`
5. Within the cloned repository, create a virtual environment:<br>
`python -m venv YOUR_ENV_NAME`
6. With your virtual environment activated, install the required packages:<br>
`pip install -r requirements.txt`
7. Download the models [here](https://drive.google.com/file/d/1dSKxlhW8ZEbewpyKkQz1LcTe_XpkQks9/view?usp=sharing) and extract them into the *models* folder.
## Run Inference
Within the `earpipeline.py`, at the very bottom, one can set three different variables:
- **audio_dirs**: This variable must be a list of strings. The strings should refer to all directories that contain wav-files and should be processed. Please make sure that you provide the different paths with forward slashes (i.e., /). You may provide as many paths as you like, the pipeline will filter out all wav-files in any given directory (not within its subdirectories!).
- **output_save_path**: Expects a path as a string, indicating under which location the Excel-output should be stored (i.e., the path must end with .xlsx).
- **log_save_path**: Expects a path as a string, indicating under which location the log file (i.e., meta information regarding the processing) should be stored (i.e., the path must end with .txt).
## Interpretation
The Excel-output contains thirteen columns, specifically:
- **file_name**: Denotes the name of the processed file.
- **file_path**: Denotes the specific path under which the given file is located.
- **fg_posterior**: Denotes the posterior of the FG algorithm (i.e., that there is foreground speech).
- **fg_label**: Denotes the FG algorithm's label (0 = there is no foreground speech, 1 = there is foreground speech).
- **whisper_label**: Denotes the Whisper algorithm's label (0 = Whisper did not detect speech, 1 = Whisper detected speech).
- **number_of_segments**: Denotes the number of segments that the Whisper algorithm identified within the given file (i.e., those do not necessarily align with speaker turns whatsoever).
- **speech_duration**: Denotes the sum of the length of all segments (i.e., sum of spoken time within a given audio file).
- **speech_fraction**: Denotes the fraction of the speech duration divided by the audio file length. Please bear in mind that Whisper always rounds to the closest full number which is why this value in some cases might exceed one by a very tiny margin.
- **detected_language**: Denotes the language that Whisper detected within the audio file. If there is no speech present in a given audio file, Whisper tends to detect norwegian (for whatever reason).
- **transcript_english**: If there is speech, this column contains the English transcript, either directely transcribed or translated from a foreign language, split up into the different segments.
- **transcript_english_segments**: This column additionally contains timestamps for every segment.
- **transcript_foreign**: If an audio file's original language is not English and there is speech, this column contains the transcript in the given foreign language, split up by its segments.
- **transcript_foreign_segments**: This column additionally contains timestamps for every segments in the given foreign language.
## Questions, Support etc.
If there are any questions regarding how to properly setup the pipeline, please reach out to [me](mailto:johannes.klinz@gmail.com).