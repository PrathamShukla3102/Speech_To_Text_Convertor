# Speech_To_Text_Convertor

This repository presents a script to evaluate the performance of Automatic Speech Recognition (ASR) using the Wav2Vec2 model from Hugging Face's Transformers library. The script calculates the Character Error Rate (CER) and Word Error Rate (WER) by comparing the generated transcription with the provided reference transcription. The dataset includes audio files in .wav format paired with text files containing reference transcriptions, facilitating a straightforward assessment of transcription accuracy.

**Installations**<br>
Create and activate a virtual environment (optional but recommended):<br>
python3 -m venv venv<br>
cd venv\Scripts\activate

Install the required dependencies:<br>
pip install -r requirements.txt


**Dataset**
Nptel pure data of used in this model.<br>
https://github.com/AI4Bharat/NPTEL2020-Indian-English-Speech-Dataset<br>

**Examples**
1)File: 00082854c239062d386c13bf5bca0bf9f8322381080e0c3ef99f460c.wav<br>
Reference: EVENT HAS NOT CHANGED BECAUSE BOTH THESE EVENTS OCCURRED EXACTLY AT THE SAME POINT THE LIGHT IS EMITTED FROM THIS POINT GOES UPS GOES UP AND COMES<br>
Transcriptions: EVENT HAS NOT CHANGED BECAUSE BOTH THESE EVENTS OCCUR EXACTLY AT THE SAME POINT THE LIGHT IS EMITTED FROM THIS POINT GOES UP GOES UP AND COMES<br>
CER: 0.0274<br>
WER: 0.0741<br>

2)File: 0008a06a2ffb049fe3a1f0561c3017efc5fcca682671cd3530b4a815.wav<br>
Reference: OR A COLLECTION OF MULTIPLE ELEMENTS WHICH ARE KNOWN AS BLOBS SO ONE IMPORTANT TERM<br>
Transcriptions: OR A COLLECTION OF A MULTIPLE ELEMENTS WHICH ARE KNOWN AS BLOBS SO ONE WOUDEN TERM<br>
CER: 0.1084<br>
WER: 0.1333<br>

**Results**

**Average Character Error Rate:-0.1432**<br>
**Average Word Error Rate  :- 0.3271**




