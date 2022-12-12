# POC - AIVideoEditor with features -  removing filler words based on transcript, generating SRT, Redact PII, Content Analyzer - Summary, Topic Modelling, Sensitivity.

## Abstract - POC for creating a AI Video Editor for Content Creators with Various Capabilities.

The video production industry is changing as a result of AI, which makes it quicker and simpler to organize clips and do flawless edits. According to a survey conducted by Business Insider, Companies that produce videos are evolving: by 2018, 78% of marketers either utilized or intended to use AI. AI functionality will shorten the editing process and expand your creative options, whether you're working on small social videos or long-form movies. Spend more time on the substance and less on editing. Fortunately, artificial intelligence (AI) has found ways to speed up and reduce the cost of production, including automatic video editing, 3D animation, and realistic-looking visuals. The video production industry is changing as a result of AI, which makes it quicker and simpler to organize clips and do flawless edits. According to a survey conducted by Business Insider, Companies that produce videos are evolving: by 2018, 78% of marketers either utilized or intended to use AI. AI functionality will shorten the editing process and expand your creative options, whether you're working on small social videos or long-form movies. Spend more time on the substance and less on editing. Fortunately, artificial intelligence (AI) has found ways to speed up and reduce the cost of production, including automatic video editing, 3D animation, and realistic-looking visuals. 

Additionally, it may enable transcription to be done more quickly and easily than by humans, reducing labor expenses.

 Imagine you have just recorded a video and you want to edit it before uploading it. There are a few ways to fix the problem. To name a few you can record an entire new video or you can record only the part you want to change and then combine it with the original video. A lot of time while recording an interview or video about yourself we tend to use filler words like “aa”, “..uhmm”, etc and while uploading the video or before sharing it with anyone we want to get rid of these words. There are times where we want to redact PII, identify topics, and remove sensitive content.

As a solution to the aforementioned problem statement, we propose in this project an all-in-one collaborative AI audio and video editing application that is as simple to use as editing a Google Doc based on text extracted from transcription. Even inexperienced users with no editing skills can edit with AI like a pro with this approach. Our proposed AI tool would offer fantastic features such as automatic transcription from audio, the ability to remove filler words with the click of a button,which makes editing your recorded audio as simple as typing, and the addition of speaker labels. Our solution entails identifying and comprehending all available cutting-edge state-of-the-art(SOTA) models like OPenAI whisper and attempting transfer learning for real-time voice cloning from existing text-speech models in research, and developing an end-to-end ML product for our solution domain.


**MODULES**

- SPEECH TO TEXT ( TRANSCRIBE )  - OPENAI WHISPER MODEL : https://github.com/openai/whisper

- STABLIZE TIMESTAMP AND GENERATE WORD TIMESTAMPS  

COLAB: https://github.com/rameshavinash94/AIVideoEditor/blob/main/colabs/Transcription_%26_stabilizing_word_timestamps.ipynb

- FILLER WORD REMOVAL - Python Moviepy and Word TImestamps
COLAB: https://github.com/rameshavinash94/AIVideoEditor/blob/main/colabs/Filler_words_removal.ipynb

- SPEAKER DIARISATION - OPENAI WHISPER WITH PYANNOTATE
COLAB: https://github.com/rameshavinash94/AIVideoEditor/blob/main/colabs/Speaker_Diarization.ipynb

-VIDEO EDITOR CAPABILITIES. 

COLAB:https://github.com/rameshavinash94/AIVideoEditor/blob/main/colabs/Video_editing_functionalities(Remove_Silent_Parts_Video%2Ctrimming%2Ccutting_videos).ipynb

- PII REDACTION , CONTENT ANALYSIS AND OTHER MODUES ALL ARE INTEGRATING WITH STREAMLTI APPLICATION

COLAB: https://github.com/rameshavinash94/AIVideoEditor/blob/main/colabs/Streamlit_Application.ipynb

Main App File : https://github.com/rameshavinash94/AIVideoEditor/blob/main/deployment_files/app.py

Requirements : https://github.com/rameshavinash94/AIVideoEditor/blob/main/deployment_files/requirements.txt


## DEPLOYMENT LINK:
Deployment url: https://huggingface.co/spaces/AvinashRamesh23/AIEditor

Hugging Face Spaces make it easy for you to create and deploy ML-powered demos. 

Deployment Repo: https://huggingface.co/spaces/AvinashRamesh23/AIEditor/tree/main



## TO RUN THE APPLICATION IN LOCALHOST RUN THE STREAMLIT COLAB UNDER COLAB FOLDER. It has all requirements to install at the top.
COLAB: https://github.com/rameshavinash94/AIVideoEditor/blob/main/colabs/Streamlit_Application.ipynb
