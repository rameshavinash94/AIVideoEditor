# POC - AIVideoEditor with features -  removing filler words based on transcript, generating SRT, Redact PII, Content Analyzer - Summary, Topic Modelling, Sensitivity.

## Abstract - POC for creating a AI Video Editor for Content Creators with Various Capabilities.

The video production industry is changing as a result of AI, which makes it quicker and simpler to organize clips and do flawless edits. According to a survey conducted by Business Insider, Companies that produce videos are evolving: by 2018, 78% of marketers either utilized or intended to use AI. AI functionality will shorten the editing process and expand your creative options, whether you're working on small social videos or long-form movies. Spend more time on the substance and less on editing. Fortunately, artificial intelligence (AI) has found ways to speed up and reduce the cost of production, including automatic video editing, 3D animation, and realistic-looking visuals. The video production industry is changing as a result of AI, which makes it quicker and simpler to organize clips and do flawless edits. According to a survey conducted by Business Insider, Companies that produce videos are evolving: by 2018, 78% of marketers either utilized or intended to use AI. AI functionality will shorten the editing process and expand your creative options, whether you're working on small social videos or long-form movies. Spend more time on the substance and less on editing. Fortunately, artificial intelligence (AI) has found ways to speed up and reduce the cost of production, including automatic video editing, 3D animation, and realistic-looking visuals. 

Additionally, it may enable transcription to be done more quickly and easily than by humans, reducing labor expenses.

 Imagine you have just recorded a video and you want to edit it before uploading it. There are a few ways to fix the problem. To name a few you can record an entire new video or you can record only the part you want to change and then combine it with the original video. A lot of time while recording an interview or video about yourself we tend to use filler words like “aa”, “..uhmm”, etc and while uploading the video or before sharing it with anyone we want to get rid of these words. There are times where we want to redact PII, identify topics, and remove sensitive content.

As a solution to the aforementioned problem statement, we propose in this project an all-in-one collaborative AI audio and video editing application that is as simple to use as editing a Google Doc based on text extracted from transcription. Even inexperienced users with no editing skills can edit with AI like a pro with this approach. Our proposed AI tool would offer fantastic features such as automatic transcription from audio, the ability to remove filler words with the click of a button,which makes editing your recorded audio as simple as typing, and the addition of speaker labels. Our solution entails identifying and comprehending all available cutting-edge state-of-the-art(SOTA) models like OPenAI whisper and attempting transfer learning for real-time voice cloning from existing text-speech models in research, and developing an end-to-end ML product for our solution domain.


## **MODULES**

- **SPEECH TO TEXT ( TRANSCRIBE )**  - OPENAI WHISPER MODEL : https://github.com/openai/whisper
Speech recognition is an interdisciplinary subfield of computer science and computational linguistics that develops methodologies and technologies that enable the recognition and translation of spoken language into text by computers with the main benefit of searchability.

Transcription helps you convert recorded speech to text.Transcription, or transcribing as it is often referred to, is the process of converting speech from an audio or video recording into text.Transcription entails more than just listening to recordings.After the transcript is generated, we are also stabilizing the word timestamps  for various other features.

**OpenAI whisper:**
Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web

**SNIPPET**

<img width="834" alt="Screen Shot 2022-12-12 at 1 20 54 AM" src="https://user-images.githubusercontent.com/87649563/207008408-fca8fe5c-9309-45fe-b19e-5370da7b71ad.png">

- **STABLIZE TIMESTAMP AND GENERATE WORD TIMESTAMPS**:
https://openai.com/blog/whisper/ only mentions "phrase-level timestamps",we infer from it that word-level timestamps are not available.
Getting word-level timestamps are not directly supported, but it could be possible using the predicted distribution over the timestamp tokens or the cross-attention weights.
Reference: https://github.com/jianfch/stable-ts

COLAB: https://github.com/rameshavinash94/AIVideoEditor/blob/main/colabs/Transcription_%26_stabilizing_word_timestamps.ipynb

- **FILLER WORD REMOVAL** - Python Moviepy and Word TImestamps
COLAB: https://github.com/rameshavinash94/AIVideoEditor/blob/main/colabs/Filler_words_removal.ipynb
The some list of Filler Words that our application will be able to clip off from the video are:
"um"
"uh"
"hmm"
"mhm"
"uh huh"


- **SPEAKER DIARISATION** - OPENAI WHISPER WITH PYANNOTATE
Speaker diarization is a combination of speaker segmentation and speaker clustering.The first aims at finding speaker change points in an audio stream.
The second aims at grouping together speech segments on the basis of speaker characteristics.

COLAB: https://github.com/rameshavinash94/AIVideoEditor/blob/main/colabs/Speaker_Diarization.ipynb
We can automatically detect the number of speakers in your audio file, and each word in the transcription text can be associated with its speaker

**SNIPPET**
<img width="764" alt="Screen Shot 2022-12-12 at 1 25 44 AM" src="https://user-images.githubusercontent.com/87649563/207009561-709f6724-2372-4d16-a1fc-813855e99c7e.png">

**FILE:** https://github.com/rameshavinash94/AIVideoEditor/blob/main/Artifacts/diarization_stats.txt

Speakers will be labeled as Speaker 0, Speaker 1, etc.

- **SRT GENERATION** 
<img width="786" alt="Screen Shot 2022-12-12 at 1 25 10 AM" src="https://user-images.githubusercontent.com/87649563/207009317-914d870f-0b87-4cad-ae0f-ea9d33231b37.png">

**GENERATED SRT FILE:**

https://github.com/rameshavinash94/AIVideoEditor/blob/main/Artifacts/30%20Second%20Elevator%20Pitch.srt

-**VIDEO EDITOR CAPABILITIES**. 
Try out various video editing capabilities like trimming, watermarking, sub clipping, silent parts removal etc..

COLAB:https://github.com/rameshavinash94/AIVideoEditor/blob/main/colabs/Video_editing_functionalities(Remove_Silent_Parts_Video%2Ctrimming%2Ccutting_videos).ipynb

- **PII REDACTION , CONTENT ANALYSIS AND OTHER MODUES ALL ARE INTEGRATING WITH STREAMLTI APPLICATION**

**PII**: Personally Identifiable Information
In this feature, the portion of the video which contains sensitive personal information is removed as per the user’s request of type of information, which can be name, occupation, email address or phone number.
Here, if the video contains any sensitive perforation information, which user doesn’t want , then, he can select the input information type and the information will be redacted in the new generated video.

**SNIPPET**

<img width="809" alt="Screen Shot 2022-12-12 at 1 27 30 AM" src="https://user-images.githubusercontent.com/87649563/207009912-7eeb95ae-42a4-497f-abdc-bba43e6edd8a.png">

**UPLOADED VIDEO**

https://user-images.githubusercontent.com/87649563/207010501-b6f38e15-1ef4-4b2a-be3b-fcf986efe960.mp4


**REDACTED VIDEO**

https://user-images.githubusercontent.com/87649563/207010370-d08be431-ab1d-4b95-89f1-74dad2b327b8.mp4

**Content Analyzer**:
With Summarization, we can generate a single abstractive summary of entire audio files submitted for transcription.
With Topic Detection,  we can label the topics that are spoken in your audio/video files. The predicted topic labels follow the standardized IAB Taxonomy, which makes them suitable for Contextual Targeting use cases. This API can predict the topic names among 698 different topics.
With Content Safety Detection, we can detect if any of the following sensitive content is spoken in your audio/video files, and pinpoint exactly when and what was spoken:

**SNIPPET**

<img width="521" alt="Screen Shot 2022-12-12 at 1 32 36 AM" src="https://user-images.githubusercontent.com/87649563/207011015-93490c0e-53d5-4db2-955c-7db518228ca9.png">


**STREAMLIT**
Streamlit is an open source app framework in Python language. It helps us create web apps for data science and machine learning in a short time.

**SNIPPET**

<img width="1326" alt="Screen Shot 2022-12-12 at 1 33 41 AM" src="https://user-images.githubusercontent.com/87649563/207011165-c2013c35-ea0b-4b4f-a6a5-eef3b7710ec3.png">

## **MLOps Architecture** 

Here, we studied several MLOps architecture for various different projects and performed literature survey for the same. Thereafter, we came across the MLOps architecture given by Vertex.AI ,which we have tried to follow during the entire duration of the project.

For the proof of concepts, we used many datasets like LibreSpeech and MoviePy dataset to perform Proof of Concepts (POCs) for various different features like speech-to-text transcription, anad attaching word timestamps. We have tested several pre-trained models like OpenAI whisper and Descrypt APIs for various services and found that OpenAI whisper gives the best accuracy. As we are using pretrained models, we do need to retrain the model  continuously. Thereafter, we have used streamlit application for front-end development. Lastly, we created a Docker Image for the source code and thereafter, deployed the entire application on HuggingFace, so that any user  can access the project and perform various operations in their audio/video input.


## INDIVIDUAL PROJECT CONTRIBUTION:

Avinash Ramesh: I started working on the different AI services like content Analysis and Personally Identifiable Information (PII) redactation.For all those features, I tested different state of the art models for these features and finalized the pretrained model as per its performance and accuracy. I also performed the  worked on the Proof of Concepts (POCs) of integrating all AI services and testing out all the functionality together and see how it works using a simple streamline application.


Nevil Shah:I was entrusted with the responsibility to generate audio/video transcripts from an audio/video and try various state of the art models for audio/video transcription and determine its accuracy to the original content. First and Foremost,I performed the proof of concept for various models like openAI whisper and descrypt APIs and found that openAI whisper model worked the best. Firstly,after saving the live recorded input file, the transcript for that input is generated.I also tried the same process for different types of audio with regards to speed of speech, noise variation and the tones. Thereafter,I attached timestamps to every word in the generated transcript.Lastly,  I added speaker labels to each of the different voices in the audio file. 


Yash Kamtekar: I developed the PoC of filler word removal service on click of a single button. This feature works audio/video file that is either recorded or a YouTube video. It extracts the audio from the file and then generates the transcripts after which it attaches timestamp to these transcripts for further processing. The generated output from the transcription service is given to the Open Whisper model which  has been given a small set of filler words. The model prints the filler word and it’s duration as per the audio generated from step 2. Using moviepy library of python and the output of the model the audio files are merged to generate the output without the filler words.



COLAB: https://github.com/rameshavinash94/AIVideoEditor/blob/main/colabs/Streamlit_Application.ipynb

Main App File : https://github.com/rameshavinash94/AIVideoEditor/blob/main/deployment_files/app.py

Requirements : https://github.com/rameshavinash94/AIVideoEditor/blob/main/deployment_files/requirements.txt

Team Project report Link(Google Doc): https://docs.google.com/document/d/1GxQFaz1FvPdZohkQL56vVRqTUAU-Sk7ix_6CRbf05eo/edit#

Team Project report (Github Link): https://github.com/rameshavinash94/AIVideoEditor/blob/main/Project%20Report.docx

## DEPLOYMENT LINK:
Deployment url: https://huggingface.co/spaces/AvinashRamesh23/AIEditor

Hugging Face Spaces make it easy for you to create and deploy ML-powered demos. 

Deployment Repo: https://huggingface.co/spaces/AvinashRamesh23/AIEditor/tree/main


## TO RUN THE APPLICATION IN LOCALHOST RUN THE STREAMLIT COLAB UNDER COLAB FOLDER. It has all requirements to install at the top.
COLAB: https://github.com/rameshavinash94/AIVideoEditor/blob/main/colabs/Streamlit_Application.ipynb



## TEAM PRESENTATION VIDEO:

https://user-images.githubusercontent.com/87649563/207011452-7fd9e377-e9a8-4c95-8c44-df8a2fae7e77.mp4



