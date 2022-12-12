
import streamlit as st
import whisper
import re
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
import math
from stable_whisper import modify_model,results_to_word_srt
import asyncio
from deepgram import Deepgram
from typing import Dict
import os
import moviepy.editor as mp
from pytube import YouTube
from time import sleep
import pandas as pd

st.title('AI Editor for Content Creators!')

@st.cache(suppress_st_warning=True) 
#load whisper model
def load_model(model_selected):
  #load medium model
  model = whisper.load_model(model_selected)
  # modify model to get word timestamp
  modify_model(model)
  return model

#transcribe
@st.cache(suppress_st_warning=True) 
def transcribe_video(vid,model_selected):
    model = load_model(model_selected)
    options = whisper.DecodingOptions(fp16=False,language="English")
    result = model.transcribe(vid, **options.__dict__)
    result['srt'] = whisper_result_to_srt(result)
    return result

#srt generation
def whisper_result_to_srt(result):
    text = []
    for i,s in enumerate(result['segments']):
        text.append(str(i+1))
        time_start = s['start']
        hours, minutes, seconds = int(time_start/3600), (time_start/60) % 60, (time_start) % 60
        timestamp_start = "%02d:%02d:%06.3f" % (hours, minutes, seconds)
        timestamp_start = timestamp_start.replace('.',',')     
        time_end = s['end']
        hours, minutes, seconds = int(time_end/3600), (time_end/60) % 60, (time_end) % 60
        timestamp_end = "%02d:%02d:%06.3f" % (hours, minutes, seconds)
        timestamp_end = timestamp_end.replace('.',',')        
        text.append(timestamp_start + " --> " + timestamp_end)
        text.append(s['text'].strip() + "\n")
    return "\n".join(text)

#compute speaking_time
async def compute_speaking_time(transcript_data: Dict,data:str) -> None:
   if 'results' in transcript_data:
       transcript = transcript_data['results']['channels'][0]['alternatives'][0]['words']
       total_speaker_time = {}
       speaker_words = []
       current_speaker = -1

       for speaker in transcript:
           speaker_number = speaker["speaker"]

           if speaker_number is not current_speaker:
               current_speaker = speaker_number
               speaker_words.append([speaker_number, [], 0])

               try:
                   total_speaker_time[speaker_number][1] += 1
               except KeyError:
                   total_speaker_time[speaker_number] = [0,1]

           get_word = speaker["word"]
           speaker_words[-1][1].append(get_word)

           total_speaker_time[speaker_number][0] += speaker["end"] - speaker["start"]
           speaker_words[-1][2] += speaker["end"] - speaker["start"]

       for speaker, words, time_amount in speaker_words:
           print(f"Speaker {speaker}: {' '.join(words)}")
           data+=f"\nSpeaker {speaker}: {' '.join(words)}"
           print(f"Speaker {speaker}: {time_amount}")
           data+=f"\nSpeaker {speaker}: {time_amount}"


       for speaker, (total_time, amount) in total_speaker_time.items():
           print(f"Speaker {speaker} avg time per phrase: {total_time/amount} ")
           data+=f"\nSpeaker {speaker} avg time per phrase: {total_time/amount} "
           print(f"Total time of conversation: {total_time}")
           data+=f"\nTotal time of conversation: {total_time}"
   return transcript,data

#extract audio from video
def extract_write_audio(vd):
  my_clip = mp.VideoFileClip(f'{vd}')
  my_clip.audio.write_audiofile(f"audio.wav")

#speaker diarization workflow
async def speaker_diarization_flow(PATH_TO_FILE):
  audio = extract_write_audio(PATH_TO_FILE)
  data = ''
  DEEPGRAM_API_KEY = "3dc39bf904babb858390455b1a1399e221bf87f8"
  deepgram = Deepgram(DEEPGRAM_API_KEY)
  with open(PATH_TO_FILE, 'rb') as audio:
       source = {'buffer': audio, 'mimetype': 'audio/wav'}
       transcription =  await deepgram.transcription.prerecorded(source, {'punctuate': True, 'diarize': True})
       transcript,final_data =  await compute_speaking_time(transcription,data)
  return final_data

# speaker diarization main funciton
async def speaker_diarization(PATH_TO_FILE):
  data = await speaker_diarization_flow(PATH_TO_FILE)
  print("data is", data)
  return data

#find filler words
def filler_words_finder(result_data):
  word_map_prior_edit=set()
  word_map_after_edit=set()
  #my filler words sample
  filler_words={'um','ah','you know','mmm','mmm','er','uh','Hmm','actually','basically','seriously','mhm','uh huh','uh','huh','ooh','aah','ooh'}
  filler_words_timestamp=set()
  for keys  in result_data:
    if keys == 'segments':
        prev=0
        for i in result_data[keys]:
            for word in i['whole_word_timestamps']:
                lower_case = re.sub(r'\W','',word['word'].lower())
                word_map_prior_edit.add(word['timestamp'])
                if lower_case in filler_words or lower_case.startswith(('hm','aa','mm','oo')):
                    print(word['word'].lower(),word['timestamp'])
                    filler_words_timestamp.add(word['timestamp'])
                    prev=word['timestamp']
                    continue
                word_map_after_edit.add((prev,word['timestamp']))
                prev=word['timestamp']
  return word_map_after_edit, filler_words_timestamp

def merge_overlapping_time_intervals(intervals):
    stack = []
    result=[intervals[0]]

    for interval in intervals:
            interval2=result[-1]

            if overlap(interval,interval2):
                result[-1] = [min(interval[0],interval2[0]),max(interval[1],interval2[1])]
            else:
                result.append(interval)
      
    return result

def overlap(interval1,interval2):
            return min(interval1[1],interval2[1])-max(interval1[0],interval2[0]) >= 0

#assembly ai endpoints
import requests
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
upload_endpoint = "https://api.assemblyai.com/v2/upload"

headers = {
	"authorization": "05e515bf6b474966bc48bbdd1448b3cf",
	"content-type": "application/json"
}

def upload_to_AssemblyAI(save_location):
  CHUNK_SIZE = 5242880
  def read_file(filename):
    with open(filename, 'rb') as _file:
      while True:
        print("chunk uploaded")
        data = _file.read(CHUNK_SIZE)
        if not data:
          break
        yield data
  
  upload_response = requests.post(
	  upload_endpoint,
	  headers=headers, data=read_file(save_location)
	)
  print(upload_response.json())	
  audio_url = upload_response.json()['upload_url']
  print('Uploaded to', audio_url)
  return audio_url


def start_analysis(audio_url,type):
	## Start transcription job of audio file
  data = {
	    'audio_url': audio_url,
	    'iab_categories': True,
	    'content_safety': True,
	    "summarization": True,
	    "summary_type": "bullets",
      "summary_model":type
	}
  if type=='conversational':
    data["speaker_labels"]= True

  transcript_response = requests.post(transcript_endpoint, json=data, headers=headers)
  print(transcript_response.json())
  transcript_id = transcript_response.json()['id']
  polling_endpoint = transcript_endpoint + "/" + transcript_id
  print("Transcribing at", polling_endpoint)
  return polling_endpoint

def get_analysis_results(polling_endpoint):	
  status = 'submitted'

  while True:
    print(status)
    polling_response = requests.get(polling_endpoint, headers=headers)
    status = polling_response.json()['status']
	  # st.write(polling_response.json())
	  # st.write(status)
    if status == 'submitted' or status == 'processing' or status == 'queued':
      print('not ready yet')
      sleep(10)
    
    elif status == 'completed':
      print('creating transcript')
      return polling_response
      break
    
    else:
      print('error')
      return False
      break

def pii_redact(audiourl,options):
  print(options,audiourl)
  endpoint = "https://api.assemblyai.com/v2/transcript"
  json = {
    "audio_url": audiourl,
    "redact_pii": True,
    "redact_pii_audio": True,
    "redact_pii_policies": options
  }

  headers = {
      "authorization": "05e515bf6b474966bc48bbdd1448b3cf",
      "content-type": "application/json",
  }

  response = requests.post(endpoint, json=json, headers=headers)
  print(response.json())
  transcript_id = response.json()['id']
  polling_endpoint = endpoint + "/" + transcript_id
  return polling_endpoint

def pii_redact_audio(polling_endpoint):
  status = 'submitted'
  headers = {
      "authorization": "05e515bf6b474966bc48bbdd1448b3cf",
      "content-type": "application/json",
  }
  while True:
    print(status)
    polling_response = requests.get(polling_endpoint, headers=headers)
    status = polling_response.json()['status']
    if status == 'submitted' or status == 'processing' or status == 'queued':
      print('not ready yet')
      sleep(10)
    
    elif status == 'completed':
      print('creating transcript')
      return polling_response
      break
    
    else:
      print('error')
      return False
      break

def download_redact_audio(pooling_enpoint):
  headers = {
      "authorization": "05e515bf6b474966bc48bbdd1448b3cf",
      "content-type": "application/json",
  }

  redacted_audio_response = requests.get(pooling_enpoint + "/redacted-audio",headers=headers)
  print(redacted_audio_response.json())
  redacted_audio = requests.get(redacted_audio_response.json()['redacted_audio_url'])
  with open('redacted_audio.mp3', 'wb') as f:
    f.write(redacted_audio.content)

def redact_audio_video_display(vd,audio):
  audioclip = AudioFileClip(audio)
  clip = VideoFileClip(vd)
  videoclip = clip.set_audio(audioclip)
  videoclip.write_videofile("Redacted_video.mp4")
  st.video("Redacted_video.mp4")

async def main(uploaded_video,model_selected):
    preview = st.video(uploaded_video)
    try:
      vid = uploaded_video.name
      with open(vid, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk
    except:
      yt = YouTube(uploaded_video)
      yt.streams.filter(file_extension="mp4").get_by_resolution("360p").download(filename="youtube.mp4")
      vid = "youtube.mp4"
    finally:
      name = vid.split('.')[0]
      #extracting the transcription result
      with st.spinner('Transcribing Video, Wait for it...'):
        result = transcribe_video(vid,model_selected)
        st.text_area("Edit Transcript",result["text"])
        col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1])
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Remove Filler Words","Edit Video" ,"Download SRT", "Perform Speaker Diarization","Content Analyzer","PII redactation"])
      
      with tab1:
        filler_word = st.button('Edit/Remove Filler Words with a click of a button')
        if filler_word:
          with st.spinner(text="In progress..."):
              word_map_after_edit, filler_words_timestamp = filler_words_finder(result)
              final_intervals = merge_overlapping_time_intervals(sorted(list(word_map_after_edit)))
              subclips=[]
              for start,end in final_intervals:
                  clip = VideoFileClip(vid)
                  tmp = clip.subclip(start,(end - end*0.1))
                  print(start,end,tmp.duration)
                  subclips.append(tmp)
              #concatenate subclips without filler words
              final_clip = concatenate_videoclips(subclips)
              final_clip.write_videofile(f"remove_{vid}")
              preview = st.video(f"remove_{vid}")

      with tab2:
        save = st.button('Edit')

      with tab3:
        download = st.download_button('Download SRT', result['srt'],f'{name}.srt')
        if download:
          st.write('Thanks for downloading!')
      
      with tab4:
        identify_download_speaker = st.button('Perform Speaker Diarization')
        if identify_download_speaker:
          with st.spinner(text="In progress..."):
              results  = await speaker_diarization(vid)
              download_speaker = st.download_button("download speaker_diarization",results,'diarization_stats.txt')
          if download_speaker:
            st.write('Thanks for downloading!')

      with tab5:
        type = st.selectbox('Summary Type?',('informative', 'conversational', 'catchy'))
        Analyze_content = st.button("Start Content Analysis")
        if Analyze_content:
          with st.spinner(text="In progress..."):
              audio = extract_write_audio(vid)
              audio_url = upload_to_AssemblyAI("audio.wav")
              # start analysis of the file
              polling_endpoint = start_analysis(audio_url,type)
              # receive the results
              results = get_analysis_results(polling_endpoint)
    
              # separate analysis results
              summary = results.json()['summary']
              content_moderation = results.json()["content_safety_labels"]
              topic_labels = results.json()["iab_categories_result"]
    
              my_expander1 = st.expander(label='Summary')
              my_expander2 = st.expander(label='Content Moderation')
              my_expander3 = st.expander(label='Topic Discussed')

          with my_expander1:
            st.header("Video summary")
            st.write(summary)

          with my_expander2:
              st.header("Sensitive content")
              if content_moderation['summary'] != {}:
                st.subheader('🚨 Mention of the following sensitive topics detected.')
                moderation_df = pd.DataFrame(content_moderation['summary'].items())
                moderation_df.columns = ['topic','confidence']
                st.dataframe(moderation_df, use_container_width=True)
              else:
                st.subheader('✅ All clear! No sensitive content detected.')

          with my_expander3:
            st.header("Topics discussed")
            topics_df = pd.DataFrame(topic_labels['summary'].items())
            topics_df.columns = ['topic','confidence']
            topics_df["topic"] = topics_df["topic"].str.split(">")
            expanded_topics = topics_df.topic.apply(pd.Series).add_prefix('topic_level_')
            topics_df = topics_df.join(expanded_topics).drop('topic', axis=1).sort_values(['confidence'], ascending=False).fillna('')
            st.dataframe(topics_df, use_container_width=True)

      with tab6:
        options = st.multiselect('Select Policies to redact from video',["medical_process","medical_condition","blood_type","drug","injury","number_sequence","email_address","date_of_birth","phone_number","us_social_security_number","credit_card_number","credit_card_expiration","credit_card_cvv","date","nationality","event","language","location","money_amount","person_name","person_age","organization","political_affiliation","occupation","religion","drivers_license","banking_information"],["person_name", 'credit_card_number'])
        Perform_redact = st.button("Start PII Redaction")
        if Perform_redact:
            with st.spinner(text="In progress..."):
              audio = extract_write_audio(vid)
              audio_url = upload_to_AssemblyAI("audio.wav")
              print(audio_url)
              print([ x for x in options ])
              polling_endpoint = pii_redact(audio_url,options)
              results  = pii_redact_audio(polling_endpoint)
              download_redact_audio(polling_endpoint)
              redact_audio_video_display(vid,"redacted_audio.mp3")

Model_type = st.sidebar.selectbox("Choose Model",('Tiny - Best for Srt generation', 'Base - Best suited for various AI services', 'Medium - Use this model for filler word removal'),0)
upload_video = st.sidebar.file_uploader("Upload mp4 file",type=["mp4","mpeg"])
youtube_url  = st.sidebar.text_input("Enter a youtube video url")
submit_button = st.sidebar.button("Extract Youtube Video")

if Model_type.startswith("Tiny"):
    model_selected = 'tiny.en'
if Model_type.startswith("Base"):
    model_selected = 'base.en'
if Model_type.startswith("Small"):
    model_selected = 'small.en'
if Model_type.startswith("Medium"):
    model_selected = 'medium.en'

if submit_button and youtube_url!=None:
  asyncio.run(main(youtube_url,model_selected))

if upload_video is not None:
  asyncio.run(main(upload_video,model_selected))
