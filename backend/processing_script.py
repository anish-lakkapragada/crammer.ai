from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi


URL = "https://www.youtube.com/playlist?list=PLZR1BGWBaZ1xNHAsECSZ5XWnHq-V4atUf"

playlist_urls = Playlist(URL)


for url in playlist_urls: 
   video_id = url[url.rfind("=") + 1 : ]
   transcript = YouTubeTranscriptApi.get_transcript(video_id)
   string = " ".join([transcript_time['text'] for transcript_time in transcript])
   
   f = open(f"videos/{video_id}.txt", "w")
   f.write(string.replace('\n', " "))
   f.close()

   