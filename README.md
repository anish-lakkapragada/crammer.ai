Basically it operates like this: 

(1) takes in a playlist yt link 
--> extract all the transcripts with YT transcript API 
--> use the transcript API as the external DB Store for RAG [this is kind slow]

(2) take in the prompts and use the context and feed into LLM [this requires LLM tokens.]


constraints: 
> requirement of the LLM tokens from another API 
> slowness of using YT transcription api. 

