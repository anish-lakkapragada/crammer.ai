# crammer.ai 

a simple ai-powered app for helping you study more efficiently from youtube playlists.

## directory structure

`/frontend` contains the [next.js application](https://project-five-orpin.vercel.app). 

`/backend` contains the flask api that uses a121's jamba llm for summarization and qa w/rag. our api runs locally on port `5000`, and our frontend assumes the api is running at `http://127.0.0.1:5000`. 

to run the flask app, you can run `sh run.sh` after installing the dependencies in `requirements.txt` or just build and run the docker container with `docker build -t api . ` and `docker run -p 5000:5000 api` respectively.

## what we could use some help on 

how do we deploy our containerized api for free? not speaking about the ml portion, just hosting. we currently think converting our api to a lambda function is the best way. 