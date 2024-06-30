"use client";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useState } from "react";
import { Paper, Textarea, Button } from "@mantine/core";
import { LoadingOverlay } from "@mantine/core";

export default function Home() {
    const router = useRouter()
    const [playlist_ID, setPlaylistID] = useState<string | null>(null);
    const [generatedSummary, setGenSummary] = useState<string | null>(null);
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [waiting, setWaiting] = useState(false);

    async function runQA() {
        setAnswer('')
        setWaiting(true);
        const response = await fetch("http://127.0.0.1:5000/qa", {
                method: "POST", 
                headers: {
                    "Content-Type": "application/json"
                }, 
                mode: "cors", 
                body: JSON.stringify({question: question, playlist_id: playlist_ID})
            }); 

        const data = await response.json(); 
        console.log(data);
        setWaiting(false);
        setAnswer(data.answer)
    }

    useEffect( () => {
        
        async function setupIDAndSummary() {
            const URL = window.location.href; 
            console.log(URL)
            const PLAYLIST_ID = URL.slice(URL.indexOf("chat/") + 5, URL.length); 
            if (PLAYLIST_ID != null) {
                setPlaylistID(PLAYLIST_ID);
            }

            /** setup the summary */

            const response = await fetch("http://127.0.0.1:5000/summary", {
                method: "POST", 
                headers: {
                    "Content-Type": "application/json"
                }, 
                mode: "cors", 
                body: JSON.stringify({playlist_id: PLAYLIST_ID})
            })

            const data = await response.json(); 
            console.log(data);
            setGenSummary(data.summary)
        }

        setupIDAndSummary();

    }, []); 
    


    return (
        <main className="flex min-h-screen flex-col items-center p-12 gap-4">
            <LoadingOverlay visible={generatedSummary == null || waiting} />
            {playlist_ID && 
                <main className="flex min-h-screen flex-col items-center gap-4">
                    <h1 className="text-4xl"> chat with our model </h1>
                    
                    <p className="text-md mt-8 w-[75%]"> here, you will receive a summary of the playlist video. you will also be able to ask questions to our model. </p> 
                    
                    {generatedSummary && 
                        <Paper className="w-[75%]" withBorder p="md">
                            <div className="w-full flex-col flex gap-4">
                                
                                <p className="text-xl w-[75%]"> <strong> summary </strong> </p> 
                                <p className="text-md"> {generatedSummary} </p> 

                                <p className="text-xl w-[75%]"> <strong> chat with jamba </strong> </p> 

                                <Textarea
                                    label="submit your questions"
                                    placeholder="what did your teacher not teach you?"
                                    autosize
                                    value={question}
                                    onChange={(event) => setQuestion(event.currentTarget.value)}
                                />

                                {answer && 
                                    <p className="text-md"> {answer} </p> 
                                }

                                <Button disabled={question.length == 0} onClick={runQA}> get some help </Button> 

                            </div>
                            
                        </Paper>
                    }

                </main> 
            }
        </main>
    );
}
