"use client";
import Image from "next/image";
import { useState } from "react";
import { TextInput, Button } from '@mantine/core';
import { useRouter } from "next/navigation";

export default function Home() {
  const [playlistURL, setPlaylistURL] = useState(null);
  const router = useRouter();

  async function establish() {
    /** This function will contact the API. */
    
    const response = await fetch("http://127.0.0.1:5000/setup", {
      method: "POST", 
      headers: {
        "Content-Type": "application/json"
      }, 
      mode: "cors", 
      body: JSON.stringify({playlist_URL: playlistURL})
    })

    const data = await response.json(); 
    console.log(data);

    const playlist_ID = playlistURL.slice(playlistURL.indexOf("=") + 1, playlistURL.length)

    console.log(playlist_ID)

    if (response.ok) {
      router.push(`/chat/${playlist_ID}`)
    }
  }

  
  return (
    <main className="flex min-h-screen flex-col items-center p-12 gap-4">
      <h1 className="text-4xl"> crammer.ai </h1>

      <p className="text-md mt-8 w-[50%]"> simply enter the url of the youtube playlist you want to answer questions about below. </p> 

      <TextInput value={playlistURL} className="w-[50%]" label="enter youtube playlist link" placeholder="https://www.youtube.com/playlist?list=PLAYLIST_ID" onChange={(event) => setPlaylistURL(event.currentTarget.value)} /> 

      <div className="w-[50%]">
        <Button fullWidth onClick={establish} disabled={playlistURL == null}> get started </Button>
      </div> 

    </main>
  );
}
