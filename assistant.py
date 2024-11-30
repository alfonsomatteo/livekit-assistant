import asyncio
from typing import Annotated
import os

from livekit import agents, rtc, api
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

class AssistantFunction(agents.llm.FunctionContext):
    """Questa classe definisce le funzioni che verranno chiamate dall'assistente."""

    @agents.llm.ai_callable(
        description=(
            "Chiamato quando viene richiesto di valutare qualcosa che richiede capacità visive,"
            "ad esempio un'immagine, un video o il feed della webcam."
        )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="Il messaggio dell'utente che ha attivato questa funzione"
            ),
        ],
    ):
        print(f"Messaggio che ha attivato le capacità visive: {user_msg}")
        return None

async def get_video_track(room: rtc.Room):
    """Ottiene la prima traccia video dalla stanza. Useremo questa traccia per processare le immagini."""

    video_track = asyncio.Future()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Utilizzando la traccia video {track_publication.track.sid}")
                break

    return await video_track

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Nome della stanza: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Il tuo nome è Alloy. Sei un bot divertente e spiritoso. La tua interfaccia con gli utenti sarà vocale e visiva."
                    "Rispondi con risposte brevi e concise. Evita di usare punteggiatura impronunciabile o emoji."
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4")

    # Poiché OpenAI non supporta lo streaming TTS, lo useremo con un StreamAdapter
    # per renderlo compatibile con il VoiceAssistant
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    latest_image: rtc.VideoFrame | None = None

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),  # Useremo il Voice Activity Detector (VAD) di Silero
        stt=deepgram.STT(),  # Useremo il Speech To Text (STT) di Deepgram
        llm=gpt,
        tts=openai_tts,  # Useremo il Text To Speech (TTS) di OpenAI
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False):
        """
        Risponde al messaggio dell'utente con il testo fornito e, facoltativamente,
        con l'ultima immagine catturata dalla traccia video.
        """
        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            content.append(ChatImage(image=latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))

        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """Questo evento si attiva ogni volta che riceviamo un nuovo messaggio dall'utente."""

        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """Questo evento si attiva quando una chiamata di funzione dell'assistente è completata."""

        if len(called_functions) == 0:
            return

        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            asyncio.create_task(_answer(user_msg, use_image=True))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Ciao! Come posso aiutarti?", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)

        async for event in rtc.VideoStream(video_track):
            # Continueremo a prendere l'ultima immagine dalla traccia video
            # e la memorizzeremo in una variabile.
            latest_image = event.frame

if __name__ == "__main__":
    # Genera un token di accesso utilizzando le variabili d'ambiente per API Key e Secret
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("LIVEKIT_API_KEY e LIVEKIT_API_SECRET devono essere impostati nelle variabili d'ambiente.")

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("python-bot")
        .with_name("Python Bot")
        .with_grants(api.VideoGrants(room_join=True, room="my-room"))
        .to_jwt()
    )

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, token=token))

