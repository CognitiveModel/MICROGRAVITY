import os
import asyncio
import base64
import json
from io import BytesIO
import mss # type: ignore
import mss.tools # type: ignore
from PIL import Image, ImageGrab # type: ignore
from google import genai # type: ignore
from google.genai import types # type: ignore
from typing import Callable, Optional, Dict, Any
import sounddevice as sd # type: ignore
import numpy as np # type: ignore
import queue
import threading
import concurrent.futures
from coding_agent.utils import config

class GeminiLiveStreamer:
    """
    Manages the continuous bidirectional WebSocket connection to the Gemini Multimodal Live API.
    Handles streaming screen frames and receiving real-time interaction predictions.
    """
    # Verified working model names for the Live API (bidiGenerateContent)
    # Must use v1alpha API version and AUDIO response modality
    LIVE_MODELS = [
        "gemini-2.0-flash-exp",                                  # Current standard for Realtime API
        "gemini-2.0-flash-001",                                  # Stable Release
        "gemini-2.5-flash-native-audio-latest",                  # Experimental Native Audio
    ]

    def __init__(self, api_key: str = None): # type: ignore
        self.api_key = config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in coding_agent/utils/config.py")
        
        # Use v1alpha API version which supports bidiGenerateContent
        self.client = genai.Client(
            api_key=self.api_key,
            http_options={'api_version': 'v1alpha'}
        )
        self.model = self.LIVE_MODELS[0]
        self.session = None
        self.is_streaming = False
        self.on_response_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.screen_observer: Optional[Any] = None # Injected by UIAgent
        
        # Audio configuration
        self.sample_rate = 16000
        self.audio_in_queue = queue.Queue()
        self.audio_out_queue = queue.Queue()
        self._audio_input_stream = None
        self._audio_output_stream = None
        
        self.current_roi = None # (x1, y1, x2, y2) in global pixels
        # Query actual screen size instead of hardcoding
        try:
            import ctypes
            _sw = ctypes.windll.user32.GetSystemMetrics(0)  # type: ignore
            _sh = ctypes.windll.user32.GetSystemMetrics(1)  # type: ignore
            self.screen_size = (_sw, _sh) if _sw > 0 and _sh > 0 else (1920, 1080)
        except Exception:
            self.screen_size = (1920, 1080)  # Final fallback
        self.debug_dir = None
        self._frame_count = 0

    async def start_session(self, system_instruction: str = None): # type: ignore
        """Establishes the WebSocket session and blocks until closed. Tries multiple model names."""
        print(f"[LiveStreamer] Attempting connection with model fallback list: {self.LIVE_MODELS}")
        
        config = {
            "response_modalities": ["AUDIO"],
        }
        if system_instruction:
            config["system_instruction"] = system_instruction # type: ignore

        print("[LiveStreamer] Initializing audio streams...")
        try:
            self._setup_audio_streams()
        except Exception as e:
            print(f"[LiveStreamer] Audio hardware error: {e}")

        # Try each model in the fallback list
        connected = False
        for model_name in self.LIVE_MODELS:
            self.model = model_name
            print(f"[LiveStreamer] Trying model: {model_name}...")
            try:
                async with self.client.aio.live.connect(model=self.model, config=config) as session: # type: ignore
                    print(f"[LiveStreamer] Connection established with '{model_name}'! Session Active.")
                    connected = True
                    self.session = session
                    self.is_streaming = True
                    
                    # Start background processes
                    tasks = [
                        asyncio.create_task(self._listen_for_responses()),
                    ]
                    if self._audio_input_stream:
                        tasks.append(asyncio.create_task(self._stream_audio_input_loop()))
                    
                    await asyncio.gather(*tasks)
                    break  # Session ended normally
                    
            except asyncio.CancelledError:
                 print("[LiveStreamer] Background session cancelled safely.")
                 break
            except Exception as e:
                error_msg = str(e).lower()
                if "1008" in error_msg or "not found" in error_msg:
                    print(f"[LiveStreamer] Model '{model_name}' not supported for bidiGenerateContent. Trying next...")
                    continue
                elif "503" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                    print(f"[LiveStreamer] Transient error with '{model_name}': {e}. Waiting to retry...")
                    await asyncio.sleep(2)
                    continue
                else:
                    print(f"[LiveStreamer] Failed during live session: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            finally:
                self.is_streaming = False
                self.session = None
                self._cleanup_audio()
        
        if not connected:
            print("[LiveStreamer] All model fallbacks exhausted. No live session established.")

    async def disconnect(self):
        """Signals the session to close."""
        self.is_streaming = False
        print("[LiveStreamer] Disconnect requested (session will organically close).")

    def _capture_screen_compressed(self) -> bytes:
        """Captures the primary monitor (or a specific ROI) using the injected screen_observer."""
        if not self.screen_observer:
             # Very early fallback if not injected yet
             with mss.mss() as sct:
                 monitor = sct.monitors[1]
                 self.screen_size = (monitor["width"], monitor["height"])
                 sct_img = sct.grab(monitor)
                 img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        else:
            if self.current_roi:
                # ROI is (x1, y1, x2, y2) in global pixels
                # ScreenObserver.capture_as_pil expects (left, top, width, height)
                x1, y1, x2, y2 = self.current_roi # type: ignore
                region = (x1, y1, x2 - x1, y2 - y1)
                img = self.screen_observer.capture_as_pil(region=region) # type: ignore
            else:
                img = self.screen_observer.capture_as_pil() # type: ignore
                self.screen_size = img.size
            
        img.thumbnail((1024, 1024))
        
        # Debug save
        if self.debug_dir:
            self._frame_count += 1
            label = "roi" if self.current_roi else "full"
            fname = f"frame_{self._frame_count:03d}_{label}.jpg"
            img.save(os.path.join(self.debug_dir, fname), format="JPEG", quality=85) # type: ignore
            
        output = BytesIO()
        img.save(output, format="JPEG", quality=60)
        return output.getvalue()

    def set_roi(self, center_x: int, center_y: int, zoom_factor: float = 2.0):
        """Sets the current region of interest based on a global center point."""
        from coding_agent.ui_agent.perception.roi_manager import ROIManager # type: ignore
        self.current_roi = ROIManager.calculate_roi(center_x, center_y, zoom_factor, self.screen_size) # type: ignore
        print(f"[LiveStreamer] ROI set to {self.current_roi} (Zoom: {zoom_factor}x)")

    def reset_roi(self):
        """Resets to full screen view."""
        self.current_roi = None
        print("[LiveStreamer] ROI reset to full screen.")

    async def stream_screen_loop(self, fps: float = 1.0):
        """Continuously captures and sends frames to the session."""
        print(f"[LiveStreamer] Starting screen stream at {fps} fps...")
        interval = 1.0 / fps
        while self.is_streaming and self.session:
            try:
                frame_bytes = self._capture_screen_compressed()
                await self.session.send( # type: ignore
                    input=types.LiveClientRealtimeInput(
                        media_chunks=[
                            types.Blob(
                                mime_type="image/jpeg",
                                data=frame_bytes
                            )
                        ]
                    )
                )
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"[LiveStreamer] Error sending frame: {e}")
                break

    async def send_frame_now(self):
        """Sends a single screenshot frame immediately (on-demand, not from the stream loop)."""
        if not self.session or not self.is_streaming:
            print("[LiveStreamer] Cannot send frame: no active session.")
            return
        try:
            frame_bytes = self._capture_screen_compressed()
            await self.session.send( # type: ignore
                input=types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(
                            mime_type="image/jpeg",
                            data=frame_bytes
                        )
                    ]
                )
            )
            print("[LiveStreamer] On-demand frame sent.")
        except Exception as e:
            print(f"[LiveStreamer] Error sending on-demand frame: {e}")

    async def send_prompt(self, text: str):
        if not self.session or not self.is_streaming:
            return
        await self.session.send( # type: ignore
            input=types.LiveClientContent(
                turns=[types.Content(role="user", parts=[types.Part.from_text(text=text)])],
                turn_complete=True
            )
        )

    async def _listen_for_responses(self):
        print("[LiveStreamer] Listener task started.")
        try:
            async for response in self.session.receive(): # type: ignore
                if response.server_content:
                    if response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.text:
                                try:
                                    json_data = json.loads(part.text)
                                    if self.on_response_callback:
                                        self.on_response_callback(json_data) # type: ignore
                                except json.JSONDecodeError:
                                    if self.on_response_callback:
                                        self.on_response_callback({"text_response": part.text}) # type: ignore
                            
                            if part.inline_data:
                                self.audio_out_queue.put(part.inline_data.data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[LiveStreamer] Error in listener loop: {e}")
            self.is_streaming = False

    def _setup_audio_streams(self):
        def input_callback(indata, frames, time, status):
            self.audio_in_queue.put(indata.copy())

        def output_callback(outdata, frames, time, status):
            try:
                data = self.audio_out_queue.get_nowait()
                decoded = np.frombuffer(data, dtype='int16')
                # Handle cases where the decoded chunk size exceeds or is less than the expected frames
                chunk_len = min(len(decoded), frames)
                outdata[:chunk_len] = decoded[:chunk_len].reshape(-1, 1)
                if chunk_len < frames:
                    outdata[chunk_len:] = 0
            except queue.Empty:
                outdata.fill(0)

        self._audio_input_stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype='int16', callback=input_callback
        )
        self._audio_output_stream = sd.OutputStream(
            samplerate=24000, channels=1, dtype='int16', callback=output_callback
        )
        self._audio_input_stream.start() # type: ignore
        self._audio_output_stream.start() # type: ignore

    async def _stream_audio_input_loop(self):
        while self.is_streaming and self.session:
            try:
                data = await asyncio.to_thread(self.audio_in_queue.get, timeout=1.0) # type: ignore
                # Using the documentation's helper pattern for real-time input
                await self.session.send_realtime_input( # type: ignore
                    audio={"data": data.tobytes(), "mime_type": "audio/pcm"}
                )
            except queue.Empty:
                continue
            except Exception as e:
                break

    def _cleanup_audio(self):
        if self._audio_input_stream:
            self._audio_input_stream.stop() # type: ignore
            self._audio_input_stream.close() # type: ignore
        if self._audio_output_stream:
            self._audio_output_stream.stop() # type: ignore
            self._audio_output_stream.close() # type: ignore

    def set_callback(self, callback: Callable[[Dict[str, Any]], None]):
        self.on_response_callback = callback

    def query_text_sync(self, prompt: str, event_loop: asyncio.AbstractEventLoop, timeout: float = 15.0) -> Optional[Dict[str, Any]]:
        """
        Synchronous blocking query: sends a text prompt to the Live API session,
        waits for the text response, and returns the parsed result.
        
        This is the primary interface for the AgenticPlanner to communicate
        with the Live API between steps.
        """
        if not self.session or not self.is_streaming:
            print("[LiveStreamer] Cannot query: no active session.")
            return None
        
        async def _async_query():
            response_event = asyncio.Event()
            result_holder = {}
            original_callback = self.on_response_callback
            
            def _capture_callback(data: Dict[str, Any]):
                result_holder.update(data)
                response_event.set()
            
            self.on_response_callback = _capture_callback
            
            try:
                # Send a fresh frame first so Gemini has current context
                await self.send_frame_now()
                await asyncio.sleep(0.5)
                
                # Send the text prompt
                await self.send_prompt(prompt)
                
                # Wait for response
                await asyncio.wait_for(response_event.wait(), timeout=timeout)
                return result_holder
            except asyncio.TimeoutError:
                print(f"[LiveStreamer] Sync query timed out after {timeout}s")
                return {"timeout": True, "text_response": ""}
            except Exception as e:
                print(f"[LiveStreamer] Sync query error: {e}")
                return None
            finally:
                self.on_response_callback = original_callback
        
        try:
            future = asyncio.run_coroutine_threadsafe(_async_query(), event_loop)
            return future.result(timeout=timeout + 5) # type: ignore
        except Exception as e:
            print(f"[LiveStreamer] query_text_sync failed: {e}")
            return None

    def send_step_feedback(self, feedback_text: str, event_loop: asyncio.AbstractEventLoop):
        """
        Sends a text feedback message to the Live API after each agentic step.
        This steers the Live API's understanding of what's happening without
        expecting a response.
        """
        if not self.session or not self.is_streaming:
            return
        
        async def _send():
            try:
                await self.send_frame_now()
                await asyncio.sleep(0.3)
                await self.send_prompt(feedback_text)
                print(f"[LiveStreamer] Step feedback sent: {feedback_text[:80]}...") # type: ignore
            except Exception as e:
                print(f"[LiveStreamer] Step feedback error: {e}")
        
        try:
            asyncio.run_coroutine_threadsafe(_send(), event_loop)
        except Exception as e:
            print(f"[LiveStreamer] send_step_feedback failed: {e}")
