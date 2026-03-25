import tkinter as tk
from threading import Thread
import threading
import time
import queue
from typing import Optional, Any, Dict, List, Tuple, Union
from PIL import Image, ImageTk # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore
from coding_agent.utils import config


class HUDOverlay:
    """
    A transparent, always-on-top overlay to display agent status and intent.
    Implemented as a singleton to ensure consistency across multiple subagent instances.
    """
    _instance: Optional['HUDOverlay'] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(HUDOverlay, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.root: Optional[tk.Tk] = None
        self.panel: Optional[tk.Toplevel] = None
        self.canvas_win: Optional[tk.Toplevel] = None
        self.label_goal: Optional[tk.Label] = None
        self.label_action: Optional[tk.Label] = None
        self.label_status: Optional[tk.Label] = None
        self.label_step: Optional[tk.Label] = None
        self.canvas: Optional[tk.Canvas] = None
        self.crosshair = None
        self._bg_image_id: Optional[int] = None
        self._bg_photo: Optional[ImageTk.PhotoImage] = None
        self._update_queue: queue.Queue = queue.Queue()
        
        # Enable DPI Awareness for high-res displays
        try:
            import ctypes
            # type: ignore[attr-defined]
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
            
        self.goal_text = "Goal: Initializing..."
        self.action_text = "Action: Idle"
        self.status_text = "Live Stream: Offline"
        self.step_text = "Step: --"
        self.status_color = "#FF4444"
        
        self.enabled = (config.HUD_ENABLED == 1)
        if not self.enabled:
            print("[HUD] Disabled via config.")
            return

        self._thread = Thread(target=self._run_tk, daemon=True)
        self._thread.start()
        
    def _run_tk(self):
        root = tk.Tk()
        self.root = root
        root.title("UI Agent HUD")
        
        # Make transparent and always on top
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", "black")
        root.overrideredirect(True) # Remove title bar
        
        sw = root.winfo_screenwidth()
        root.geometry(f"420x180+{sw - 440}+20")
        root.config(bg="black")
        
        # Status Labels
        font_style = ("Consolas", 12, "bold")
        
        label_goal = tk.Label(root, text=self.goal_text, fg="#00FF00", bg="black", font=font_style, wraplength=380, justify="left")
        label_goal.pack(anchor="w", padx=10, pady=2)
        self.label_goal = label_goal
        
        label_action = tk.Label(root, text=self.action_text, fg="#00FFFF", bg="black", font=font_style, wraplength=380, justify="left")
        label_action.pack(anchor="w", padx=10, pady=2)
        self.label_action = label_action
        
        label_status = tk.Label(root, text=self.status_text, fg="#FF4444", bg="black", font=font_style)
        label_status.pack(anchor="w", padx=10, pady=2)
        self.label_status = label_status
        
        label_step = tk.Label(root, text=self.step_text, fg="#FFD700", bg="black", font=("Consolas", 10), wraplength=400, justify="left")
        label_step.pack(anchor="w", padx=10, pady=2)
        self.label_step = label_step
        
        # 2. Global Canvas (Full Screen for Boundaries)
        canvas_win = tk.Toplevel(root)
        self.canvas_win = canvas_win
        canvas_win.attributes("-topmost", True)
        canvas_win.attributes("-transparentcolor", "black")
        canvas_win.overrideredirect(True)
        canvas_win.config(bg="black")
        sh = root.winfo_screenheight()
        canvas_win.geometry(f"{sw}x{sh}+0+0")
        
        canvas = tk.Canvas(canvas_win, bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        self.canvas = canvas

        self._update_loop()
        root.mainloop()

    def _update_loop(self) -> None:
        root = self.root
        l_goal = self.label_goal
        l_action = self.label_action
        l_status = self.label_status
        l_step = self.label_step
        
        # 1. Process Message Queue (Thread-Safe Updates)
        while not self._update_queue.empty():
            try:
                task = self._update_queue.get_nowait()
                cmd = task.get("cmd")
                data = task.get("data", {})
                
                if cmd == "draw_rect":
                    self._draw_rect(**data)
                elif cmd == "clear":
                    if self.canvas: self.canvas.delete("all")
                elif cmd == "draw_image":
                    self._draw_image(data.get("image"))
            except queue.Empty:
                break
            except Exception as e:
                print(f"[HUD:Queue] Error: {e}")

        # 2. Update Labels (Periodic Sync)
        if (root is not None and 
            self.label_goal is not None and 
            self.label_action is not None and 
            self.label_status is not None and 
            self.label_step is not None):
            
            self.label_goal.config(text=self.goal_text)
            self.label_action.config(text=self.action_text)
            self.label_status.config(text=self.status_text, fg=self.status_color)
            self.label_step.config(text=self.step_text)
            
            # Use type: ignore for after() as Pyre is being overly strict with the signature
            root.after(50, self._update_loop) # type: ignore

    def update_goal(self, text: str):
        if not getattr(self, "enabled", True): return
        self.goal_text = f"Goal: {text}"
        # Reset other fields for a new goal
        self.action_text = "Action: Idle"
        self.step_text = "Step: --"

    def update_action(self, text: str):
        if not getattr(self, "enabled", True): return
        self.action_text = f"Action: {text}"

    def update_status(self, is_streaming: bool, fallback_active: bool = False):
        if not getattr(self, "enabled", True): return
        if is_streaming:
            status = "ONLINE"
            self.status_color = "#00FF00" 
        elif fallback_active:
            status = "Static VLM Active"
            self.status_color = "#FFA500"
        else:
            status = "Offline"
            self.status_color = "#FF4444"
        self.status_text = f"Live Stream: {status}"

    def update_step(self, step_num: int, reasoning: str = ""):
        if not getattr(self, "enabled", True): return
        if reasoning:
            res_str = str(reasoning)
            if len(res_str) > 60:
                # Type ignore indexing as Pyre is failing to recognize str slicing
                slice_val = res_str[0:60] # type: ignore
                self.step_text = f"Step {step_num}: {slice_val}..."
            else:
                self.step_text = f"Step {step_num}: {res_str}"
        else:
            self.step_text = f"Step {step_num}"

    def add_rect(self, x: float, y: float, w: float, h: float, label: str = "", color: Tuple[int, int, int] = (0, 255, 0)) -> None:
        """Draws a boundary rectangle on the global canvas. Thread-safe."""
        if not getattr(self, "enabled", True): return
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        self._update_queue.put({
            "cmd": "draw_rect",
            "data": {
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "label": label, "color": hex_color
            }
        })

    def _draw_rect(self, x: int, y: int, w: int, h: int, label: str, color: str) -> None:
        canvas = self.canvas
        root = self.root
        if canvas is not None and root is not None:
            rect_id = canvas.create_rectangle(x, y, x+w, y+h, outline=color, width=3)
            if label:
                 text_id = canvas.create_text(x, y-10, text=label, fill=color, font=("Consolas", 10, "bold"), anchor="sw")
                 root.after(4000, lambda: canvas.delete(rect_id, text_id)) # type: ignore
            else:
                 root.after(4000, lambda: canvas.delete(rect_id)) # type: ignore

    def display_image(self, image: np.ndarray):
        """Displays a full-screen image (e.g., edge map) on the HUD canvas. Thread-safe."""
        if not getattr(self, "enabled", True): return
        self._update_queue.put({"cmd": "draw_image", "data": {"image": image}})

    def _draw_image(self, image: np.ndarray):
        # ... (implementation same as before, but called in UI thread)
        canvas = self.canvas
        if canvas is not None:
            # Convert OpenCV (numpy) to PIL
            if len(image.shape) == 2:
                # Grayscale
                pil_img = Image.fromarray(image)
            else:
                # BGR to RGB
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # type: ignore
            
            # Scale to fit screen if necessary (assuming HUD is full screen)
            sw = canvas.winfo_width()
            sh = canvas.winfo_height()
            if sw > 1 and sh > 1:
                pil_img = pil_img.resize((sw, sh), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            self._bg_photo = ImageTk.PhotoImage(image=pil_img)
            
            # Clear previous image if exists
            bg_id = self._bg_image_id
            if bg_id is not None:
                canvas.delete(bg_id)
            
            # Draw at 0,0
            # Note: Tkinter doesn't support true alpha blending for images on canvas easily 
            # while keeping the window transparent. But since 'black' is the transparent color
            # for the window, black pixels in the image will be transparent.
            new_id = canvas.create_image(0, 0, image=self._bg_photo, anchor="nw")
            self._bg_image_id = new_id
            canvas.tag_lower(new_id)

    def clear(self) -> None:
        if not getattr(self, "enabled", True): return
        self._update_queue.put({"cmd": "clear"})

    def update(self):
        """No-op for compatibility."""
        pass

    def stop(self):
        if not getattr(self, "enabled", True): return
        if self.root:
            self.root.quit() # type: ignore

if __name__ == "__main__":
    # Test standalone
    hud = HUDOverlay()
    time.sleep(2)
    hud.update_goal("Open CMD and cd Downloads")
    hud.update_action("Clicking Start Button")
    hud.update_status(True)
    time.sleep(5)
    hud.stop()
