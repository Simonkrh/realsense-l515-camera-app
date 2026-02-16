import os
import subprocess
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    rs = None


class RealSenseCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense L515 Camera App")

        # Output folder
        self.save_dir = filedialog.askdirectory(title="Choose folder to save images")
        if not self.save_dir:
            self.save_dir = os.path.join(os.getcwd(), "captures")
        os.makedirs(self.save_dir, exist_ok=True)

        # Filename counter
        self.counter = self._next_index(self.save_dir, prefix="img_", ext=".jpg")

        # UI
        self.image_label = tk.Label(root)
        self.image_label.pack(padx=10, pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.capture_btn = tk.Button(
            btn_frame, text="Capture (SPACE)", command=self.capture
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        self.quit_btn = tk.Button(btn_frame, text="Quit (ESC)", command=self.close)
        self.quit_btn.pack(side=tk.LEFT, padx=5)

        self.status = tk.Label(root, text=f"Saving to: {self.save_dir}")
        self.status.pack(padx=10, pady=5)
        self.stream_info = tk.Label(root, text="Stream: initializing...")
        self.stream_info.pack(padx=10, pady=(0, 8))

        # Key bindings
        root.bind("<space>", lambda e: self.capture())
        root.bind("<Escape>", lambda e: self.close())

        # Prefer pyrealsense2, fallback to V4L2 camera nodes.
        self.backend = None
        self.pipeline_started = False
        self.pipeline = None
        self.cap = None

        realsense_error = None
        if rs is not None:
            try:
                self._start_realsense()
            except Exception as exc:
                realsense_error = str(exc)
        else:
            realsense_error = "Missing dependency: pyrealsense2. Install with `python3 -m pip install --user pyrealsense2`."

        if self.backend is None:
            self.cap = self._start_v4l_fallback()
            if self.cap is None:
                raise RuntimeError(
                    f"Failed to start RealSense pipeline: {realsense_error}. "
                    "No usable /dev/video* fallback device found."
                )
            self.backend = "v4l2"
            messagebox.showwarning(
                "RealSense Backend Warning",
                f"RealSense SDK unavailable ({realsense_error}). Using V4L2 fallback.",
            )

        self.status.configure(
            text=f"Saving to: {self.save_dir} | backend: {self.backend}"
        )
        self._last_stream_info = None

        self.last_frame = None
        self.running = True
        self.update_frame()

    def _next_index(self, folder, prefix="img_", ext=".jpg"):
        max_i = -1
        for name in os.listdir(folder):
            if name.startswith(prefix) and name.endswith(ext):
                mid = name[len(prefix) : -len(ext)]
                if mid.isdigit():
                    max_i = max(max_i, int(mid))
        return max_i + 1

    def update_frame(self):
        if not self.running:
            return

        frame = self._read_frame()
        if frame is not None:
            self.last_frame = frame
            h, w = frame.shape[:2]
            channels = frame.shape[2] if len(frame.shape) == 3 else 1
            info = f"Stream: {w}x{h} | channels: {channels} | backend: {self.backend}"
            if info != self._last_stream_info:
                self.stream_info.configure(text=info)
                self._last_stream_info = info

            # Convert for Tkinter display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def capture(self):
        if self.last_frame is None:
            return

        filename = os.path.join(self.save_dir, f"img_{self.counter:05d}.jpg")
        cv2.imwrite(filename, self.last_frame)
        self.status.configure(text=f"Saved: {filename} | backend: {self.backend}")
        self.counter += 1

    def close(self):
        self.running = False
        try:
            if self.pipeline_started:
                self.pipeline.stop()
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()

    def _start_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.pipeline_started = True
        self.backend = "realsense"

    def _read_frame(self):
        if self.backend == "realsense":
            frames = self.pipeline.poll_for_frames()
            if not frames:
                return None
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            return np.asanyarray(color_frame.get_data())  # BGR

        if self.backend == "v4l2" and self.cap is not None:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
        return None

    def _start_v4l_fallback(self):
        candidates = self._sorted_v4l2_candidates()
        for idx in candidates:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue

            # Ask OpenCV to deliver BGR frames even when camera outputs YUYV/MJPG.
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            cap.release()
        return None

    def _realsense_video_indices(self):
        sys_path = "/sys/class/video4linux"
        if not os.path.isdir(sys_path):
            return list(range(10))

        indices = []
        for name in os.listdir(sys_path):
            if not name.startswith("video"):
                continue
            idx_txt = name.replace("video", "")
            if not idx_txt.isdigit():
                continue
            dev_name_path = os.path.join(sys_path, name, "name")
            try:
                with open(dev_name_path, "r", encoding="utf-8") as f:
                    dev_name = f.read()
            except OSError:
                continue
            if "RealSense" in dev_name:
                indices.append(int(idx_txt))

        if not indices:
            return list(range(10))
        return sorted(indices)

    def _sorted_v4l2_candidates(self):
        indices = self._realsense_video_indices()
        scored = []
        for idx in indices:
            formats = self._v4l2_formats(idx)
            score = self._format_score(formats)
            scored.append((score, idx))

        # Highest score first, then lower device index.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _, idx in scored]

    def _v4l2_formats(self, idx):
        try:
            out = subprocess.check_output(
                ["v4l2-ctl", "-d", f"/dev/video{idx}", "--list-formats-ext"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return []

        formats = []
        for line in out.splitlines():
            line = line.strip()
            if not line.startswith("["):
                continue
            if "'" not in line:
                continue
            fmt = line.split("'")[1].strip()
            if fmt:
                formats.append(fmt.upper())
        return formats

    def _format_score(self, formats):
        # Prefer color streams and penalize depth/IR streams
        if not formats:
            return 0

        score = 0
        color_formats = {"YUYV", "MJPG", "RGB3", "BGR3", "UYVY", "NV12"}
        bad_formats = {"Z16", "GREY", "Y8I", "Y12I", "CNF4", "INZI", "INVR"}

        for fmt in formats:
            if fmt in color_formats:
                score += 10
            if fmt in bad_formats:
                score -= 10
        return score


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = RealSenseCameraApp(root)
    except RuntimeError as exc:
        messagebox.showerror("Startup Error", str(exc))
        root.destroy()
        raise SystemExit(1)
    root.mainloop()
