import os
import subprocess
import sys
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
        self.root.title("Camera Capture App")

        # Output folder
        self.save_dir = filedialog.askdirectory(title="Choose folder to save images")
        if not self.save_dir:
            self.save_dir = os.path.join(os.getcwd(), "captures")
        os.makedirs(self.save_dir, exist_ok=True)

        # Filename counter
        self.counter = self._next_index(self.save_dir, prefix="img_", ext=".jpg")
        self.capture_count = 0

        # UI
        self.image_label = tk.Label(root)
        self.image_label.pack(padx=10, pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.capture_btn = tk.Button(
            btn_frame, text="Capture (SPACE)", command=self.capture
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        self.name_label = tk.Label(btn_frame, text="File name:")
        self.name_label.pack(side=tk.LEFT, padx=(10, 4))
        self.name_entry = tk.Entry(btn_frame, width=24)
        self.name_entry.pack(side=tk.LEFT, padx=4)

        self.quit_btn = tk.Button(btn_frame, text="Quit (ESC)", command=self.close)
        self.quit_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(
            btn_frame, text="Reset Counter", command=self.reset_counter
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        self.select_camera_btn = tk.Button(
            btn_frame, text="Select Camera", command=self.select_camera
        )
        self.select_camera_btn.pack(side=tk.LEFT, padx=5)

        self.status = tk.Label(root, text=f"Saving to: {self.save_dir}")
        self.status.pack(padx=10, pady=5)
        self.counter_label = tk.Label(root, text=f"Captured this session: {self.capture_count}")
        self.counter_label.pack(padx=10, pady=(0, 5))
        self.stream_info = tk.Label(root, text="Stream: initializing...")
        self.stream_info.pack(padx=10, pady=(0, 8))

        # Key bindings
        root.bind("<space>", lambda e: self.capture())
        root.bind("<Escape>", lambda e: self.close())

        # Prefer pyrealsense2, fallback to V4L2 camera nodes.
        self.backend = None
        self.pipeline_started = False
        self.pipeline = None
        self.realsense_color_format = None
        self.realsense_device_name = None
        self.realsense_device_serial = None
        self.cap = None
        self.opencv_index = None
        self.camera_source_id = None
        self.device_label_text = "Device: (none)"

        self.device_label = tk.Label(root, text=self.device_label_text)
        self.device_label.pack(padx=10, pady=(0, 5))

        self._start_camera_interactive(initial=True)

        self.status.configure(
            text=f"Saving to: {self.save_dir} | backend: {self.backend}"
        )
        self._last_stream_info = None

        self.last_frame = None
        self.running = True
        self.update_frame()

    def _camera_sources(self):
        sources = [
            {
                "id": "auto",
                "kind": "auto",
                "label": "Auto (prefer RealSense, fallback to OpenCV)",
            },
            {
                "id": "opencv:auto",
                "kind": "opencv_auto",
                "label": "Auto (OpenCV only)",
            },
        ]

        for dev in self._realsense_devices():
            serial = dev.get("serial")
            if not serial:
                continue
            name = dev.get("name") or "RealSense"
            label = f"RealSense: {name} (S/N: {serial})"
            sources.append(
                {
                    "id": f"realsense:{serial}",
                    "kind": "realsense",
                    "serial": serial,
                    "label": label,
                }
            )

        if sys.platform.startswith("linux"):
            opencv_devices = self._linux_v4l2_devices()
            if not opencv_devices:
                opencv_devices = [{"index": i, "name": None} for i in range(10)]
        else:
            opencv_devices = [{"index": i, "name": None} for i in self._probe_opencv_indices()]

        for dev in opencv_devices:
            idx = dev["index"]
            name = dev.get("name")
            label = f"OpenCV: camera index {idx}"
            if sys.platform.startswith("linux") and os.path.exists(f"/dev/video{idx}"):
                label = f"OpenCV: /dev/video{idx}"
            if name:
                label = f"{label} ({name})"
            sources.append({"id": f"opencv:{idx}", "kind": "opencv", "index": idx, "label": label})

        return sources

    def _prompt_camera_source(self, sources, selected_id=None):
        top = tk.Toplevel(self.root)
        top.title("Select camera")
        top.transient(self.root)
        top.grab_set()
        top.resizable(False, False)

        tk.Label(top, text="Choose camera source:").pack(padx=10, pady=(10, 0), anchor="w")

        list_frame = tk.Frame(top)
        list_frame.pack(padx=10, pady=8, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        listbox = tk.Listbox(
            list_frame,
            width=70,
            height=min(14, max(6, len(sources))),
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for src in sources:
            listbox.insert(tk.END, src["label"])

        preselect = 0
        if selected_id:
            for i, src in enumerate(sources):
                if src["id"] == selected_id:
                    preselect = i
                    break
        listbox.selection_set(preselect)
        listbox.activate(preselect)
        listbox.see(preselect)

        result = {"index": None}

        def on_ok(event=None):
            sel = listbox.curselection()
            if not sel:
                return
            result["index"] = int(sel[0])
            top.destroy()

        def on_cancel(event=None):
            result["index"] = None
            top.destroy()

        btns = tk.Frame(top)
        btns.pack(padx=10, pady=(0, 10), fill=tk.X)
        tk.Button(btns, text="Use camera", command=on_ok).pack(side=tk.LEFT)
        tk.Button(btns, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=8)

        listbox.bind("<Double-Button-1>", on_ok)
        top.bind("<Return>", on_ok)
        top.bind("<Escape>", on_cancel)
        top.protocol("WM_DELETE_WINDOW", on_cancel)

        self.root.wait_window(top)
        if result["index"] is None:
            return None
        return sources[result["index"]]

    def _stop_camera(self):
        try:
            if self.pipeline_started and self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        self.pipeline_started = False
        self.pipeline = None
        self.realsense_color_format = None
        self.realsense_device_name = None
        self.realsense_device_serial = None

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        self.opencv_index = None

        self.backend = None
        self.camera_source_id = None
        self.last_frame = None
        self.device_label_text = "Device: (none)"
        self.device_label.configure(text=self.device_label_text)

    def _start_camera_from_source(self, source):
        kind = source["kind"]

        if kind == "auto":
            realsense_error = None
            if rs is not None and self._realsense_devices():
                try:
                    self._start_realsense(serial=None)
                    return
                except Exception as exc:
                    realsense_error = str(exc)

            try:
                self._start_opencv_auto()
            except Exception as exc:
                if realsense_error:
                    raise RuntimeError(
                        f"RealSense could not be started:\n{realsense_error}\n\nOpenCV could not be started:\n{exc}"
                    ) from exc
                raise
            if realsense_error:
                messagebox.showwarning(
                    "RealSense not started",
                    f"RealSense could not be started:\n\n{realsense_error}\n\nUsing OpenCV camera instead.",
                )
            return

        if kind == "opencv_auto":
            self._start_opencv_auto()
            return

        if kind == "realsense":
            if rs is None:
                raise RuntimeError(
                    "pyrealsense2 is not installed. Install with `python3 -m pip install --user pyrealsense2`."
                )
            self._start_realsense(serial=source.get("serial"))
            return

        if kind == "opencv":
            self._start_opencv_index(source["index"])
            return

        raise RuntimeError(f"Unknown camera source kind: {kind}")

    def _start_camera_interactive(self, initial):
        sources = self._camera_sources()
        current_id = self.camera_source_id or "auto"

        previous_id = self.camera_source_id

        while True:
            selected = self._prompt_camera_source(sources, selected_id=current_id)
            if selected is None:
                if initial:
                    # User cancelled at startup; behave like Auto.
                    selected = next((s for s in sources if s["id"] == "auto"), None)
                    if selected is None:
                        raise RuntimeError("No camera sources found.")
                else:
                    return

            if not initial:
                previous_id = self.camera_source_id

            self._stop_camera()
            try:
                self._start_camera_from_source(selected)
                self.camera_source_id = selected["id"]
                return
            except Exception as exc:
                messagebox.showerror("Camera Error", f"Failed to start selected camera:\n\n{exc}")
                # Try to restore previous camera when switching.
                if not initial and previous_id:
                    prev = next((s for s in sources if s["id"] == previous_id), None)
                    if prev is not None:
                        try:
                            self._start_camera_from_source(prev)
                            self.camera_source_id = prev["id"]
                            current_id = prev["id"]
                        except Exception:
                            pass

                current_id = selected["id"]

    def select_camera(self):
        self._start_camera_interactive(initial=False)

    def _realsense_devices(self):
        if rs is None:
            return []

        try:
            ctx = rs.context()
            queried = ctx.query_devices()
        except Exception:
            return []

        devices = []
        for dev in queried:
            try:
                name = dev.get_info(rs.camera_info.name)
            except Exception:
                name = "RealSense"
            try:
                serial = dev.get_info(rs.camera_info.serial_number)
            except Exception:
                serial = None
            try:
                product_line = dev.get_info(rs.camera_info.product_line)
            except Exception:
                product_line = None
            devices.append(
                {
                    "name": name,
                    "serial": serial,
                    "product_line": product_line,
                }
            )
        return devices

    def _linux_v4l2_devices(self):
        sys_path = "/sys/class/video4linux"
        if not os.path.isdir(sys_path):
            return []

        devices = []
        for entry in os.listdir(sys_path):
            if not entry.startswith("video"):
                continue
            idx_txt = entry.replace("video", "")
            if not idx_txt.isdigit():
                continue
            idx = int(idx_txt)
            devnode = f"/dev/video{idx}"
            if not os.path.exists(devnode):
                continue
            name_path = os.path.join(sys_path, entry, "name")
            dev_name = None
            try:
                with open(name_path, "r", encoding="utf-8") as f:
                    dev_name = f.read().strip()
            except OSError:
                dev_name = None
            devices.append({"index": idx, "name": dev_name})

        devices.sort(key=lambda d: d["index"])
        return devices

    def _opencv_preferred_backend(self):
        if sys.platform.startswith("linux"):
            return getattr(cv2, "CAP_V4L2", None)
        if sys.platform == "win32":
            return getattr(cv2, "CAP_DSHOW", None)
        if sys.platform == "darwin":
            return getattr(cv2, "CAP_AVFOUNDATION", None)
        return None

    def _open_opencv_capture(self, index):
        backend = self._opencv_preferred_backend()
        if backend is not None:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(index)

    def _probe_opencv_indices(self, max_index=10):
        indices = []
        for idx in range(max_index):
            cap = self._open_opencv_capture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                indices.append(idx)
        return indices

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

        base_name = self.name_entry.get().strip()
        if not base_name:
            filename = self._next_default_filename()
        else:
            filename = self._unique_filename(base_name, ext=".jpg")
        if cv2.imwrite(filename, self.last_frame):
            self.capture_count += 1
            self.counter_label.configure(
                text=f"Captured this session: {self.capture_count}"
            )
            self.status.configure(text=f"Saved: {filename} | backend: {self.backend}")
        else:
            self.status.configure(text=f"Failed to save: {filename} | backend: {self.backend}")

    def reset_counter(self):
        self.capture_count = 0
        self.counter_label.configure(text=f"Captured this session: {self.capture_count}")

    def _next_default_filename(self):
        filename = os.path.join(self.save_dir, f"img_{self.counter:05d}.jpg")
        while os.path.exists(filename):
            self.counter += 1
            filename = os.path.join(self.save_dir, f"img_{self.counter:05d}.jpg")
        self.counter += 1
        return filename

    def _unique_filename(self, base_name, ext=".jpg"):
        base_name = os.path.basename(base_name)
        if base_name.lower().endswith(ext):
            base_name = base_name[: -len(ext)]
        base_name = base_name.strip() or "img"

        candidate = os.path.join(self.save_dir, f"{base_name}{ext}")
        suffix = 1
        while os.path.exists(candidate):
            candidate = os.path.join(self.save_dir, f"{base_name}_{suffix}{ext}")
            suffix += 1
        return candidate

    def close(self):
        self.running = False
        self._stop_camera()
        self.root.destroy()

    def _start_realsense(self, serial=None):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed.")

        attempts = [
            (1280, 720, 30, rs.format.bgr8),
            (1280, 720, 30, rs.format.rgb8),
            (640, 480, 30, rs.format.bgr8),
            (640, 480, 30, rs.format.rgb8),
            (1280, 720, 15, rs.format.bgr8),
            (1280, 720, 15, rs.format.rgb8),
            (640, 480, 15, rs.format.bgr8),
            (640, 480, 15, rs.format.rgb8),
        ]

        last_exc = None
        for width, height, fps, fmt in attempts:
            pipeline = rs.pipeline()
            config = rs.config()
            if serial:
                config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, fmt, fps)

            try:
                profile = pipeline.start(config)
            except Exception as exc:
                last_exc = exc
                try:
                    pipeline.stop()
                except Exception:
                    pass
                continue

            dev = profile.get_device()
            try:
                dev_name = dev.get_info(rs.camera_info.name)
            except Exception:
                dev_name = "RealSense"
            try:
                dev_serial = dev.get_info(rs.camera_info.serial_number)
            except Exception:
                dev_serial = serial

            self.pipeline = pipeline
            self.pipeline_started = True
            self.backend = "realsense"
            self.realsense_color_format = fmt
            self.realsense_device_name = dev_name
            self.realsense_device_serial = dev_serial

            label = f"Device: RealSense {dev_name}"
            if dev_serial:
                label = f"{label} (S/N: {dev_serial})"
            self.device_label_text = label
            self.device_label.configure(text=self.device_label_text)
            self.status.configure(text=f"Saving to: {self.save_dir} | backend: {self.backend}")
            return

        suffix = f" (S/N: {serial})" if serial else ""
        raise RuntimeError(f"Failed to start RealSense camera{suffix}: {last_exc}")

    def _read_frame(self):
        if self.backend == "realsense":
            if rs is None or self.pipeline is None:
                return None
            frames = self.pipeline.poll_for_frames()
            if not frames:
                return None
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            frame = np.asanyarray(color_frame.get_data())
            try:
                fmt = color_frame.get_profile().format()
            except Exception:
                fmt = self.realsense_color_format
            if fmt == rs.format.rgb8:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame  # BGR

        if self.backend == "opencv" and self.cap is not None:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
        return None

    def _configure_opencv_capture(self, cap):
        # Ask OpenCV to deliver BGR frames even when camera outputs YUYV/MJPG.
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

    def _start_opencv_index(self, idx):
        cap = self._open_opencv_capture(idx)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open OpenCV camera index {idx}.")

        self._configure_opencv_capture(cap)
        frame = None
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        if frame is None:
            cap.release()
            raise RuntimeError(f"OpenCV camera index {idx} opened but returned no frames.")

        self.cap = cap
        self.backend = "opencv"
        self.opencv_index = idx

        label = f"Device: OpenCV camera index {idx}"
        if sys.platform.startswith("linux") and os.path.exists(f"/dev/video{idx}"):
            label = f"Device: /dev/video{idx}"
            for dev in self._linux_v4l2_devices():
                if dev["index"] == idx and dev.get("name"):
                    label = f"{label} ({dev['name']})"
                    break

        self.device_label_text = label
        self.device_label.configure(text=self.device_label_text)
        self.status.configure(text=f"Saving to: {self.save_dir} | backend: {self.backend}")

    def _start_opencv_auto(self):
        candidates = self._sorted_v4l2_candidates()
        for idx in candidates:
            try:
                self._start_opencv_index(idx)
                return idx
            except Exception:
                continue
        raise RuntimeError("No usable OpenCV camera found.")

    def _v4l2_video_indices(self):
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
            idx = int(idx_txt)
            if os.path.exists(f"/dev/video{idx}"):
                indices.append(idx)

        return sorted(indices) if indices else list(range(10))

    def _sorted_v4l2_candidates(self):
        indices = self._v4l2_video_indices()
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
