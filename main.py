import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import openai
import queue
import tempfile
import soundfile as sf
from io import BytesIO
import datetime
import sounddevice as sd
from faster_whisper import WhisperModel
import scipy.signal

# Load the Together AI key from a local, git-ignored .env file.
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
openai.api_key = os.getenv("TOGETHER_API_KEY")
openai.api_base = "https://api.together.xyz/v1"
if not openai.api_key:
    raise RuntimeError("Set TOGETHER_API_KEY in the local .env file before starting the app.")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1000, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1000),
        )
    def forward(self, x):
        return self.net(x)

def simple_reconstruct(x, anomaly_type, desc):
    x_recon = x.copy()
    if anomaly_type in ['short_circuit', 'open_conductor', 'power_interruption']:
        if ':' in desc:
            start, end = map(int, desc.split()[-1].split(':'))
            if start > 0 and end < len(x_recon):
                x_recon[start:end] = np.interp(np.arange(start, end), [start-1, end], [x_recon[start-1], x_recon[end]])
    elif anomaly_type == 'voltage_sag':
        if ':' in desc:
            start, end = map(int, desc.split()[-1].split(':'))
            x_recon[start:end] /= 0.7
    elif anomaly_type == 'voltage_swell':
        if ':' in desc:
            start, end = map(int, desc.split()[-1].split(':'))
            x_recon[start:end] /= 1.3
    elif anomaly_type == 'switching_transient':
        x_recon = scipy.signal.medfilt(x_recon, kernel_size=11)
    elif anomaly_type == 'harmonics':
        b, a = scipy.signal.butter(4, 0.15)
        x_recon = scipy.signal.filtfilt(b, a, x_recon)
    elif anomaly_type == 'voltage_spike':
        if 'at' in desc:
            idx = int(desc.split('at')[1].split()[0])
            if 1 < idx < len(x_recon)-2:
                x_recon[idx] = (x_recon[idx-1] + x_recon[idx+1]) / 2
    return x_recon

def inject_td_anomaly(x, t, sampling_rate=1000):
    anomaly_type = np.random.choice([
        'short_circuit', 'voltage_sag', 'voltage_swell', 'open_conductor',
        'switching_transient', 'harmonics', 'voltage_spike', 'power_interruption'
    ])
    desc = "Normal"
    if anomaly_type == 'short_circuit':
        start = np.random.randint(200, 700)
        end = start + int(0.02 * sampling_rate)
        x[start:end] = 0
        desc = f"Short circuit {start}:{end}"
    elif anomaly_type == 'voltage_sag':
        start = np.random.randint(200, 700)
        end = start + int(0.1 * sampling_rate)
        x[start:end] *= 0.7
        desc = f"Voltage sag {start}:{end}"
    elif anomaly_type == 'voltage_swell':
        start = np.random.randint(200, 700)
        end = start + int(0.1 * sampling_rate)
        x[start:end] *= 1.3
        desc = f"Voltage swell {start}:{end}"
    elif anomaly_type == 'open_conductor':
        start = np.random.randint(200, 700)
        end = start + int(0.05 * sampling_rate)
        x[start:end] = 0
        desc = f"Open conductor {start}:{end}"
    elif anomaly_type == 'switching_transient':
        idx = np.random.randint(200, 800)
        x[idx:idx+10] += np.linspace(0, 2, 10)
        x[idx+10:idx+20] -= np.linspace(2, 0, 10)
        desc = f"Switching transient at {idx}"
    elif anomaly_type == 'harmonics':
        x += 0.2*np.sin(2*np.pi*150*t) + 0.1*np.sin(2*np.pi*250*t)
        desc = "Harmonics (3rd, 5th) injected"
    elif anomaly_type == 'voltage_spike':
        idx = np.random.randint(200, 800)
        x[idx] += 3.0
        desc = f"Voltage spike at {idx}"
    elif anomaly_type == 'power_interruption':
        start = np.random.randint(200, 700)
        end = start + int(0.1 * sampling_rate)
        x[start:end] = 0
        desc = f"Power interruption {start}:{end}"
    else:
        desc = "Normal"
    return x, anomaly_type, desc

audio_queue = queue.Queue()
sample_rate = 16000
whisper_model = WhisperModel("base", device="cpu")

def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    import os
    try: os.remove(file_path)
    except: pass
    audio_stream = BytesIO(audio_bytes)
    segments, _ = whisper_model.transcribe(audio_stream)
    full_text = " ".join([seg.text for seg in segments])
    return full_text.strip()

def voice_llm_response(user_text):
    prompt = (
        f"You are an expert assistant for transmission/distribution system anomaly restoration.\n"
        f"User question: {user_text}\n"
        "Give a short, clear, technical recommendation or answer."
    )
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response["choices"][0]["message"]["content"].strip()
        return answer
    except Exception as e:
        return f"LLM error: {e}"

class DashboardApp:
    def __init__(self, root, generator):
        self.root = root
        self.generator = generator
        self.counter = 1
        self.inject_anomaly = False
        self.anomaly_records = []

        self.sampling_rate = 1000
        self.signal_length = 10000  # 10 seconds window
        self.window = 1000  # display last 1000 samples (1 second)
        self.t = np.linspace(0, 10, self.signal_length)
        self.signal_orig = np.sin(2 * np.pi * 50 * self.t)
        self.signal_anom = self.signal_orig.copy()

        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        btn_start = ttk.Button(button_frame, text="Start", command=self.start_stream)
        btn_start.pack(side=tk.LEFT, padx=5)
        btn_inject = ttk.Button(button_frame, text="Inject Anomaly", command=self.toggle_anomaly)
        btn_inject.pack(side=tk.LEFT, padx=5)
        btn_diag = ttk.Button(button_frame, text="🧠 Diagnosis", command=self.generate_diagnosis)
        btn_diag.pack(side=tk.LEFT, padx=5)
        btn_exit = ttk.Button(button_frame, text="Exit", command=self.root.destroy)
        btn_exit.pack(side=tk.LEFT, padx=5)

        self.table = ttk.Treeview(self.root, columns=("Time", "Anomaly", "Description", "Recommendation"), show='headings', height=7)
        for col in self.table["columns"]:
            self.table.heading(col, text=col)
            self.table.column(col, anchor='center', width=200)
        self.table.pack(fill=tk.X, pady=6)

        # --- Figure and 3 axes ---
        self.fig, self.axs = plt.subplots(3, 1, figsize=(9, 7))
        plt.subplots_adjust(hspace=0.7)
        self.ax_orig, self.ax_classic, self.ax_gan = self.axs

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        frame_va = tk.Frame(self.root)
        frame_va.pack(side=tk.TOP, fill=tk.X, pady=10)
        self.voice_label = tk.Label(frame_va, text="🎤 Voice Assistant", font=("Helvetica", 12, "bold"))
        self.voice_label.pack()
        self.record_button = tk.Button(frame_va, text="🎙️", command=self.toggle_recording, font=("Helvetica", 18), width=3, height=1, bg="#4CAF50", fg="white")
        self.record_button.pack()
        self.log_text = scrolledtext.ScrolledText(frame_va, height=6, width=60, font=("Courier", 10))
        self.log_text.pack()

        self.is_running = False
        self.recording = False
        self.recording_stream = None

        self.curr_pos = self.window

    def toggle_anomaly(self):
        self.inject_anomaly = True

    def start_stream(self):
        self.is_running = True
        self.signal_anom = self.signal_orig.copy()
        self.curr_pos = self.window
        self.run_stream()

    def run_stream(self):
        if not self.is_running:
            return

        self.curr_pos += 1
        if self.curr_pos > len(self.signal_orig):
            self.curr_pos = self.window

        anomaly_type = "Normal"
        anomaly_desc = "Normal"
        recommendation = "No action required."
        if self.inject_anomaly:
            x = self.signal_orig[self.curr_pos - self.window:self.curr_pos].copy()
            t_win = self.t[self.curr_pos - self.window:self.curr_pos]
            x_anom, anomaly_type, anomaly_desc = inject_td_anomaly(x, t_win)
            self.signal_anom[self.curr_pos - self.window:self.curr_pos] = x_anom
            recommendation = self.get_llm_recommendation(anomaly_type, anomaly_desc)
            self.anomaly_records.append({
                "strength": float(np.std(x_anom)),
                "anomaly_type": anomaly_type,
                "desc": anomaly_desc
            })
            time_str = datetime.datetime.now().strftime('%H:%M:%S')
            self.table.insert('', 'end', values=(time_str, anomaly_type, anomaly_desc, recommendation))
            self.inject_anomaly = False

        x_win = self.signal_anom[self.curr_pos - self.window:self.curr_pos]
        t_win = self.t[self.curr_pos - self.window:self.curr_pos]
        if self.anomaly_records:
            anomaly_type = self.anomaly_records[-1]['anomaly_type']
            anomaly_desc = self.anomaly_records[-1]['desc']
        x_classic = simple_reconstruct(x_win, anomaly_type, anomaly_desc)

        # --- GAN output FIXED scaling ---
        x_norm = (x_win - np.mean(x_win)) / (np.std(x_win) + 1e-8)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            x_recon = self.generator(x_tensor).squeeze().numpy()
        # Rescale GAN output to match original window's mean and std
        gan_mean = np.mean(x_recon)
        gan_std = np.std(x_recon)
        if gan_std > 1e-8:
            x_gan = ((x_recon - gan_mean) / gan_std) * np.std(x_win) + np.mean(x_win)
        else:
            x_gan = np.ones_like(x_recon) * np.mean(x_win)

        # --- Redraw all three plots with cla() ---
        self.ax_orig.cla()
        self.ax_orig.plot(t_win, x_win, color='blue', label='Anomalous Signal')
        self.ax_orig.set_title("Original Signal with Anomaly")
        self.ax_orig.set_xlim(t_win[0], t_win[-1])
        self.ax_orig.set_ylim(-2, 2)
        if anomaly_type != "Normal":
            if "at" in anomaly_desc:
                idx = int(anomaly_desc.split("at")[1].split()[0])
                idx_rel = idx - (self.curr_pos-self.window)
                if 0 <= idx_rel < self.window:
                    self.ax_orig.axvline(t_win[idx_rel], color='red', linestyle='--')
            elif ":" in anomaly_desc:
                s = anomaly_desc.split()[-1]
                start, end = map(int, s.split(":"))
                s_rel = max(start - (self.curr_pos-self.window), 0)
                e_rel = min(end - (self.curr_pos-self.window), self.window-1)
                if e_rel > s_rel:
                    self.ax_orig.axvspan(t_win[s_rel], t_win[e_rel], color='red', alpha=0.3)
        self.ax_orig.legend()

        self.ax_classic.cla()
        self.ax_classic.plot(t_win, x_classic, color='orange', label='Classical Recon')
        self.ax_classic.set_title("Classical Reconstruction")
        self.ax_classic.set_xlim(t_win[0], t_win[-1])
        self.ax_classic.set_ylim(-2, 2)
        self.ax_classic.legend()

        self.ax_gan.cla()
        self.ax_gan.plot(t_win, x_gan, color='green', label='GAN Recon')
        self.ax_gan.set_title("GAN Reconstruction")
        self.ax_gan.set_xlim(t_win[0], t_win[-1])
        self.ax_gan.set_ylim(-2, 2)
        self.ax_gan.legend()

        self.canvas.draw()
        self.root.after(30, self.run_stream)

    def get_llm_recommendation(self, anomaly_type, desc):
        prompt = (
            f"In a power transmission/distribution system, what is the best method to fix or reconstruct a signal affected by: '{anomaly_type}'?\n"
            f"Details: {desc}\n"
            "Give a brief technical recommendation for restoration."
        )
        try:
            response = openai.ChatCompletion.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response["choices"][0]["message"]["content"].strip()
            return answer
        except Exception as e:
            return f"LLM error: {e}"

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
            self.record_button.config(bg="#ff4d4d", text="⏹️")
        else:
            self.stop_recording_and_process()
            self.record_button.config(bg="#4CAF50", text="🎙️")

    def start_recording(self):
        self.recording = True
        audio_queue.queue.clear()
        def callback(indata, frames, time, status):
            if status:
                print(status)
            if self.recording:
                audio_queue.put(indata.copy())
        self.recording_stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=callback)
        self.recording_stream.start()
        self.log_text.insert(tk.END, "🎙️ Recording... Click again to stop.\n")

    def stop_recording_and_process(self):
        self.recording = False
        if self.recording_stream:
            self.recording_stream.stop()
            self.recording_stream.close()
        frames = []
        while not audio_queue.empty():
            frames.append(audio_queue.get())
        if not frames:
            self.log_text.insert(tk.END, "\n⚠️ No audio captured.\n")
            return
        audio_data = np.concatenate(frames, axis=0)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, audio_data, sample_rate)
        self.process_voice_input(temp_file.name)

    def process_voice_input(self, audio_path):
        self.log_text.delete("1.0", tk.END)
        try:
            user_text = transcribe_audio(audio_path)
            self.log_text.insert(tk.END, f"\n📝 Transcribed: {user_text}\n")
            response = voice_llm_response(user_text)
            self.log_text.insert(tk.END, f"\n💡 Recommendation:\n{response}\n")
        except Exception as err:
            self.log_text.insert(tk.END, f"\n❌ Error: {err}\n")
        finally:
            import os
            try: os.remove(audio_path)
            except Exception: pass

    def generate_diagnosis(self):
        if not self.anomaly_records:
            self.log_text.insert(tk.END, "\nNo anomalies injected yet.\n")
            return

        strengths = [r['strength'] for r in self.anomaly_records]
        anomaly_types = [r['anomaly_type'] for r in self.anomaly_records]
        descs = [r['desc'] for r in self.anomaly_records]
        record_count = len(strengths)
        avg_strength = np.mean(strengths)
        max_strength = np.max(strengths)
        min_strength = np.min(strengths)

        summary = (
            f"Diagnosis complete.\n"
            f"Detected {record_count} anomalies.\n"
            f"Average signal std (strength): {avg_strength:.3f}, "
            f"range: {min_strength:.3f} to {max_strength:.3f}.\n"
            f"Anomaly types: {', '.join(anomaly_types)}\n"
        )

        prompt = f"""
Detected anomalies:
- Count: {record_count}
- Average signal std: {avg_strength:.3f}
- Min signal std: {min_strength:.3f}
- Max signal std: {max_strength:.3f}
- Anomaly types: {', '.join(anomaly_types)}

Based on the signal strength and anomaly types, determine:
1. The spoofing/severity reason (1 sentence)
2. The best GAN-based and classical signal processing resolution technique (1-2 sentences)
"""
        try:
            response = openai.ChatCompletion.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": prompt}]
            )
            explanation = response["choices"][0]["message"]["content"].strip()
            diagnosis = summary + "\n" + explanation
        except Exception as e:
            diagnosis = summary + f"\nLLM error: {e}"

        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, diagnosis)

if __name__ == '__main__':
    generator = Generator()
    generator.load_state_dict(torch.load(BASE_DIR / "signal2signal_cgan.pth", map_location="cpu"))
    generator.eval()
    root = tk.Tk()
    root.title("T&D Anomaly Dashboard: Classical vs GAN + Voice Assistant + Diagnosis")
    root.geometry("1200x1000")
    app = DashboardApp(root, generator)
    root.mainloop()
