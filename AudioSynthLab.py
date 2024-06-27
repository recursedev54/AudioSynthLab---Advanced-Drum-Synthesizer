import sys
import numpy as np
import pyaudio
from scipy import signal
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QGroupBox, QGridLayout
from PyQt5.QtCore import Qt

SAMPLE_RATE = 44100
BUFFER_SIZE = 1024

class AudioThread(QtCore.QThread):
    def __init__(self):
        QtCore.QThread.__init__(self)
        self.stream = None
        self.p = pyaudio.PyAudio()
        self.buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.sound_queue = []
        self.current_sound = None
        self.current_index = 0

    def run(self):
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=SAMPLE_RATE,
                                  output=True,
                                  frames_per_buffer=BUFFER_SIZE,
                                  stream_callback=self.callback)
        self.stream.start_stream()
        while self.stream.is_active():
            self.msleep(100)  # Sleep for 100 milliseconds

    def callback(self, in_data, frame_count, time_info, status):
        if self.current_sound is None and self.sound_queue:
            self.current_sound = self.sound_queue.pop(0)
            self.current_index = 0

        if self.current_sound is not None:
            remaining = len(self.current_sound) - self.current_index
            if remaining >= frame_count:
                output = self.current_sound[self.current_index:self.current_index + frame_count]
                self.current_index += frame_count
            else:
                output = np.zeros(frame_count, dtype=np.float32)
                output[:remaining] = self.current_sound[self.current_index:]
                self.current_sound = None
                self.current_index = 0
        else:
            output = np.zeros(frame_count, dtype=np.float32)

        return (output.tobytes(), pyaudio.paContinue)

    def add_sound(self, sound):
        self.sound_queue.append(sound)

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

class DrumSynthesizer(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.audio_thread = AudioThread()
        self.audio_thread.start()
        self.distortion = 0
        self.compressor_threshold = 0.5
        self.compressor_ratio = 4
        self.input_gain = 1.0
        self.output_gain = 1.0

    def apply_distortion(self, sound):
        return np.tanh(sound * (1 + self.distortion * 10)) / (1 + self.distortion * 10)

    def apply_compression(self, sound):
        # Apply input gain
        sound = sound * self.input_gain
        
        # Compress
        sound_compressed = np.zeros_like(sound)
        for i, sample in enumerate(sound):
            if abs(sample) > self.compressor_threshold:
                if sample > 0:
                    sound_compressed[i] = self.compressor_threshold + (sample - self.compressor_threshold) / self.compressor_ratio
                else:
                    sound_compressed[i] = -self.compressor_threshold + (sample + self.compressor_threshold) / self.compressor_ratio
            else:
                sound_compressed[i] = sample
        
        # Apply output gain
        return sound_compressed * self.output_gain

    def synthesize_kick(self, freq=55, decay=0.5, pitch_decay=0.05):
        t = np.linspace(0, 1, int(SAMPLE_RATE * decay), False)
        freq_env = freq * np.exp(-t * pitch_decay * 20)
        wave = np.sin(2 * np.pi * freq_env * t)
        envelope = np.exp(-t * 5)
        sound = (wave * envelope).astype(np.float32)
        return self.apply_compression(self.apply_distortion(sound))

    def synthesize_snare(self, tone_freq=200, noise_amount=0.5, tone_mix=0.5):
        t = np.linspace(0, 1, int(SAMPLE_RATE * 0.2), False)
        tone = np.sin(2 * np.pi * tone_freq * t)
        noise = np.random.uniform(-1, 1, len(t))
        wave = tone_mix * tone + (1 - tone_mix) * noise
        envelope = np.exp(-t * 20)
        sound = (wave * envelope * noise_amount).astype(np.float32)
        return self.apply_compression(self.apply_distortion(sound))

    def synthesize_hihat(self, freq=500, decay=2.0):
        t = np.linspace(0, decay, int(SAMPLE_RATE * decay), False)
        noise = np.random.uniform(-1, 1, len(t))
        # Apply bandpass filter
        sos = signal.butter(10, [freq * 0.8, freq * 1.2], btype='bandpass', fs=SAMPLE_RATE, output='sos')
        filtered_noise = signal.sosfilt(sos, noise)
        envelope = np.exp(-t * (5 / decay))
        sound = (filtered_noise * envelope).astype(np.float32)
        return self.apply_compression(self.apply_distortion(sound))

    def synthesize_crash(self, freq=1000, decay=1):
        t = np.linspace(0, decay, int(SAMPLE_RATE * decay), False)
        noise = np.random.uniform(-1, 1, len(t))
        
        # Create and apply notch filter
        b, a = signal.iirnotch(freq, Q=10, fs=SAMPLE_RATE)
        filtered_noise = signal.lfilter(b, a, noise)
        
        envelope = np.exp(-t * 2)
        sound = (filtered_noise * envelope).astype(np.float32)
        return self.apply_compression(self.apply_distortion(sound))

    def play_sound(self, sound):
        self.audio_thread.add_sound(sound)

    def set_distortion(self, value):
        self.distortion = value

    def set_compressor_threshold(self, value):
        self.compressor_threshold = value

    def set_compressor_ratio(self, value):
        self.compressor_ratio = value

    def set_input_gain(self, value):
        self.input_gain = value

    def set_output_gain(self, value):
        self.output_gain = value

class AudioSynthLab(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AudioSynthLab - Advanced Drum Synthesizer')
        self.setGeometry(100, 100, 800, 600)
        self.synth = DrumSynthesizer()
        self.controls = {}
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        title = QLabel('AudioSynthLab - Advanced Drum Synthesizer', self)
        title.setStyleSheet("font-size: 24px; font-weight: bold; text-align: center; color: #333;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        controls_layout = QGridLayout()

        kick_group = self.createControlGroup('Kick Drum', [
            ('Frequency', 20, 150, 55, 'Hz'),
            ('Decay', 0.1, 2, 0.5, 's'),
            ('Pitch Decay', 0.01, 0.5, 0.05, 's')
        ])
        controls_layout.addWidget(kick_group, 0, 0)

        snare_group = self.createControlGroup('Snare Drum', [
            ('Tone Frequency', 100, 500, 200, 'Hz'),
            ('Noise Amount', 0, 1, 0.5, ''),
            ('Tone/Noise Mix', 0, 1, 0.5, '')
        ])
        controls_layout.addWidget(snare_group, 0, 1)

        hihat_group = self.createControlGroup('Hi-Hat', [
            ('Frequency', 100, 12000, 500, 'Hz'),  # Extended frequency range
            ('Decay', 0.01, 2, 0.05, 's')
        ])
        controls_layout.addWidget(hihat_group, 1, 0)

        crash_group = self.createControlGroup('Crash Cymbal', [
            ('Notch Frequency', 500, 5000, 1000, 'Hz'),
            ('Decay', 0.5, 3, 1, 's')
        ])
        controls_layout.addWidget(crash_group, 1, 1)

        # Global Distortion Control
        distortion_group = QGroupBox("Global Distortion")
        distortion_layout = QVBoxLayout()
        distortion_label = QLabel("Distortion: 0.00")
        distortion_slider = QSlider(Qt.Horizontal)
        distortion_slider.setMinimum(0)
        distortion_slider.setMaximum(100)
        distortion_slider.setValue(0)
        distortion_slider.setSingleStep(1)
        distortion_slider.valueChanged.connect(lambda value: self.updateDistortion(value, distortion_label))
        distortion_layout.addWidget(distortion_label)
        distortion_layout.addWidget(distortion_slider)
        distortion_group.setLayout(distortion_layout)
        controls_layout.addWidget(distortion_group, 2, 0, 1, 2)

        # Global Compressor Control
        compressor_group = QGroupBox("Global Compressor")
        compressor_layout = QGridLayout()
        
        threshold_label = QLabel("Threshold: 0.50")
        threshold_slider = QSlider(Qt.Horizontal)
        threshold_slider.setMinimum(1)
        threshold_slider.setMaximum(100)
        threshold_slider.setValue(50)
        threshold_slider.setSingleStep(1)
        threshold_slider.valueChanged.connect(lambda value: self.updateCompressorThreshold(value, threshold_label))
        
        ratio_label = QLabel("Ratio: 4.00")
        ratio_slider = QSlider(Qt.Horizontal)
        ratio_slider.setMinimum(100)
        ratio_slider.setMaximum(2000)
        ratio_slider.setValue(400)
        ratio_slider.setSingleStep(10)
        ratio_slider.valueChanged.connect(lambda value: self.updateCompressorRatio(value, ratio_label))
        
        input_gain_label = QLabel("Input Gain: 0.00 dB")
        input_gain_slider = QSlider(Qt.Horizontal)
        input_gain_slider.setMinimum(-2000)
        input_gain_slider.setMaximum(2000)
        input_gain_slider.setValue(0)
        input_gain_slider.setSingleStep(10)
        input_gain_slider.valueChanged.connect(lambda value: self.updateInputGain(value, input_gain_label))
        
        output_gain_label = QLabel("Output Gain: 0.00 dB")
        output_gain_slider = QSlider(Qt.Horizontal)
        output_gain_slider.setMinimum(-2000)
        output_gain_slider.setMaximum(2000)
        output_gain_slider.setValue(0)
        output_gain_slider.setSingleStep(10)
        output_gain_slider.valueChanged.connect(lambda value: self.updateOutputGain(value, output_gain_label))

        compressor_layout.addWidget(threshold_label, 0, 0)
        compressor_layout.addWidget(threshold_slider, 0, 1)
        compressor_layout.addWidget(ratio_label, 1, 0)
        compressor_layout.addWidget(ratio_slider, 1, 1)
        compressor_layout.addWidget(input_gain_label, 2, 0)
        compressor_layout.addWidget(input_gain_slider, 2, 1)
        compressor_layout.addWidget(output_gain_label, 3, 0)
        compressor_layout.addWidget(output_gain_slider, 3, 1)
        
        compressor_group.setLayout(compressor_layout)
        controls_layout.addWidget(compressor_group, 3, 0, 1, 2)

        layout.addLayout(controls_layout)
        self.setLayout(layout)

    def createControlGroup(self, title, controls):
        group_box = QGroupBox(title)
        group_layout = QVBoxLayout()

        self.controls[title] = {}

        for control in controls:
            label = QLabel(f'{control[0]}: {control[3]} {control[4]}')
            slider = QSlider(Qt.Horizontal)
            slider_min = int(control[1] * 100)
            slider_max = int(control[2] * 100)
            slider.setMinimum(slider_min)
            slider.setMaximum(slider_max)
            slider.setValue(int(control[3] * 100))
            slider.setSingleStep(1)
            slider.valueChanged.connect(lambda value, l=label, c=control: self.updateLabel(l, value / 100, c[4], c[0]))

            self.controls[title][control[0]] = {'label': label, 'slider': slider}

            group_layout.addWidget(label)
            group_layout.addWidget(slider)

        play_button = QPushButton(f'Play {title}')
        play_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        play_button.clicked.connect(lambda: self.playSound(title))

        group_layout.addWidget(play_button)
        group_box.setLayout(group_layout)
        return group_box

    def updateLabel(self, label, value, unit, param):
        label.setText(f'{param}: {value:.2f} {unit}')

    def updateDistortion(self, value, label):
        distortion = value / 100
        label.setText(f"Distortion: {distortion:.2f}")
        self.synth.set_distortion(distortion)

    def updateCompressorThreshold(self, value, label):
        threshold = value / 100
        label.setText(f"Threshold: {threshold:.2f}")
        self.synth.set_compressor_threshold(threshold)

    def updateCompressorRatio(self, value, label):
        ratio = value / 100
        label.setText(f"Ratio: {ratio:.2f}")
        self.synth.set_compressor_ratio(ratio)

    def updateInputGain(self, value, label):
        gain_db = value / 100
        gain = 10 ** (gain_db / 20)
        label.setText(f"Input Gain: {gain_db:.2f} dB")
        self.synth.set_input_gain(gain)

    def updateOutputGain(self, value, label):
        gain_db = value / 100
        gain = 10 ** (gain_db / 20)
        label.setText(f"Output Gain: {gain_db:.2f} dB")
        self.synth.set_output_gain(gain)

    def playSound(self, instrument):
        if instrument == 'Kick Drum':
            freq = self.controls[instrument]['Frequency']['slider'].value() / 100
            decay = self.controls[instrument]['Decay']['slider'].value() / 100
            pitch_decay = self.controls[instrument]['Pitch Decay']['slider'].value() / 100
            sound = self.synth.synthesize_kick(freq, decay, pitch_decay)
        elif instrument == 'Snare Drum':
            tone_freq = self.controls[instrument]['Tone Frequency']['slider'].value() / 100
            noise_amount = self.controls[instrument]['Noise Amount']['slider'].value() / 100
            tone_mix = self.controls[instrument]['Tone/Noise Mix']['slider'].value() / 100
            sound = self.synth.synthesize_snare(tone_freq, noise_amount, tone_mix)
        elif instrument == 'Hi-Hat':
            freq = self.controls[instrument]['Frequency']['slider'].value() / 100
            decay = self.controls[instrument]['Decay']['slider'].value() / 100
            sound = self.synth.synthesize_hihat(freq, decay)
        elif instrument == 'Crash Cymbal':
            freq = self.controls[instrument]['Notch Frequency']['slider'].value() / 100
            decay = self.controls[instrument]['Decay']['slider'].value() / 100
            sound = self.synth.synthesize_crash(freq, decay)
        
        self.synth.play_sound(sound)

    def closeEvent(self, event):
        self.synth.audio_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication([])
    window = AudioSynthLab()
    window.show()
    sys.exit(app.exec_())
