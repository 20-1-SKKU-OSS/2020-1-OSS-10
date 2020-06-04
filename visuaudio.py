"""python application to view audio equalizer*
of input sound source"""
import struct
import sys
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyaudio
from scipy.fftpack import fft
from pynput.keyboard import Listener, Key



class AudioStream:
    """stream audio from input source (mic) and continuously
    plot (bar) based on audio spectrum from waveform data"""

    def __init__(self, num,symbol):
        self.traces = set()
        self.number = num
        self.symbol = symbol
        # pyaudio setup
        self.FORMAT = pyaudio.paInt16  # bytes / sample
        self.CHANNELS = 1  # mono sound
        self.RATE = 44100  # samples / sec (44.1 kHz)
        self.CHUNK = 1024  # how much audio processed / frame -- set smaller for higher frame rate

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )
        # spectrum x points
        self.f = np.linspace(1, int(self.RATE / 2), 64)

        # pyqtgraph setup
        pg.setConfigOptions(antialias=True)
        self.app = QtGui.QApplication(sys.argv)
        self.win = pg.GraphicsWindow(title="Audio Spectrum")
        self.win.setGeometry(10, 52, 480 * 2, 200 * 2)
        # window content setup
        self.audio_plot = self.win.addPlot(row=1, col=1)
        self.audio_plot.setYRange(0.00, 0.25)
        self.audio_plot.setXRange(2000, int(self.RATE / 2))
        self.audio_plot.showGrid(x=True, y=True)
        self.audio_plot.hideAxis("bottom")
        self.audio_plot.hideAxis("left")

        self.a = {
            'color': 'w',
            'color_x':100,
            'color_y':100,
            'color_z':250,
        }

        # graph init
        self.graph = None

    @staticmethod
    def set_gradient_brush():
        """set color gradient, return QtGui.QBrush obj"""
        grad = QtGui.QLinearGradient(0, 0, 0, 1)
        grad.setColorAt(0.1, pg.mkColor("#FF0000"))
        grad.setColorAt(0.24, pg.mkColor("#FF7F00"))
        grad.setColorAt(0.38, pg.mkColor("#FFFF00"))
        grad.setColorAt(0.52, pg.mkColor("#00FF00"))
        grad.setColorAt(0.66, pg.mkColor("#0000FF"))
        grad.setColorAt(0.80, pg.mkColor("#4B0082"))
        grad.setColorAt(0.94, pg.mkColor("#9400D3"))
        grad.setCoordinateMode(QtGui.QGradient.ObjectMode)
        brush = QtGui.QBrush(grad)

        return brush

    # Bar Graph
    def set_plotdata_1(self, name, data_x, data_y):
        """set plot with init and new data -- reference
        self.traces to verify init or recurring data input"""
        if name in self.traces:
            # update bar plot content
            self.graph.setOpts(x=data_x, height=data_y, width=350)
        else:
            self.traces.add(name)
            # initial setup of bar plot
            brush = self.set_gradient_brush()
            self.graph = pg.BarGraphItem(
                x=data_x, height=data_y, width=50, brush=brush, pen=(0, 0, 0)
            )
        self.audio_plot.addItem(self.graph)

    # Scatter Plot Graph
    def set_plotdata_2(self, name, data_x, data_y):
        data_y = data_y[:64]
        if name in self.traces:
            # update scatter plot content
            self.graph.clear()
            self.graph = pg.ScatterPlotItem(
                x=data_x, y=data_y, pen=None, symbol=self.symbol, size=30, brush=(self.a.color_x, self.color_y, self.color_z, 100),
            )
        else:
            self.traces.add(name)
            # initial setup of scatter plot
            self.graph = pg.ScatterPlotItem(
                x=data_x, y=data_y, pen=None, symbol=self.symbol, size=30, brush=(100, 100, 255, 100),
            )
        self.audio_plot.addItem(self.graph)

    # Curve Graph
    def set_plotdata_3(self, name, data_x, data_y):
        data_y = data_y[:64]
        if name in self.traces:
            # update curve plot content
            self.graph.clear()
            pen1 = pg.mkPen(color=self.a['color'], width=0)
            self.graph = pg.PlotCurveItem(
                x=data_x, y=data_y, pen=pen1,
            )
        else:
            self.traces.add(name)
            # initial setup of curve plot
            self.graph = pg.PlotCurveItem(
                x=data_x, y=data_y, pen='r',
            )
        self.audio_plot.addItem(self.graph)
    # Line Graph    
    def set_plotdata_4(self, name, data_x, data_y):
        data_y = data_y[:64]
        if name in self.traces:
            # update curve plot content
            self.graph.clear()
            pen1=pg.mkPen(color=self.a['color'], width=15, style=QtCore.Qt.DashLine)
            self.graph.setData(data_x, data_y,pen=pen1,shadowPen='#19070B')
            """self.graph = pg.PlotCurveItem(
                x=data_x, y=data_y, pen=self.a['pen'], shadowPen=self.a['shadowPen'],
            )"""
        else:
            pen1 = pg.mkPen(color=(250, 0, 0), width=15, style=QtCore.Qt.DashLine)
            self.traces.add(name)
            # initial setup of curve plot
            self.graph = pg.PlotCurveItem(
                x=data_x, y=data_y, pen=pen1, shadowPen='r',
            )
        self.audio_plot.addItem(self.graph)
        
    def update(self):
        """update plot by number which user chose"""
        if self.number == 1:  # Bar Graph
            self.set_plotdata_1(name="spectrum", data_x=self.f, data_y=self.calculate_data())
        elif self.number == 2:  # Scatter Graph
            self.set_plotdata_2(name="spectrum", data_x=self.f, data_y=self.calculate_data())
        elif self.number == 3:  # Curve Graph
            self.set_plotdata_3(name="spectrum", data_x=self.f, data_y=self.calculate_data())
        elif self.number == 4:  # Line Graph
            self.set_plotdata_4(name="spectrum", data_x=self.f, data_y=self.calculate_data())
            
    def calculate_data(self):
        """get sound data and manipulate for plotting using fft"""
        # get and unpack waveform data
        wf_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
        wf_data = struct.unpack(
            str(2 * self.CHUNK) + "B", wf_data
        )  # 2 * self.CHUNK :: wf_data 2x len of CHUNK -- wf_data range(0, 255)
        # generate spectrum data for plotting using fft (fast fourier transform)
        sp_data = fft(
            np.array(wf_data, dtype="int8") - 128
        )  # - 128 :: any int less than 127 will wrap around to 256 down
        # np.abs (below) converts complex num in fft to real magnitude
        sp_data = (
                np.abs(sp_data[0:int(self.CHUNK)])  # slice: slice first half of our fft
                * 2
                / (256 * self.CHUNK)
        )  # rescale: mult 2, div amp waveform and no. freq in your spectrum
        sp_data[sp_data <= 0.001] = 0
        return sp_data

    @staticmethod
    def start():
        """start application"""
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QtGui.QApplication.instance().exec_()

    @staticmethod
    def changeColor(key):
        """change color in curve graph after start"""
        """
            black : #19070B     red : #FF0000       orange : #FF8000
            yellow : #FFFF00    green : #40FF00     skyblue : #00FFFF
            blue : #0040FF      purple: #BF00FF     white : #FFFFFF
        """
        global colorIndex
        colorList = ['#19070B', '#FF0000', '#FF8000', '#FFFF00', '#40FF00', '#00FFFF', '#0040FF', '#BF00FF', ' #FFFFFF']
        if(colorIndex==8):
            colorIndex=0
        try:
            if (key == Key.enter):
                AUDIO_APP.a['color'] = colorList[colorIndex]
                colorIndex+=1
            elif(key==Key.enter):
                sys.exit(1)
        except:
            pass

    def animation(self):
        """call self.start and self.update for continuous
        output application"""
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        # self.update()
        self.start()
        
def InttoSymbol(symbol):
    if symbol==1:
        sym = 'd'
    elif symbol==2:
        sym = 'o'
    elif symbol==3:
        sym = 'x'
    elif symbol==4:
        sym = 't'
    elif symbol==5:
        sym = 's'
    return sym
global colorIndex
colorIndex=0

if __name__ == "__main__":
    print("Choose and type number.")
    print("-" * 20)
    print("1: Bar Graph")
    print("2: Scatter Graph")
    print("3: Curve Graph")
    print("4: Line Graph")
    print("5: quit")
    print("-" * 20)
    number = int(input("Graph Type:"))
    while number<1 or number>5:
        number = int(input("Out of range! try again:"))
    if number==5:
        exit()
    symbol ='o'
    if number==2:
        print("Choose symbol")
        print("-" * 20)
        print("1: Diamond")
        print("2: Circular")
        print("3: Cross")
        print("4: Triangular")
        print("5: Square")
        print("-" * 20)
        symbol = int(input("Symbol:"))
        while symbol<1 or symbol>5:
            symbol = int(input("Out of range! try again:"))
        symbol = InttoSymbol(symbol)
    AUDIO_APP = AudioStream(number,symbol)
    with Listener(on_press=AUDIO_APP.changeColor) as listener:
        AUDIO_APP.animation()
    listener.join()
