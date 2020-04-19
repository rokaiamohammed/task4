from PyQt5 import QtWidgets
from task4 import Ui_MainWindow
import sys
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import scipy.io.wavfile as wav
from scipy.io import wavfile
import os
import wave
import pylab
import hashlib
import numpy as np
from numpy.lib import stride_tricks
import pandas as pd
from glob import glob
from scipy.io import wavfile
from PIL import Image
import imagehash

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.Browse1.clicked.connect(lambda: self.browse(1))
        self.ui.Browse2.clicked.connect(lambda: self.browse(2))
        self.signal=[]
        self.hash=[]
        self.ui.mixer.setValue(0)
        self.ui.mixer.setTickPosition(QtWidgets.QSlider.TicksBelow)
        # self.ui.mixer.valueChanged.connect(self.valuechange)

        self.data_dir='./database'
        self.SG=glob(self.data_dir+'/*.wav')
        print(len(self.SG)) #Number of files in our database

        self.arrayofHash=[]
        self.ui.mix.clicked.connect(self.valuechange)
        
        for i in range(len(self.SG)):
            FS, data = wavfile.read(self.SG[i])  # read wav file
            self.ls=(plt.specgram(data[:,0], Fs=FS, NFFT=128, noverlap=0))  # The spectogram
            #plt.show() #if you want to show the spectogram
            # ======= This part may help in getting the first minute ==========
            
            self.ls=self.ls[:60]

            ax = plt.axes()
            ax.set_axis_off()
            plt.savefig('sp_xyz' + str(i)+'.png', bbox_inches='tight',  transparent=True,pad_inches=0, frameon='false')
            file='sp_xyz'+str(i)+'.png'
            img = Image.open(file)
            hashedVersion = imagehash.phash(img)
            self.arrayofHash.append(str(hashedVersion))

            # print(self.ls)


        print(self.arrayofHash)


    def browse(self,n):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Please Choose a .wav file", "","Wav Files (*.wav)", options=options)
        if fileName:
            FS, data = wavfile.read(fileName)  # read wav file
            # plt.specgram(data[:,0], Fs=FS, NFFT=128, noverlap=0)  # plot
            # plt.show()
            if n==1:
                self.file1=plt.specgram(data[:,0], Fs=FS, NFFT=128, noverlap=0)
            elif n==2:
                self.file2=plt.specgram(data[:,0], Fs=FS, NFFT=128, noverlap=0)

    def valuechange(self):
       self.MixerValue=self.ui.mixer.value()/100
    #    print(self.file1)
    #    print(self.file2)
    #    self.file1=self.file1*self.MixerValue
    #    self.file2=self.file2*(1-self.MixerValue)
    #    print(self.file1)
    #    print(self.file2)


    # def browse(self,n):
    #     options = QtWidgets.QFileDialog.Options()
    #     options |= QtWidgets.QFileDialog.DontUseNativeDialog
    #     filePath = QtWidgets.QFileDialog.getOpenFileNames(self,"Please Choose a .wav file", "","Wav Files (*.wav)", options=options)
    #     temp=filePath
    #     for vs in temp[0]:
    #         self.ext = os.path.splitext(vs)[-1].lower()
    #         pinky=filePath[0]
    #         self.samplerate, self.signal=wavfile.read(pinky[0])
    #         self.secs=np.arange(self.signal.shape[0])/float(self.samplerate)
    #     if n==1:
    #         self.file1=self.signal
    #     elif n==2:
    #         self.file2=self.signal

        # fig,ax=plt.subplots()
        # ax.plot(self.secs,self.file1)
        # ax.set(xlabel='Time', ylabel='Sound Amplitude')
        # plt.show


        # f, t, Sxx = signal.spectrogram(self.file1[1], self.samplerate)
        # plt.pcolormesh(t, f, Sxx)
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()






        # Plot the spectrogram
        # plot.subplot(212)
        # powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(self.signal[1], Fs=self.samplerate)
        # plot.xlabel('Time')
        # plot.ylabel('Frequency')
        # plot.show()






        # self.total_ts_sec = self.x
        # print("The total time series length = {} sec (N points = {}) ".format(self.total_ts_sec, len(self.signal)))
        # plt.figure(figsize=(20,3))
        # plt.plot(self.signal)
        # plt.xticks(np.arange(0,len(self.signal),self.samplerate),
        #            np.arange(0,len(self.signal)/self.samplerate,1))
        # plt.ylabel("Amplitude")
        # plt.xlabel("Time (second)")
        # # plt.title("The total length of time series = {} sec, sample_rate = {}".format(len(ts)/sample_rate, sample_rate))
        # plt.show()


        # def get_xn(Xs,n):
        #     '''
        #     calculate the Fourier coefficient X_n of
        #     Discrete Fourier Transform (DFT)
        #     '''
        #     L  = len(Xs)
        #     ks = np.arange(0,L,1)
        #     xn = np.sum(Xs*np.exp((1j*2*np.pi*ks*n)/L))/L
        #     return(xn)


        # def get_xns(ts):

        #     '''
        #     Compute Fourier coefficients only up to the Nyquest Limit Xn, n=1,...,L/2
        #     and multiply the absolute value of the Fourier coefficients by 2,
        #     to account for the symetry of the Fourier coefficients above the Nyquest Limit.
        #     '''
        #     mag = []
        #     L = len(self.signal)
        #     for n in range(int(L/2)): # Nyquest Limit
        #         mag.append(np.abs(get_xn(self.signal,n))*2)
        #     return(mag)
        # mag = get_xns(self.signal)

        # # the number of points to label along xaxis
        # Nxlim = 10

        # plt.figure(figsize=(20,3))
        # plt.plot(mag)
        # plt.xlabel("Frequency (k)")
        # plt.title("Two-sided frequency plot")
        # plt.ylabel("|Fourier Coefficient|")
        # plt.show()




    #     self.plotstft("Adele_Someone_Like_You_Official_Music_Video_accompaniment.wav")




    # """ short time fourier transform of audio signal """
    # def stft(self,sig, frameSize, overlapFac=0.5, window=np.hanning):
    #     win = window(frameSize)
    #     hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    #     # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    #     samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)
    #     cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    #     # zeros at end (thus samples can be fully covered by frames)
    #     samples = np.append(samples, np.zeros(frameSize))

    #     frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    #     frames *= win

    #     return np.fft.rfft(frames)

    # """ scale frequency axis logarithmically """
    # def logscale_spec(self,spec, sr=44100, factor=20.):
    #     timebins, freqbins = np.shape(spec)

    #     scale = np.linspace(0, 1, freqbins) ** factor
    #     scale *= (freqbins-1)/max(scale)
    #     scale = np.unique(np.round(scale))

    #     # create spectrogram with new freq bins
    #     newspec = np.complex128(np.zeros([timebins, len(scale)]))
    #     for i in range(0, len(scale)):
    #         if i == len(scale)-1:
    #             newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
    #         else:
    #             newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)

    #     # list center freq of bins
    #     allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    #     freqs = []
    #     for i in range(0, len(scale)):
    #         if i == len(scale)-1:
    #             freqs += [np.mean(allfreqs[scale[i]:])]
    #         else:
    #             freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    #     return newspec, freqs

    # """ plot spectrogram"""
    # def plotstft(self,audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    #     samplerate, samples = wav.read(audiopath)
    #     s = self.stft(samples, binsize)

    #     sshow, freq = self.logscale_spec(s, factor=1.0, sr=samplerate)
    #     ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    #     timebins, freqbins = np.shape(ims)

    #     plt.figure(figsize=(15, 7.5))
    #     plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    #     plt.colorbar()

    #     plt.xlabel("time (s)")
    #     plt.ylabel("frequency (hz)")
    #     plt.xlim([0, timebins-1])
    #     plt.ylim([0, freqbins])

    #     xlocs = np.float32(np.linspace(0, timebins-1, 5))
    #     plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    #     ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    #     plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    #     if plotpath:
    #         plt.savefig(plotpath, bbox_inches="tight")
    #     else:
    #         plt.show()

    #     plt.clf()


    # # def graph_spectrogram(self,wav_file):
    # #     sound_info, frame_rate = get_wav_info(wav_file)
    # #     plt.figure(num=None, figsize=(19, 12))
    # #     plt.subplot(111)
    # #     plt.title('spectrogram of %r' % wav_file)
    # #     plt.specgram(sound_info, Fs=frame_rate)
    # #     plt.savefig('spectrogram.png')
    # def get_wav_info(self,wav_file):
    #     wav = wave.open(wav_file, 'r')
    #     frames = wav.readframes(-1)
    #     sound_info = pylab.fromstring(frames, 'int16')
    #     frame_rate = wav.getframerate()
    #     wav.close()
    #     return sound_info, frame_rate

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()