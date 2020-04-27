from PyQt5 import QtWidgets,QtGui
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
from os.path import relpath
import difflib
from scipy.io.wavfile import write
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.Browse1.clicked.connect(lambda: self.browse(1))
        self.ui.Browse2.clicked.connect(lambda: self.browse(2))
        self.signal=[] 
        self.data= []
        self.songs=[]
        self.similarity=[]
        self.ui.mixer.setValue(0)
        self.ui.mixer.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.ui.mix.clicked.connect(self.valuechange)
        # self.ui.table1.setItem(1, 1,QtWidgets.QTableWidgetItem("self.similarity[i]"))
        self.data_dir='./database/'
        self.SG=glob(self.data_dir+'*.wav')
        for i in range(len(self.SG)):
            self.songs.append(relpath(self.SG[i], './database\\'))

        # print(self.SG) #Number of files in our database
        self.arrayofHash=[]
        for i in range(len(self.SG)):
            FS, data = wavfile.read(self.SG[i])  # read wav file
            data=data[0:60*FS]
            self.ls=(plt.specgram(data[:,0], Fs=FS, NFFT=128, noverlap=0))  # The spectogram
            #plt.show() #if you want to show the spectogram
            
            # ======= This part may help in getting the first minute ==========          
            self.ls=self.ls[:60]

            ax = plt.axes()
            ax.set_axis_off()
            plt.savefig('./SG_DataBase/sp_xyz' + str(i)+'.png', bbox_inches='tight',  transparent=True,pad_inches=0, frameon='false')
           

        #     # print(self.ls)


      

        self.data_dir_SG='./SG_DataBase'
        self.SG_database=glob(self.data_dir_SG+'/*.png')
        for i in range(len(self.SG_database)): 
            file=self.SG_database[i]
            img = Image.open(file)
            hashedVersion = imagehash.phash(img)
            self.arrayofHash.append(hashedVersion)
        # print(self.arrayofHash)

    def browse(self,n):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Please Choose a .wav file", "","Wav Files (*.wav)", options=options)
        if fileName:
           
            # data=wave.open(fileName,'rb')
            # print(fileName)
            FS, data = wavfile.read(fileName)  # read wav file
            data=data[0:60*FS]
            # plt.specgram(data[:,0], Fs=FS, NFFT=128, noverlap=0)  # plot
            # plt.show() 
            
            if n==1:
                self.data1=data
                # self.data.append( [data.getparams(),data.readframes(data.getnframes())] )
                self.fs1=FS
                
                # song=relpath(fileName, 'C:/Users/Lenovo/Desktop/task4/database')
                # self.index1 = self.songs.index(song) 
                # arr=np.array(self.data1,dtype=np.float64)
                # self.data1=arr.astype(np.int16)
                # print (data)
                # print(len(self.data))
            
            elif n==2:
                self.data2=data
                # self.data.append( [data.getparams(), data.readframes(data.getnframes())] )
                self.fs2=FS 
                # song=relpath(fileName, 'C:/Users/Lenovo/Desktop/task4/database')
                # self.index2 = self.songs.index(song) 
                # arr=np.array(self.data2,dtype=np.float64)
                # self.data2=arr.astype(np.int16)
                # print (index)
                # data.close()
                # print(len(self.data))

    def valuechange(self):
        self.MixerValue=self.ui.mixer.value()/100
        # fill_value = 0

        # if self.data1.shape[0]>self.data2.shape[0]:
        #     temp = self.data2
        #     self.data2 = np.ones(self.data1.shape)*fill_value
        #     self.data2[:temp.shape[0],:] = temp
        # elif self.data1.shape[0]<self.data2.shape[0]:
        #     temp = self.data1
        #     self.data1 = np.ones(self.data2.shape)*fill_value
        #     self.data1[:temp.shape[0],:] = temp
        # print(self.data1.shape[0])
        # print(self.data2.shape[0])
        


        output=self.data1*self.MixerValue+self.data2*(1-self.MixerValue)
        data=np.array(output,dtype=np.float64)
        
        output=data.astype(np.int16)
     

        print(output)
        write("output.wav", 44100, output)

        SG_Output=plt.specgram(output[:,0], Fs=self.fs1, NFFT=128, noverlap=0)
        ax = plt.axes()
        ax.set_axis_off()
        plt.savefig('output.png', bbox_inches='tight',  transparent=True,pad_inches=0, frameon='false')
        file='output.png'
        img = Image.open(file)
        self.hashedVersion = imagehash.phash(img)
        print(self.hashedVersion)
        # hash1=self.arrayofHash[self.index1]
        
        # hash2=self.arrayofHash[self.index2]
        # print(hash1)
        # print(hash2)

        for i in range(len(self.arrayofHash)):
            diff=self.hashedVersion-self.arrayofHash[i]    
            sim=diff/64
            sim=1-sim

            # sim=difflib.SequenceMatcher(None,str(hashedVersion) ,str(self.arrayofHash[i])).ratio()
            self.similarity.append(str(sim)+" "+self.songs[i])
            # self.similarity.append(self.songs[i])
        self.similarity.sort(reverse = True) 
        for i in range (6):
            self.ui.table1.setItem(i, 0,QtWidgets.QTableWidgetItem(str(self.similarity[i])))
        print(self.similarity)
        # print(hash1+hash2)
        # for i in range(len(self.arrayofHash)):
        #     res = len(set(hashedVersion) & set(self.arrayofHash[i])) / float(len(set(hashedVersion) | set(self.arrayofHash[i]))) * 100
        #     print(res)


        # outfile = "sounds.wav"
        # out = wave.open(outfile, 'wb')
        # out.setparams(output[0][0])
        # for params,frames in output:
        #     out.writeframes(fs)
        # out.close()
        # FS, data = wavfile.read("sounds.wav")  # read wav file
        # self.ls=(plt.specgram(data[:,0], Fs=FS, NFFT=128, noverlap=0))  # The spectogram
        # self.ls=self.ls[:60]
        # ax = plt.axes()
        # ax.set_axis_off()
        # plt.savefig('mixing.png', bbox_inches='tight',  transparent=True,pad_inches=0, frameon='false')
        # file='mixing.png'
        # img = Image.open(file)
        # hashedVersion = imagehash.phash(img)
        # print(str(hashedVersion))


        # self.MixerValue=self.ui.mixer.value()/100
        # print(self.data1)
        # print(self.data2)
        # print(self.fs1)
        # print(self.fs2)
        # self.data1=self.data1*self.MixerValue
        # self.data2=self.data2*(1-self.MixerValue)
        # self.fs1=self.fs1*self.MixerValue
        # self.fs2=self.fs2*(1-self.MixerValue)
        # print(self.data1)
        # print(self.data2)
        # print(self.fs1)
        # print(self.fs2)

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