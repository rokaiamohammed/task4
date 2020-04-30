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
from matplotlib.mlab import specgram
from skimage.feature import peak_local_max
import math
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
        self.similarityFeature1=[]
        self.avrage=[]
        self.length=[]
        self.sumHashAndFeature=[]
        self.ui.mixer.setValue(0)
        self.ui.mixer.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.ui.mix.clicked.connect(self.mixing)
        self.data_dir='./database/macro-output/'
        self.SG=glob(self.data_dir+'*.wav')
        for i in range(len(self.SG)):
            self.songs.append(relpath(self.SG[i], './database\\'))
        self.spectral_Centroid=[]
        self.SG_Maker()
        self.Feature1()
        self.Feature2()

        self.arrayofHash=self.Hash('./SG_DataBase/')
        # print(self.arrayofHash[2])
        self.arrayofHashFeature=self.Hash('./SG_DataBaseFeature/')
        # print(self.arrayofHashFeature[2])


       
        

        # print(len(self.arrayofHashFeature))
        #     # print(self.ls)
       
        # print(self.freq)
        # print(self.time)
        # print(coordinates)
    def Hash(self,folder):
        arrayofHash=[]
        self.data_dir_SG=folder
        self.files=glob(self.data_dir_SG+'/*.png')
        for i in range(len(self.files)): 
            file=self.files[i]
            img = Image.open(file)
            hashedVersion = imagehash.phash(img)
            arrayofHash.append(hashedVersion)
        return arrayofHash
    
    def SG_Maker(self):
        for i in range(len(self.SG)):
            FS, data = wavfile.read(self.SG[i])  # read wav file
            data=data[0:60*FS]
            if data.ndim==2:
                plt.specgram(data[:,0], Fs=FS,  NFFT=128, noverlap=0)   
            else: 
                plt.specgram(data, Fs=FS,  NFFT=128, noverlap=0)   
                # self.fs1=FS
            #plt.show() #if you want to show the spectogram
            
            # ======= This part may help in getting the first minute ==========    
              
            ax = plt.axes()
            ax.set_axis_off()
            plt.savefig('./SG_DataBase/sp_xyz' + str(i)+'.png', bbox_inches='tight',  transparent=True,pad_inches=0, frameon='false')
           
            
    def Feature1(self):
        for i in range(len(self.SG)):
            FS, data = wavfile.read(self.SG[i])  # read wav file
            data=data[0:60*FS]
            if data.ndim==2:
                self.ls,self.freq, self.time =specgram(data[:,0], Fs=FS,  NFFT=4096, noverlap=2048)  # The spectogram
            else: 
                self.ls,self.freq, self.time =specgram(data, Fs=FS,  NFFT=4096, noverlap=2048)  # The spectogram
            self.ls[self.ls == 0] = 1e-6

            Z1, freqs1 = self.cutSpecgram(self.ls, self.freq)
            coordinates = peak_local_max(Z1, min_distance=20, threshold_abs=20)
            self.showPeaks(Z1, freqs1, self.time, coordinates,i)

    def Feature2(self):
        for i in range(len(self.SG)):
            FS, data = wavfile.read(self.SG[i])  # read wav file
            data=data[0:60*FS]
            if data.ndim==2:
                self.ls,self.freq, self.time =specgram(data[:,0], Fs=FS,  NFFT=4096, noverlap=2048)  # The spectogram
            else: 
                self.ls,self.freq, self.time =specgram(data, Fs=FS,  NFFT=4096, noverlap=2048)  # The spectogram            
            self.ls[self.ls == 0] = 1e-6
            self.spectral_Centroid.append(self.spectralCentroid(self.ls))




    def browse(self,n):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Please Choose a .wav file", "","Wav Files (*.wav)", options=options)
        if fileName:

            FS, data = wavfile.read(fileName)  # read wav file
            data=data[0:60*FS]
            if n==1:
                if data.ndim==2:
                    self.data1=data[:,0]
                else: 
                    self.data1=data
                self.fs1=FS
                
            elif n==2:
                if data.ndim==2:
                    self.data2=data[:,0]
                else:
                    self.data2=data
                self.fs2=FS 
    
    
    def spectralCentroid(self,arr,samplerate=44100 ):
        # magnitudes = np.abs(np.fft.rfft(arr))
        # length = len(arr)
        # freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1])
        # magnitudes = magnitudes[:length//2+1]
        # return np.sum(magnitudes*freqs) / np.sum(magnitudes) 
        h = arr.shape[0]
        w = arr.shape[1]
        x = np.arange(w)
        y = np.arange(h)
        vx = arr.sum(axis=0)
        vx =vx / vx.sum()
        vy = arr.sum(axis=1)
        vy = vy / vy.sum()    
        return np.dot(vx,x),np.dot(vy,y)

    def cutSpecgram(self, spec, freqs):
        min_freq=0
        max_freq=10000
        spec_cut = spec[(freqs >= min_freq) & (freqs <= max_freq)]
        freqs_cut = freqs[(freqs >= min_freq) & (freqs <= max_freq)]
        Z_cut = 10.0 * np.log10(spec_cut)
        Z_cut = np.flipud(Z_cut)
        return Z_cut, freqs_cut

    def showPeaks(self, Z, freqs, t, coord,i):
        fig = plt.figure(figsize=(10, 8), facecolor='white')
        plt.imshow(Z, cmap='viridis')
        plt.scatter(coord[:, 1], coord[:, 0])
        ax = plt.gca()
        plt.xlabel('Time bin')
        plt.ylabel('Frequency')
        plt.title('Feature: Peaks', fontsize=18)
        plt.axis('auto')
        ax.set_xlim([0, len(t)])
        ax.set_ylim([len(freqs), 0])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        if i=="output":
            plt.savefig('FeaturePeak' + str(i)+'.png', bbox_inches='tight',  transparent=True,pad_inches=0, frameon='false')
        else:
            plt.savefig('./SG_DataBaseFeature/FeaturePeak' + str(i)+'.png', bbox_inches='tight',  transparent=True,pad_inches=0, frameon='false')




    def mixing(self):
        self.MixerValue=self.ui.mixer.value()/100

        output=self.data1*self.MixerValue+self.data2*(1-self.MixerValue)
        data=np.array(output,dtype=np.float64)
        
        output=data.astype(np.int16)
     

        # print(output)
        write("output.wav", 44100, output)
        plt.specgram(output, Fs=self.fs1, NFFT=128, noverlap=0)
        
        ax = plt.axes()
        ax.set_axis_off()
        plt.savefig('output.png', bbox_inches='tight',  transparent=True,pad_inches=0, frameon='false')
        file='output.png'
        img = Image.open(file)
        self.hashedVersion = imagehash.phash(img)
        print(self.hashedVersion)

        for i in range(len(self.arrayofHash)):
            diff=self.arrayofHash[i]-self.hashedVersion    
            sim=diff/64
            sim=1-sim
            self.similarity.append(sim)

            # self.similarity.append(str(sim)+" "+self.songs[i])
        # self.similarity.sort(reverse = True) 
       
        # print(self.similarity)

        self.lsOUT,self.freqOUT, self.timeOUT =specgram(output, Fs=self.fs1,  NFFT=4096, noverlap=2048)  # The spectogram
        self.lsOUT[self.lsOUT == 0] = 1e-6
        Z1, freqs1 = self.cutSpecgram(self.lsOUT, self.freqOUT)
        coordinates = peak_local_max(Z1, min_distance=20, threshold_abs=20)
        self.showPeaks(Z1, freqs1, self.timeOUT, coordinates,"output")
        self.spectral_CentroidOutput=self.spectralCentroid(self.lsOUT)
        print(self.spectral_CentroidOutput)

        fileFeature='FeaturePeakoutput.png'
        imgFeature = Image.open(fileFeature)
        self.FeatureHash = imagehash.phash(imgFeature)
        # print(self.FeatureHash)
        
        for j in range(len(self.arrayofHashFeature)):
            diffFeature=self.arrayofHashFeature[j]-self.FeatureHash  
            simFeature=diffFeature/64
            simFeature=1-simFeature
            self.similarityFeature1.append(simFeature)
            # self.similarityFeature1.append(str(simFeature)+" "+self.songs[j])
        # self.similarityFeature1.sort(reverse = True) 
        # print(self.similarityFeature1)


        # for k in range (len(self.arrayofHashFeature)):
        #     avr=(self.similarity[k]+self.similarityFeature1[k])/2
        #     self.avrage.append(str(avr)+" "+self.songs[k])
        # self.avrage.sort(reverse = True) 
        # print(self.avrage)
       




            
        for j in range(len(self.spectral_Centroid)):
            diffFeature2X=self.spectral_Centroid[j][0]-self.spectral_CentroidOutput[0] #x2-x1
            diffFeature2Y=self.spectral_Centroid[j][1]-self.spectral_CentroidOutput[1] #y2-y1
            diffFeature2X=math.pow(diffFeature2X,2)
            diffFeature2Y=math.pow(diffFeature2Y,2)
            lenth=math.sqrt(diffFeature2X+diffFeature2Y)
            # diffFeature2=np.subtract(self.spectral_Centroid[j],self.spectral_CentroidOutput)

            # simFeature2=np.divide(diffFeature2,self.spectral_Centroid[j])
            # simFeature2=1-simFeature2
            # print(simFeature2)
            # print(lenth)
            self.length.append(lenth)
        # self.length.sort(reverse = False) 
        # print(self.length)
        for i in range(len(self.spectral_Centroid)):
            self.sumHashAndFeature.append(str(self.similarityFeature1[i]+self.similarity[i])+" "+self.songs[i])
        self.sumHashAndFeature.sort(reverse = True) 
        # print(self.similarityFeature1)
       


        for i in range (6):
            self.ui.table1.setItem(i, 0,QtWidgets.QTableWidgetItem(str(self.sumHashAndFeature[i])))

    

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()