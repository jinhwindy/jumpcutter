from contextlib import closing
from PIL import Image
import subprocess
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import copyfile, rmtree
import os
import argparse
from pytube import YouTube

def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

def copyFrame(inputFrame,outputFrame):
    src = TEMP_FOLDER+"/frame{:06d}".format(inputFrame+1)+".jpg"
    dst = TEMP_FOLDER+"/newFrame{:06d}".format(outputFrame+1)+".jpg"
    if not os.path.isfile(src):
        return False
    copyfile(src, dst)
    if outputFrame%20 == 19:
        print(str(outputFrame+1)+" time-altered frames saved.")
    return True

def inputToOutputFilename(filename):
    dotIndex = filename.rfind(".")
    return filename[:dotIndex]+"_ALTERED"+filename[dotIndex:]

def createPath(s):
    try:  
        os.mkdir(s)
    except OSError:  
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"

def deletePath(s): # Dangerous! Watch out!
    try:  
        rmtree(s,ignore_errors=False)
    except OSError:  
        print ("Deletion of the directory %s failed" % s)
        print(OSError)

SampleRate=44100
Threshold=0.04
FramMargin=1
Speed=[5,2]
FramQuality=3

TEMP_FOLDER = "TEMP"
createPath(TEMP_FOLDER)

InputName=input("Please input the name of video:")
OutputName=input("Please input the name of output:")

command = "ffmpeg -i "+InputName+" -qscale:v "+str(FramQuality)+" "+TEMP_FOLDER+"/frame%06d.jpg -hide_banner"
subprocess.call(command, shell=True)

command = "ffmpeg -i "+InputName+" -ab 160k -ac 2 -ar "+str(SampleRate)+" -vn "+TEMP_FOLDER+"/audio.wav"
subprocess.call(command, shell=True)

command = "ffmpeg -i "+InputName+" 2>&1"
f = open(TEMP_FOLDER+"/params.txt", "w")
subprocess.call(command, shell=True, stdout=f)
f.close()

f = open(TEMP_FOLDER+"/params.txt", 'r+')
pre_params = f.read()
f.close()
params = pre_params.split('\n')
for line in params:
    m = re.search('Stream #.*Video.*fps',line)
    if m is not None:
        print(m.group(0))
        tempString=str(m.group())
        pattern = re.compile("\\d*\.\\d* fps")
        m1=pattern.search(tempString,-10)

        if m1 is not None:
            print("WTF")
            print(m1.group(0))
            ImportantNum=(float)(m1.group(0)[:-4])
print(ImportantNum)
FrameRate=int(math.ceil(ImportantNum))

sampleRate, audioData = wavfile.read(TEMP_FOLDER+"/audio.wav")
audioSampleCount = audioData.shape[0]
maxAudioVolume = getMaxVolume(audioData)

# 每帧的声音采样数
samplesPerFrame = sampleRate/FrameRate
# 采样总数/每帧采样数=声音帧数,即视频有多少帧（ceil:返回大于等于参数x的最小整数）
audioFrameCount = int(math.ceil(audioSampleCount/samplesPerFrame))

# HasLoudVolume有可能含有只在一两帧中出现声音，这个声音应该不是真正的讲话声
FadeSize = 400 

HasLoudVolume = np.zeros((audioFrameCount))

# 视频帧为参考，每帧视频包含多帧音频//////////////////////////////////////////
for i in range(audioFrameCount):
    start = int(i*samplesPerFrame)
    end = min(int((i+1)*samplesPerFrame),audioSampleCount)
    audioSoundFragment = audioData[start:end]
    maxSoundFragmentVolume = float(getMaxVolume(audioSoundFragment))/maxAudioVolume
    # 当音量大于设定的阈值
    if maxSoundFragmentVolume >= Threshold:
        HasLoudVolume[i] = 1

# SoundFragment [n][0]:start;
#               [n][1]:end;
#               [n][2]:silent Fragment or sounded Fragment(1:sounded;0:silent)
SoundFragment = [[0,0,0]]
shouldIncludeFrame = np.zeros((audioFrameCount))
for i in range(audioFrameCount):
    start = int(max(0,i-FramMargin))
    end = int(min(audioFrameCount,i+1+FramMargin))
    #判断该段帧内是否有声音（大于阈值）,如果有，shouldInclude[i]填1
    shouldIncludeFrame[i] = np.max(HasLoudVolume[start:end])
    # 当前帧被判断有无声音的结果与上一帧不同，如果太短应该是干扰，剔除；如果大于FADE值，应确实为说话片段
    if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]): # Did we flip?
        SoundFragment.append([SoundFragment[-1][1],i,shouldIncludeFrame[i-1]])

SoundFragment.append([SoundFragment[-1][1],audioFrameCount,shouldIncludeFrame[i-1]])
SoundFragment = SoundFragment[1:]

outputAudioData = np.zeros((0,audioData.shape[1]))
outputPointer = 0

# 视频帧为参考，每帧视频包含多帧音频//////////////////////////////////////////
lastExistingFrame = None
for Fragment in SoundFragment:
    # audioFragment:当前帧包含的声音帧
    audioChunk = audioData[int(Fragment[0]*samplesPerFrame):int(Fragment[1]*samplesPerFrame)]
    
    sFile = TEMP_FOLDER+"/tempStart.wav"#每帧变速前的声音文件
    eFile = TEMP_FOLDER+"/tempEnd.wav"#每帧变速后的声音文件
    wavfile.write(sFile,SampleRate,audioChunk)
    with WavReader(sFile) as reader:
        with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=Speed[int(Fragment[2])])
            tsm.run(reader, writer)
    _, alteredAudioData = wavfile.read(eFile)
    leng = alteredAudioData.shape[0]#变速后的声音帧数
    # outputPointer初始值为0，每次与变速后的声音长度相加，计算当前处理后声音的长度
    endPointer = outputPointer+leng
    # 将每帧的声音数据不断累加
    outputAudioData = np.concatenate((outputAudioData,alteredAudioData/maxAudioVolume))

    # smooth out transitiion's audio by quickly fading in/out
    
    if leng < FadeSize:
        # 大于阈值的噪声而已
        outputAudioData[outputPointer:endPointer] = 0

    
    # outputPointer、endPointer为音频指针
    # 以变速后的音频除以每帧的音频采样率得出变速后的视频应该有的帧数
    startOutputFrame = int(math.ceil(outputPointer/samplesPerFrame))
    endOutputFrame = int(math.ceil(endPointer/samplesPerFrame))
    # 在音频内实际原有的视频帧数可能大于endOutputFrame-startOutputFrame
    for outputFrame in range(startOutputFrame, endOutputFrame):
        # Fragment中为变速前视频帧数
        inputFrame = int(Fragment[0]+Speed[int(Fragment[2])]*(outputFrame-startOutputFrame))
        didItWork = copyFrame(inputFrame,outputFrame)
        if didItWork:
            lastExistingFrame = inputFrame
        else:
            copyFrame(lastExistingFrame,outputFrame)

    outputPointer = endPointer

wavfile.write(TEMP_FOLDER+"/audioNew.wav",SampleRate,outputAudioData)
command = "ffmpeg -framerate "+str(FrameRate)+" -i "+TEMP_FOLDER+"/newFrame%06d.jpg -i "+TEMP_FOLDER+"/audioNew.wav -strict -2 "+OutputName
subprocess.call(command, shell=True)

deletePath(TEMP_FOLDER)

