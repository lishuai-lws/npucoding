# 导入相应的包
#-*- coding: utf-8 -*-
import numpy, wave
import matplotlib.pyplot as plt
import numpy as np
import os

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for i in range(5):
	file_path = '../iemocap/segment/Session'+str(i+1)
	spectrogramfile_path='../iemocap/spectrogram/Session'+str(i+1)
	file_list=os.listdir(file_path)
	filenum=0
	print('Session',i+1)
	for file in file_list:
		print(file)
		filenum = filenum + 1
		audio_path=os.path.join(file_path,file)
		spectrogram_path=os.path.join(spectrogramfile_path,file)
		isExists = os.path.exists(spectrogram_path)
		if isExists:
			continue
		if not isExists:
			os.makedirs(spectrogram_path)
		audio_list=os.listdir(audio_path)
		for audio in audio_list:
			print(audio)
			if audio[-4:]=='.wav':
				this_path_input=os.path.join(audio_path,audio)
				this_path_output=os.path.join(spectrogram_path,audio[:-4]+'.png')
				f = wave.open(this_path_input,'rb')
				# 得到语音参数
				params = f.getparams()
				nchannels, sampwidth, framerate,nframes = params[:4]
				# 得到的数据是字符串，需要将其转成int型
				strData = f.readframes(nframes)
				wavedata = np.frombuffer(strData,dtype=np.int16)
				# 归一化
				wavedata = wavedata * 1.0/max(abs(wavedata))
				# .T 表示转置
				wavedata_emphasis = [wavedata[0]]
				for i in range(1, len(wavedata)):
					wavedata_emphasis.append(wavedata[i] - 0.98 * wavedata[i - 1])  # 预加重
				wavedata_emphasis = np.asarray(wavedata_emphasis)
				# wavedata = wavedata * 1.0 / (max(abs(wavedata)))  # wave幅值归一化
				wavedata_emphasis = wavedata_emphasis * 1.0 / (max(abs(wavedata_emphasis)))  # 预加重wave幅值归一化
				wavedata_emphasis = np.reshape(wavedata_emphasis,[nframes,nchannels]).T
				f.close()
				NFFT=960
				# 绘制频谱
				plt.specgram(wavedata_emphasis[0], NFFT=NFFT, window=np.hamming(NFFT), noverlap=NFFT*2/3, Fs=framerate ,scale_by_freq=True, sides='default')
				plt.ylim(0, 8000)
				plt.axis('off')
				plt.savefig(this_path_output,bbox_inches='tight',pad_inches = 0)
		print('filenum:', filenum)
print('finish!')