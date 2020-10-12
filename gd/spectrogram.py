# 导入相应的包
import numpy, wave
import matplotlib.pyplot as plt
import numpy as np
import os


audio_path = 'D:/data/emo-db/wav'
spectrogram_path='D:/data/output/emo-db_spectrogram'
audio_list=os.listdir(audio_path)
for audio in audio_list:
	if audio[-4:]=='.wav':
		this_path_input=os.path.join(audio_path,audio)
		this_path_output=os.path.join(spectrogram_path,audio[:-4]+'.png')
		f = wave.open(this_path_input,'rb')
		# 得到语音参数
		params = f.getparams()
		nchannels, sampwidth, framerate,nframes = params[:4]
		# 得到的数据是字符串，需要将其转成int型
		strData = f.readframes(nframes)
		wavaData = np.frombuffer(strData,dtype=np.int16)
		# 归一化
		wavaData = wavaData * 1.0/max(abs(wavaData))
		# .T 表示转置
		wavaData = np.reshape(wavaData,[nframes,nchannels]).T
		f.close()
		# 绘制频谱
		plt.specgram(wavaData[0],Fs = framerate,scale_by_freq=True,sides='default')
		plt.axis('off')
		plt.savefig(this_path_output,bbox_inches='tight',pad_inches = 0)
print('finish!')