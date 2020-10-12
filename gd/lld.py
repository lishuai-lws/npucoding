import os
audio_path='D:/data/emo-db/wav'
lld_path='D:/data/output/emo-db_lld'
audio_list=os.listdir(audio_path)
for audio in audio_list:
	if audio[-4:]=='.wav':
		this_path_input=os.path.join(audio_path,audio)
		this_path_output=os.path.join(lld_path,audio[:-4]+'.csv')
		cmd='D: && cd D:/opensmile/opensmile-2.3.0_0925/bin/Win32 \
		&& SMILExtract_Release -C D:/opensmile/opensmile-2.3.0_0925/config1/is13lld.conf -I ' \
		+ this_path_input +' -D '+ this_path_output
		os.system(cmd)
print('finish!')