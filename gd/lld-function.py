# -*- coding: utf-8 -*-
import numpy as np
import os

audio_path=r'F:\goodgoodstudy\CHEAVD1.0_reclass\epdet3\wave-ed\01'
output_path=r'G:\lld\txt\01'
output_lld=r'G:\lld\lld\01'
output_func=r'G:\lld\func\01'
audio_list=os.listdir(audio_path)
for audio in audio_list:
    if audio[-4:]=='.wav':
        this_path_input=os.path.join(audio_path,audio)
        this_path_output=os.path.join(output_path,audio[:-4]+'.txt')
        this_path_output_lld=os.path.join(output_lld,audio[:-4]+'.csv')
        this_path_output_func=os.path.join(output_func,audio[:-4]+'.csv')
        #cmd='cd /d E:/studytools/opensmile-2.3.0/bin/Win32 && SMILExtract_Release -C E:/studytools/opensmile-2.3.0/config/IS09_emotion.conf -I '+this_path_input+' -O '+this_path_output
        cmd1='cd /d E:/studytools/opensmile-2.3.0/bin/Win32 && SMILExtract_Release -C E:/studytools/opensmile-2.3.0/config1/is13lld.conf -I '+this_path_input+' -D '+this_path_output_lld
        cmd2='cd /d E:/studytools/opensmile-2.3.0/bin/Win32 && SMILExtract_Release -C E:/studytools/opensmile-2.3.0/config1/functional.conf -I '+this_path_output_lld+' -D '+this_path_output_func
    os.system(cmd1)
    os.system(cmd2)
    