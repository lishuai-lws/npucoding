import os
functional_path='D:/data/output/emo-db_functional'
lld_path='D:/data/output/emo-db_lld'
lld_list=os.listdir(lld_path)
for lld in lld_list:
	if lld[-4:]=='.csv':
		this_path_input=os.path.join(lld_path,lld)
		this_path_output=os.path.join(functional_path,lld)
		cmd='D: && cd D:/opensmile/opensmile-2.3.0_0925/bin/Win32 \
		&& SMILExtract_Release -C D:/opensmile/opensmile-2.3.0_0925/config1/functional.conf -I ' \
		+ this_path_input +' -D '+ this_path_output
		os.system(cmd)
print('finish!')