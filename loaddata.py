import numpy as np

### data feature

# data = np.loadtxt('microarray.original.txt', skiprows=1, usecols=np.arange(1, 5897))
# np.save('microarray.original.npy', data)

'''
data = np.load('microarray.original.npy')
data = np.transpose(data, (1,0))
print (data.shape)
'''

### data label
label_file = open('E-TABM-185.sdrf.txt', 'r')
label_file.readline() ### skip the first row
label_lines = label_file.readlines()

'''
label_dict = {}
for i in range (5896):
        line = label_lines[i].split('\t')       
        if line[7] != '  ':
                # print(i,line[0],line[7],line[28])
                if line[7] not in label_dict:
                        label_dict[line[7]] = 1
                else:
                        label_dict[line[7]] += 1
for k in sorted(label_dict,key=label_dict.__getitem__,reverse=True):
        print(k,label_dict[k])
'''             

### 5 diseases are selected
disease_selected = ['negative5', 'breast tumor', 'acute myeloid leukemia', '"B-cell lymphoma, dlbcl"',
                    'breast cancer', '"acute lymphoblastic leukemia, chemotherapy response"'] 
label = np.zeros((5896,1))
for i in range(5896):
	_label = label_lines[i].split('\t')[7]
	if (_label not in disease_selected):
		label[i] = 0
	else:
		label[i] = disease_selected.index(_label)

for i in range(5896):
        if label[i] != 0:
                print (i, label[i])
