test = []
val  = []
avg  = []

def show(name, list):
	print(name)
	batchs = [1,10,50,100,300,600,1000,1400]
	x = 0
	for y in batchs:
		print('%4d %.2f' % (y, max(list[x:y])))
		x = y

with open('99999.txt', 'r') as f:
	for line in f.readlines():
		if(line.startswith('Epoch:')):
			ss = line.split('|')
			test.append(float(ss[2].split(':')[1]))
			val.append(float(ss[3].split(':')[1]))
			avg.append(float(ss[4].split(':')[1]))

show('test:', test)
show('val:', val)
show('avg:', avg)
