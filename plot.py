import matplotlib.pyplot as plt

test = []
val  = []
avg  = []

def read(list):
	x = 0
	tx = []
	ty = []
	for y in range(10, 1400, 10):
		tx.append(y)
		ty.append(max(list[x:y]))
		x = y
	return tx,ty

with open('99999.txt', 'r') as f:
	for line in f.readlines():
		if(line.startswith('Epoch:')):
			ss = line.split('|')
			test.append(float(ss[2].split(':')[1]))
			val.append(float(ss[3].split(':')[1]))
			avg.append(float(ss[4].split(':')[1]))

tx,ty = read(test)
vx,vy = read(val)

plt.title('Prediction Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.plot(tx, ty, 'b', label='Test Dataset')
plt.plot(vx, vy, 'r', label='Validation Dataset')

plt.legend(loc='lower right')
plt.grid()
plt.show()
