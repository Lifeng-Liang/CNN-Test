import math

width = 15

inputArgs = [
	[3,1,1],
	[3,1,1],
	[3,1,1],
	[1,1,1]
]

def getNewWidth(w, p, k, s):
	return (w + p * 2.0 - k) / s + 1

for a in inputArgs:
	k = int(a[0])
	s = int(a[1])
	p = int(a[2])
	o = width
	w = getNewWidth(width, p, k, s)
	width = math.floor(w)
	if width==w:
		print(o, k, s, p, ' ', '=>', width)
	else:
		print(o, k, s, p, '*', '=>', width)
