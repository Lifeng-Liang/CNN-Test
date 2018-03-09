width = 42;

inputArgs = [
	[7,1,2],
	[2,2,0],

	[5,1,2],
	[2,2,0],

	[3,1,0],
	[2,2,0]
];

function getNewWidth(w, p, k, s) {
	return (w + p * 2 - k) / s + 1;
}

inputArgs.forEach(a => {
	var k = Number(a[0]);
	var s = Number(a[1]);
	var p = Number(a[2]);
	var o = width;
	var w = getNewWidth(width, p, k, s);
	width = Math.floor(w)
	console.log(o, k, s, p, width===w ? ' ' : '*', '=>', width)
})
