import math

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

#sort array of points according to X coordinate
def compareX(a, b):
	p1, p2 = a, b
	return (p1.x != p2.x) * (p1.x - p2.x) + (p1.y - p2.y)

#sort array of points according to Y coordinate
def compareY(a, b):
	p1, p2 = a, b
	return (p1.y != p2.y) * (p1.y - p2.y) + (p1.x - p2.x)


# A utility function to find the distance between two points
def dist(p1, p2):
	return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

'''A Brute Force method to return the smallest distance between two points
in P[] of size n'''
def bruteForce(P, n):
	min = float('inf')
	for i in range(n):
		for j in range(i+1, n):
			if dist(P[i], P[j]) < min:
				min = dist(P[i], P[j])
	return min

# A utility function to find a minimum of two float values
def min(x, y):
	return x if x < y else y

'''# A utility function to find the distance between the closest points of strip of a given size. All points in strip[] are sorted according to
y coordinate. They all have an upper bound on minimum distance as d.Note that this method seems to be a O(n^2) method, but it's a O(n) method as the inner loop runs at most 6 times'''
def stripClosest(strip, size, d):
	min = d
	for i in range(size):
		for j in range(i+1, size):
			if (strip[j].y - strip[i].y) < min:
				if dist(strip[i],strip[j]) < min:
					min = dist(strip[i], strip[j])

	return min

'''A recursive function to find the smallest distance. 
The array Px contains all points sorted according to x coordinates and Py contains all points sorted according to y coordinates'''
def closestUtil(Px, Py, n):
	if n <= 3:
		return bruteForce(Px, n)

	mid = n // 2
	midPoint = Px[mid]

	Pyl = [None] * mid
	Pyr = [None] * (n-mid)
	li = ri = 0
	for i in range(n):
		if ((Py[i].x < midPoint.x or (Py[i].x == midPoint.x and Py[i].y < midPoint.y)) and li<mid):
			Pyl[li] = Py[i]
			li += 1
		else:
			Pyr[ri] = Py[i]
			ri += 1

	dl = closestUtil(Px, Pyl, mid)
	dr = closestUtil(Px[mid:], Pyr, n-mid)

	d = min(dl, dr)
	strip = [None] * n
	j = 0
	for i in range(n):
		if abs(Py[i].x - midPoint.x) < d:
			strip[j] = Py[i]
			j += 1
	return stripClosest(strip, j, d)


def closest(P, n):
	Px = P
	Py = P
	Px.sort(key=lambda x:x.x)
	Py.sort(key=lambda x:x.y)

	return closestUtil(Px, Py, n)

#example usage
if __name__ == '__main__':
	P = [Point(2, 3), Point(12, 30), Point(40, 50), Point(5, 1), Point(12, 10), Point(3, 4)]
	n = len(P)
	print("The smallest distance is", closest(P, n))

'''output
The smallest distance is 1.4142135623730951
'''
