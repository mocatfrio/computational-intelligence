### LOAD LIBRARY ###
import csv
import random
import math
import operator
import numpy

### HANDLE DATA ###
#Fungsi untuk menge-load dataset dan membaginya jadi data train dan data test 
def loadDataset (filename, split, trainingSet=[], testSet=[]):
	with open(filename, 'rb') as csvfile:
		#Membacanya dengan format CSV (Comma Separated Value) -> untuk memisahkan data (biasanya dengan , atau ;)
		lines = csv.reader(csvfile, delimiter = ' ', skipinitialspace = True) #Delimiter untuk membatasi/memisahkan data
		dataset = list(lines) #Membuat data menjadi array 2 dimensi
		flag = 0 #Flag untuk menandai data masuk data train atau data tes
		
	#Handle missing value dengan rata-rata
	#Mengeceknya per-kolom dari atas ke bawah, lalu dilanjutkan ke kolom selanjutnya 
	for x in range(len(dataset)):
		for y in range (len (dataset[x]) - 1):
			if float (dataset[x][y]) == 0.0:
				kolom = [float (i[y]) for i in dataset]
				tidakNol = len(kolom) - kolom.count (0)
				dataset[x][y] = float(sum(kolom)/tidakNol) #Mencari rata-rata
                	else:
                    		dataset[x][y] = float(dataset[x][y])

	#Fungsi Normalisasi
	for x in range(len(dataset)):
		#Mencari nilai paling minimal dalam data
		minx = min([i for i in dataset[x][:-1]])
		#Mencari nilai paling maksimal dalam data
		maxx = max([i for i in dataset[x][:-1]])
		for y in range (len (dataset[x]) - 1):
			#Sesuai dengan rumus normalisasi
			dataset[x][y] = (dataset[x][y] - minx) / (maxx - minx) 
 
	#Membagi data training dan data testing
	for x in range(len(dataset)-1):
		for y in range(4):
			dataset[x][y] = float (dataset[x][y])
		#Saya membaginya 80% Training 20% Testing, sehingga jumlah datanya konsisten dalam beberapa kali percobaan
		if flag < split * len(dataset):
			trainingSet.append (dataset[x])
			flag += 1 
		else:
			testSet.append (dataset[x])

### SIMILARITY ###
# EUCLIDEAN DISTANCE
#Fungsi menghitung jarak dengan Euclidean Distance
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		#Sesuai dengan rumus
		distance += float(pow((float(instance1[x]) - float(instance2[x])), 2)) 
	return float(math.sqrt(distance))

# MANHATTAN DISTANCE
#Fungsi menghitung jarak dengan Manhattan Distance
def manhattanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		#Sesuai dengan rumus
		temp = instance1[x] - instance2[x] 
        if temp < 0:
            temp*=-1
        distance += temp
	return math.sqrt(distance)

# COSINE SIMILARITY
#Fungsi menghitung jarak dengan Cosine Similarity
def cosineSimilarity(instance1, instance2, length):
	distance = 0
	sumxx, sumxy, sumyy = 0, 0, 0
	for i in range(length):
		x = float(instance1[i])
		y = float(instance2[i])
		sumxx += x * x
		sumyy += y * y
		sumxy += x * y
	#Sesuai dengan rumus
	return 1 - (sumxy / math.sqrt(sumxx * sumyy))


### K NEIGHBORS ###
#Fungsi untuk mencari data tetangga dengan jarak terkecil (tetangga terdekat) sejumlah k
def getNeighbors(trainingSet, testInstance, k, typeDist):
	distances = []
	length = len(testInstance) - 1
	for x in range(len(trainingSet)):
		#Memilih ingin menggunakan fungsi perhitungan jarak apa
		dist = typeDist(testInstance,trainingSet[x],length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

### RESPONSE -- MAJORITY VOTE OF CLASS ###
#Fungsi untuk menentukan prediksi class data test berdasarkan vote terbanyak dari class-class tetangga
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
	if response in classVotes:
		classVotes[response] += 1
	else:
		classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

### ACCURACY ###
#Fungsi untuk menghitung akurasi
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] is predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

### MAIN ###
def main():
	#Menyiapkan data
	trainingSet = []
	testSet = []
	split = 0.8 # Karena saya ingin 80% Training, maka saya split 0.8
	total_regresi = 0 
    
    #Load Dataset
    loadDataset('5115100043_housing.data', split, trainingSet, testSet)
	print('\nData Train: ' + repr(len(trainingSet)))
	print('Data Test: ' + repr(len(testSet)))
    
	#Generate predictions
	predictions = []
	print "\nMasukkan nilai k : "
	k = int(raw_input())
	print "\n1. Euclidean Distance\n2. Manhattan Distance\n3. Cosine Similarity\n"
	print "\nMasukkan pilihan algoritma yang ingin digunakan : "
	typeDist = [euclideanDistance, manhattanDistance, cosineSimilarity][int(raw_input())]
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k, typeDist) # Cari k-data terdekat
		# Fungsi menghitung regresi (Regresi dihitung dari rata-rata properti data sejumlah k terdekat)
		regresiBukan = sum ([n[-1] for n in neighbors]) / k
		regresi = sum ([typeDist(testSet[x], n, len(testSet[x])) for n in neighbors]) / k
		total_regresi += regresi
		result = getResponse(neighbors) # Simpulkan class nya dengan majority vote
		predictions.append(result) # Prediksi dan aktual
		print('> Predicted=' + repr(result) + ', Actual=' + repr(testSet[x][-1]))
		for n in neighbors:
			print (n[-1])
		print ('Regresi bukan rumus : ' + str(regresiBukan))
	accuracy = getAccuracy(testSet, predictions)
	print('Regresi : ' + str(total_regresi/len(testSet)))
	
main()
