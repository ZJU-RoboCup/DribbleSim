import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_img(file_name):
	csvFile = open(file_name, "r")
	reader = csv.reader(csvFile)
	
	step = []
	value = []

	for item in reader:
	    # ignore the first line
	    if reader.line_num == 1:
	        continue
	    step.append(float(item[1]))
	    value.append(float(item[2]))
	
	csvFile.close()

	print(step[0])
	print(value[0])

	plt.plot(step,value,"b-",linewidth=1)
	plt.xlabel('Steps')
	plt.ylabel('Ball position/m')
	#plt.ylabel('Dribbling system angle/rad')
	#plt.ylabel('Reward')
	#plt.ylabel('Torque/mN·m')
	plt.savefig(file_name[:-3]+"eps")
	plt.savefig(file_name[:-3]+"png")
	plt.show()

if __name__ == '__main__':
	plot_img("dist.csv")