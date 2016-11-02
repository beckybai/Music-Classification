import pandas as pd
import numpy as np
import matplotlib

testdata = pd.read_csv("./10000.log")

index = testdata['Num']
accuracy = testdata['accuracy']
loss = testdata['loss']

nn = -1

for i in range(1,3000,15):
	print ("this is #",i)
	ww =sum(accuracy[i:i+15]>0)
	print float(sum(accuracy[i:i+15]>0))
	if(ww>7):
		nn = nn+1

print nn

