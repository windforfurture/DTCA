# Example of the Mann-Whitney U Test
from scipy.stats import mannwhitneyu
with open("my_model_result.txt",'r',encoding="utf-8") as f1:
	a = f1.readline()
	data_1_1 = eval(a)
with open("my_model_result_a.txt",'r',encoding="utf-8") as f2:
	a = f2.readline()
	data_2_2 = eval(a)
data1 = []
data2 = []
for dataa in data_1_1:
	data1 += dataa
	
for dataa in data_2_2:
	data2 += dataa

# data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
# data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = mannwhitneyu(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')