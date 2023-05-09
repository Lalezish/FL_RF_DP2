import matplotlib.pyplot as plt

categories = ['RF', 'RF & FL', 'RF-LEFT', 'RF-RIGHT', 'RF-BOTTOM']
values1 = [0.7529128999018001, 0.7491524443821884, 0.7068439110453772, 0.5906058367698808, 0.7405831090056993]
plt.bar(categories, values1, label='Stack 1')

plt.xlabel('Implementations')
plt.ylabel('AUC on test-set')
plt.title('Traditional Random Forest Approaches')
plt.show()
