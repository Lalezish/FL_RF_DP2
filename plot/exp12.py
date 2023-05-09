import matplotlib.pyplot as plt
eps = ['Normal', '0.01', '0.1', '0.5', '1.0', '5.0', '10.0', '50.0', '100.0']
aucs = [0.7529128999018001, 0.6456962038395044, 0.7069769860427081, 0.7084359272682187, 0.6939819613086563, 0.7088527676183646, 0.7098271453459575, 0.7088475168866074, 0.7016935077863553]
plt.bar(eps, aucs, label='Stack 1')

plt.ylim(0.0, 1.0)
plt.xlabel('Epsilon')
plt.ylabel('AUC on test-set')
plt.title('Random Forests overview')

plt.show()
