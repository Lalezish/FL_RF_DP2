import matplotlib.pyplot as plt

eps = ['0.01', '0.1', '0.5', '1.0', '5.0', '10.0', '50.0', '100.0']
aucs = [0.6443832103730521, 0.7095510020231126, 0.5462785664508616, 0.5464922373576917, 0.5461431201553177, 0.7098271453459575, 0.5462838171826188, 0.5468413545600658]

plt.bar(eps, aucs, label='Stack 1')
plt.ylim(0.0, 1.0)
plt.xlabel('Epsilon values')
plt.ylabel('AUC on test-set')
plt.title('Differentially private Random Forests with Federated Learning')
plt.show()
