import numpy as np
import matplotlib.pyplot as plt

eps = ['Normal' ,'0.01', '0.1', '0.5', '1.0', '5.0', '10.0', '50.0', '100.0']
aucsL = [0.7068439110453772, 0.5471852210306825, 0.5471852210306825, 0.5469715501238523, 0.5467683806805366, 0.5470445240033814, 0.5467631299487794, 0.5472529441784544, 0.5472529441784544]
aucsR = [0.5906058367698808, 0.5445058798879867, 0.5455178596300989, 0.546007673859774, 0.5464245142099199, 0.5466276836532356, 0.546289067914376, 0.5465599605054636, 0.5463515403303907]
aucsB = [0.7405831090056993, 0.617569981668171, 0.6219987762972032, 0.6853690109828935, 0.6907254631187755, 0.6734213378559636, 0.6850876169282915, 0.6967591467323765, 0.6908661601460765]

bar_width = 0.25

index = np.arange(len(eps))
bar_pos_L = index
bar_pos_R = index + bar_width
bar_pos_B = index + 2 * bar_width

plt.bar(bar_pos_B, aucsB, width=bar_width, label='Bottom')
plt.bar(bar_pos_L, aucsL, width=bar_width, label='Left')
plt.bar(bar_pos_R, aucsR, width=bar_width, label='Right')

plt.ylim(0.0, 1.0)

plt.xlabel('Epsilon values')
plt.ylabel('AUC on test-set')
plt.title('Client-trained Random Forests overview')
plt.xticks(index + bar_width, eps)

plt.legend()
plt.show()
