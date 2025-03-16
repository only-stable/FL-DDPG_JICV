import matplotlib.pyplot as plt



def plot(loss, x_label, y_label, epi):
	plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.savefig(y_label + '_' + str(epi) + '.png')


def pllot(avr_reward, x_label, y_label, epi):
	plt.plot([i+1 for i in range(0, len(avr_reward), 2)], avr_reward[::2])
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.savefig(y_label + '_' + str(epi) + '.png')


