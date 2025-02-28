
import matplotlib.pyplot as plt

def plt_result(self, plt_data, path):
	plt.style.use('fivethirtyeight')
	plt.figure(figsize=(32, 16))
	plt.xlabel('stock data time', fontsize=18)
	plt.ylabel('Close Price', fontsize=18)
	plt.scatter(self.plt_data.index,self.plt_data['buy_signal'], color='green', label = 'buy', marker='^', alpha=1)
	plt.scatter(self.plt_data.index,self.plt_data['sell_signal'], color='red', label = 'sell', marker='v', alpha=1)
	plt.plot(self.plt_data['close'], label='Close Price',alpha=0.4)
	plt.legend(loc='upper left')

	plt.savefig(path+".png")
