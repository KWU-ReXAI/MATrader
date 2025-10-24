import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
from phase import phase2quarter
import matplotlib.dates as mdates
from matplotlib.dates import date2num, MonthLocator


PHASE = 4

if __name__ == '__main__':
	base = os.path.dirname(os.path.abspath(__file__))

	# 후에 ROK 내의 랜덤한 주식을 가져 오는 식으로 변경 가능함
	# 현재는 인자로 들어온 주식만 저장
	stock_list = sys.argv[1:]

	for stock in stock_list:
		fpath = os.path.join(base, 'data/{}/{}.csv'.format('ROK', stock))

		data = pd.read_csv(fpath, thousands=',', converters={'date': lambda x: str(x)})
		data.columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
		data['date'] = data['date'][:].str.replace('-', '')
		data.drop(0, axis=0)
		data = data.dropna()
		data['date'] = pd.to_datetime(data['date'])
		data['date_float'] = date2num(data['date'])

		fig, axs = plt.subplots(2, 2, figsize=(20, 12))  # 2행 2열
		plt.style.use('fivethirtyeight')
		axs = axs.flatten()

		colors = ['lightcoral', 'lightskyblue']
		bound = [phase2date(x) for x in range(1,PHASE+1)]
		for i, (start, end) in enumerate(bound):
			ax = axs[i]
			train_start, test_start = start
			train_end, test_end = end
			day = [(train_start, train_end), (test_start, test_end)]

			df = data[(data['date'] >= train_start) & (data['date'] <= test_end)].copy()
			df = df.dropna()

			for j, (p_start, p_end) in enumerate(day):
				s = date2num(pd.to_datetime(p_start))
				e = date2num(pd.to_datetime(p_end))
				ax.axvspan(s, e, color=colors[j % len(colors)], alpha=0.3)

			ax.plot(df['date_float'], df['close'], label=f'Close Price ({train_start}~{test_end})', alpha=0.5)

			ax.set_title(f"phase {i+1}")
			ax.xaxis.set_major_locator(MonthLocator(bymonth=(1, 7)))
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
			ax.tick_params(axis='x', rotation=30)
			ax.legend(loc='upper left')

		plt.suptitle("{} Phases(red: train, blue: test)".format(stock), fontsize=30)
		plt.tight_layout()
		path = os.path.join(base, stock)
		plt.savefig(path + ".png")