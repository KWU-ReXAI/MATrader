import os
import re
import glob
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from parameters import parameters

def phase2quarter(s2fe):
	# 각 페이즈의 분기별 선택된 주식, 거래일 및 훈련일 반환
	folder_path = os.path.join(parameters.BASE_DIR,
							   f'data/S2FE/{s2fe}')
	csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
	results = []
	pattern = re.compile(r"test_selected_stocks_p(\d+)_(\d+)\.csv$")
	for csv_file in csv_files:
		try:
			file_name = os.path.basename(csv_file)
			match = pattern.search(file_name)
			df = pd.read_csv(csv_file, header=0)
			for _, row in df.iterrows():
				quarter = row.iloc[1]
				test_start = row.iloc[2]
				test_end = row.iloc[3]
				test_start_dt = datetime.strptime(test_start, '%Y-%m-%d')
				train_start = (test_start_dt - relativedelta(years=3)).strftime('%Y-%m-%d')
				train_end = (test_start_dt - timedelta(days=1)).strftime('%Y-%m-%d')

				stocks = row.iloc[4:].tolist()
				stocks = [str(int(stock)).zfill(6) for stock in stocks if pd.notna(stock)]

				result_dict = {'phase': match.group(1), 'testNum': match.group(2),
							   'quarter': quarter, 'train_start': train_start,
							   'train_end': train_end, 'test_start': test_start,
							   'test_end': test_end, 'stock_codes': stocks}
				results.append(result_dict)

		except FileNotFoundError:
			print(f"오류: '{csv_file}' 파일을 찾을 수 없습니다.")
	df = pd.DataFrame(results)
	return df

if __name__ == '__main__':
	phase2quarter('result_1')
