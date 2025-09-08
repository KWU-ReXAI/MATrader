def phase2date(phase):
	# 사실 페이즈 4의 4개 분기에 대한 것
	# 분기별로 종목이 바뀌기 때문
	# 짧은 기간에도 성능이 나오는 지를 보기 위함이니, 추후 수정 요망
	if phase == 1:
		train_start = '2020-03-23'; test_start = '2024-03-23'
		train_end = '2024-03-22'; test_end = '2024-05-17'
	elif phase == 2:
		train_start = '2020-05-18'; test_start = '2024-05-18'
		train_end = '2024-05-17'; test_end = '2024-08-15'
	elif phase == 3:
		train_start = '2020-08-16'; test_start = '2024-08-16'
		train_end = '2024-08-15'; test_end = '2024-11-15'
	elif phase == 4:
		train_start = '2020-11-16'; test_start = '2024-11-16'
		train_end = '2024-11-15'; test_end = '2025-03-21'
	else:
		train_start = None; test_start = None
		train_end = None; test_end = None
	start = (train_start, test_start)
	end = (train_end, test_end)
	return start, end
