def phase2date(phase):
	if phase == 1:
		train_start = '20160101'; test_start = '20200701'
		train_end = '20200630'; test_end = '20210630'
	elif phase == 2:
		train_start = '20170101'; test_start = '20210701'
		train_end = '20210630'; test_end = '20220630'
	elif phase == 3:
		train_start = '20180101'; test_start = '20220701'
		train_end = '20220630'; test_end = '20230630'
	elif phase == 4:
		train_start = '20190101'; test_start = '20230701'
		train_end = '20230630'; test_end = '20240630'
	else:
		train_start = None; test_start = None
		train_end = None; test_end = None
	start = (train_start, test_start)
	end = (train_end, test_end)
	return start, end
