import requests
import json
import time
from datetime import datetime, timedelta
import logging
import pandas as pd

class KISApiHandler:
    def __init__(self, app_key, app_secret, acnt_no, is_mock=False):
        self.url_base = (
            'https://openapivts.koreainvestment.com:29443' if is_mock
            else 'https://openapi.koreainvestment.com:9443'
        )
        self.app_key, self.app_secret, self.acnt_no = app_key, app_secret, acnt_no
        self.access_token, self.token_expire_time = "", None
        self.headers = {"content-type": "application/json"}
        print(f"[KIS] ENV={'MOCK' if is_mock else 'REAL'} URL={self.url_base}")  # ← 확인 로그

    def get_access_token(self):
        if self.access_token and self.token_expire_time and datetime.now() < self.token_expire_time:
            return self.access_token  # ← 반드시 반환
        logging.info("인증 토큰 발급 요청")
        body = {"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret}
        res = requests.post(f'{self.url_base}/oauth2/tokenP', headers=self.headers, data=json.dumps(body))
        if res.status_code != 200:
            logging.error(f"토큰 발급 실패: Status {res.status_code}, Response: {res.text}")
            raise Exception("토큰 발급 실패")
        token_data = res.json()
        self.access_token = token_data['access_token']
        self.token_expire_time = datetime.now() + timedelta(seconds=token_data.get('expires_in', 86400) - 300)
        return self.access_token  # ← 반드시 반환

    def get_hash_key(self, data):
        path = "/uapi/hashkey"
        url = f"{self.url_base}{path}"
        headers = {'content-type': 'application/json', 'appkey': self.app_key, 'appsecret': self.app_secret}
        res = requests.post(url, headers=headers, data=json.dumps(data))
        if res.status_code == 200 and "HASH" in res.json():
            return res.json()["HASH"]
        else:
            logging.error(f"Hashkey 발급 실패: Status {res.status_code}, Response: {res.text}")
            raise Exception("Hashkey 발급 실패")

    def api_request(self, method, path, tr_id, params=None, data=None):
        time.sleep(0.5)
        try:
            self.get_access_token()
        except Exception as e:
            logging.error(f"API 인증 실패: {e}"); return None
        url = f"{self.url_base}{path}"
        headers = self.headers.copy()
        headers.update({'authorization': f'Bearer {self.access_token}', 'appKey': self.app_key, 'appSecret': self.app_secret, 'tr_id': tr_id, 'custtype': 'P'})
        if method.lower() == 'post' and data:
            try:
                headers['hashkey'] = self.get_hash_key(data)
                res = requests.post(url, headers=headers, data=json.dumps(data))
            except Exception as e:
                logging.error(f"Hashkey 또는 POST 요청 실패: {e}"); return None
        else:
            res = requests.get(url, headers=headers, params=params)
        if res.status_code != 200:
            logging.error(f"API 요청 실패: {res.status_code}, {res.text}"); return None
        res_json = res.json()
        if res_json.get('rt_cd') != '0':
            logging.error(f"API 응답 오류: {res_json.get('msg1', '알 수 없는 오류')}")
        return res_json

    def get_current_price(self, iscd):
        path = '/uapi/domestic-stock/v1/quotations/inquire-price'
        params = {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': iscd}
        res = self.api_request('get', path, 'FHKST01010100', params=params)
        return int(res['output']['stck_prpr']) if res and res.get('rt_cd') == '0' else None
    
    ### 분봉
    def _prev_hhmmss(self, hhmmss: str) -> str:
        t = datetime.strptime(hhmmss, "%H%M%S")
        t -= timedelta(seconds=1)
        return t.strftime("%H%M%S")

    def get_minute_candles(self, code: str, base_time: str | None = None) -> pd.DataFrame:
        self.get_access_token()

        if base_time is None:
            base_time = datetime.now().strftime("%H%M%S")

        url = f"{self.url_base}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        headers = {
            "content-type": "application/json; charset=UTF-8",
            "accept": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHKST03010200",
            "custtype": "P",
        }
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_HOUR_1": base_time,   # 기준시각 '이전'부터 반환
            "FID_PW_DATA_INCU_YN": "N",
        }

        # 500 방어: 짧은 백오프 재시도
        last_exc = None
        for i in range(3):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=10)
                if r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code} {r.reason}")
                r.raise_for_status()
                data = r.json()
                if data.get("rt_cd") != "0":
                    raise RuntimeError(data.get("msg1"))
                rows = data.get("output2", [])
                df = pd.DataFrame(rows)
                if df.empty:
                    return df
                rename = {
                    "stck_bsop_date": "date",
                    "stck_cntg_hour": "time",
                    "stck_oprc": "open",
                    "stck_hgpr": "high",
                    "stck_lwpr": "low",
                    "stck_prpr": "close",
                    "cntg_vol": "volume",
                }
                df.rename(columns={k:v for k,v in rename.items() if k in df.columns}, inplace=True)
                if "date" in df.columns and "time" in df.columns:
                    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y%m%d %H%M%S", errors="coerce")
                df.sort_values("time", inplace=True)
                df.reset_index(drop=True, inplace=True)
                return df
            except Exception as e:
                last_exc = e
                time.sleep(0.35 * (i + 1))  # 0.35s, 0.7s, 1.05s
        raise last_exc

    def get_minute_candles_all_today(self, code: str, sleep_sec: float = 0.25, max_pages: int = 80) -> pd.DataFrame:
        all_df = pd.DataFrame()
        base_time = datetime.now().strftime("%H%M%S")

        for _ in range(max_pages):
            df = self.get_minute_candles(code, base_time=base_time)
            if df.empty:
                break
            all_df = pd.concat([all_df, df], ignore_index=True)

            # 다음 페이지 기준시각: 이번 페이지 '가장 오래된 시각 - 1초'
            first_time = str(df.iloc[0]["time"])
            next_time = self._prev_hhmmss(first_time)
            # 더 내려갈 수 없거나 30건 미만이면 종료
            if next_time == base_time or len(df) < 30:
                break
            base_time = next_time
            time.sleep(sleep_sec)  # 호출 간격

        if not all_df.empty:
            all_df.drop_duplicates(subset=["date", "time"], inplace=True)
            all_df.sort_values(["date", "time"], inplace=True)
            all_df.reset_index(drop=True, inplace=True)
        return all_df

    ###

    def get_period_data(self, iscd):
        dt_now = datetime.now()
        dt_end = dt_now.strftime('%Y%m%d')
        dt_start = (dt_now - timedelta(days=100)).strftime('%Y%m%d')
        path = '/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice'
        params = {
            'FID_COND_MRKT_DIV_CODE': 'J',
            'FID_INPUT_ISCD': iscd,
            'FID_INPUT_DATE_1': dt_start,
            'FID_INPUT_DATE_2': dt_end,
            'FID_PERIOD_DIV_CODE': 'D',
            'FID_ORG_ADJ_PRC': '1'
        }
        res = self.api_request('get', path, 'FHKST03010100', params=params)
        if res and res.get('rt_cd') == '0' and 'output2' in res:
            return res['output2']
        else:
            logging.error(f"[{iscd}] 차트 데이터 API 응답 오류")
            return []

    def get_account_balance(self):
        path = '/uapi/domestic-stock/v1/trading/inquire-balance'
        tr_id = 'VTTC8434R'
        params = {
            "CANO": self.acnt_no.split('-')[0],
            "ACNT_PRDT_CD": self.acnt_no.split('-')[1],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "N",
            "INQR_DVSN": "01",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        res = self.api_request('get', path, tr_id, params=params)
        if res and res.get('rt_cd') == '0':
            try:
                summary = res.get('output2', [{}])[0]
                holdings = res.get('output1', [])

                balance_data = {
                    'deposit': int(summary.get('dnca_tot_amt', 0)),
                    'total_eval': int(summary.get('tot_evlu_amt', 0)),
                    'holdings': []
                }

                for item in holdings:
                    balance_data['holdings'].append({
                        'iscd': item.get('pdno', ''),
                        'name': item.get('prdt_name', ''),
                        'qty': int(item.get('hldg_qty', 0)),
                        'buy_price': float(item.get('pchs_avg_pric', 0)),
                        'cur_price': int(item.get('prpr', 0)),
                        'eval_price': int(item.get('evlu_amt', 0)),
                        'profit': int(item.get('evlu_pfls_amt', 0)),
                        'profit_rate': float(item.get('evlu_pfls_rt', 0))
                    })

                return balance_data
            except Exception as e:
                logging.error(f"잔고 데이터 파싱 오류: {e}, 응답: {res}")
                return None
        else:
            logging.error("계좌 잔고 조회 실패")
            return None

    def order_cash(self, iscd, qty, price, buy_sell_cd, order_type='01'):
        tr_id = 'VTTC0802U' if buy_sell_cd == '02' else 'VTTC0801U'
        data = {
            'CANO': self.acnt_no.split('-')[0],
            'ACNT_PRDT_CD': self.acnt_no.split('-')[1],
            'PDNO': iscd,
            'ORD_DVSN': order_type,
            'ORD_QTY': str(qty),
            'ORD_UNPR': str(price) if order_type == '00' else '0'
        }
        return self.api_request('post', '/uapi/domestic-stock/v1/trading/order-cash', tr_id, data=data)

