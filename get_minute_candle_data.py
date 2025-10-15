# utils/get_minute_candle_data.py

import datetime
import pandas as pd
import os
import json
import time
import requests
import dotenv
from kis_api import KISApiHandler  # kis_api.py의 get_access_token 재사용

class MultiStockFetcher:
    def __init__(self, app_key, app_secret, acnt_no, is_mock=False):

        # 실전만 지원(API 문서: 모의투자 미지원)
        self.api = KISApiHandler(
            app_key=app_key,
            app_secret=app_secret,
            acnt_no=acnt_no,
            is_mock=is_mock  # ← 실전: False
        )

    def get_minute_data_one_day(self, stock_code: str, ymd: str):
        # 1) 토큰을 변수로 받되, 방어적으로 self.api.access_token 재확인
        token = self.api.get_access_token() or self.api.access_token

        url = f"{self.api.url_base}/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
        base_headers = {
            "authorization": f"Bearer {token}",
            "appkey": self.api.app_key,
            "appsecret": self.api.app_secret,
            "tr_id": "FHKST03010230",  # 국내주식-213 (실전)
            "Content-Type": "application/json",
            "accept": "application/json",
            "custtype": "P",
        }

        # 2) 보수적 → 점진적 재시도(파라미터 조합)
        variants = [
            # (1) 가장 기본: 00:00부터, 과거 미포함, FAKE 파라미터 보내지 않음
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code,
                "FID_INPUT_HOUR_1": "000000",
                "FID_INPUT_DATE_1": ymd,
                "FID_PW_DATA_INCU_YN": "N",
            },
            # (2) 과거 포함
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code,
                "FID_INPUT_HOUR_1": "000000",
                "FID_INPUT_DATE_1": ymd,
                "FID_PW_DATA_INCU_YN": "Y",
            },
            # (3) 시작 09:00
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code,
                "FID_INPUT_HOUR_1": "090000",
                "FID_INPUT_DATE_1": ymd,
                "FID_PW_DATA_INCU_YN": "N",
            },
            # (4) 허봉 파라미터를 "N"으로 명시
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code,
                "FID_INPUT_HOUR_1": "000000",
                "FID_INPUT_DATE_1": ymd,
                "FID_PW_DATA_INCU_YN": "N",
                "FID_FAKE_TICK_INCU_YN": "N",
            },
            # (5) 시장 통합으로 재시도(혹시 모를 케이스)
            {
                "FID_COND_MRKT_DIV_CODE": "UN",
                "FID_INPUT_ISCD": stock_code,
                "FID_INPUT_HOUR_1": "000000",
                "FID_INPUT_DATE_1": ymd,
                "FID_PW_DATA_INCU_YN": "N",
            },
        ]

        last_err = None
        for params in variants:
            try:
                r = requests.get(url, headers=base_headers, params=params, timeout=12)
                if r.status_code >= 500:
                    time.sleep(0.5)
                    continue
                r.raise_for_status()
                data = r.json()

                if data.get("rt_cd") != "0":
                    last_err = data.get("msg1", "rt_cd != 0")
                    continue

                rows = data.get("output2", [])
                if not rows:
                    last_err = "empty output2"
                    continue

                df = pd.DataFrame(rows)
                rename = {
                    "stck_bsop_date": "date",
                    "stck_cntg_hour": "time",
                    "stck_oprc": "open",
                    "stck_hgpr": "high",
                    "stck_lwpr": "low",
                    "stck_prpr": "close",
                    "cntg_vol": "volume",
                }
                df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                if {"date", "time"}.issubset(df.columns):
                    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y%m%d %H%M%S", errors="coerce")
                    df.drop_duplicates(subset=["date", "time"], inplace=True)
                    df.sort_values(["date", "time"], inplace=True)
                df.reset_index(drop=True, inplace=True)
                return df

            except Exception as e:
                last_err = str(e)
                time.sleep(0.35)

        print(f"[{stock_code}] {ymd} 실패: {last_err}")
        return pd.DataFrame()

    def fetch_and_save(self, stock_code: str, start_date: datetime.date, end_date: datetime.date, delay_sec: float = 0.2):
        """
        날짜 범위를 하루씩 조회해 합친 뒤
        ./data/{종목코드}/{종목코드}_{시작}_{종료}.csv 로 저장
        """
        all_list = []
        cur = start_date

        while cur <= end_date:
            ymd = cur.strftime("%Y%m%d")
            print(f"[{stock_code}] {ymd} 조회 중...")
            try:
                day_df = self.get_minute_data_one_day(stock_code, ymd)
                if not day_df.empty:
                    all_list.append(day_df)
            except Exception as e:
                print(f"[{stock_code}] {ymd} 에러: {e}")
            cur += datetime.timedelta(days=1)
            time.sleep(delay_sec)

        if not all_list:
            print(f"[{stock_code}] 저장할 데이터가 없습니다.")
            return

        df = pd.concat(all_list, ignore_index=True)

        save_dir = os.path.join("./data", 'RealTime')
        os.makedirs(save_dir, exist_ok=True)

        file_name = f"{stock_code}.csv"
        save_path = os.path.join(save_dir, file_name)

        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"✅ 저장 완료: {save_path}")


if __name__ == "__main__":
    # 예시
    dotenv.load_dotenv()
    start_date = datetime.date(2025, 10, 1)
    end_date   = datetime.date(2025, 10, 15)
    stock_codes = ['005930', '000660', '035720']

    fetcher = MultiStockFetcher(os.getenv("REAL_APP_KEY"), os.getenv("REAL_APP_SECRET"), os.getenv("REAL_ACNT_NO"), is_mock=False)  # 실전만 지원
    for code in stock_codes:
        fetcher.fetch_and_save(code, start_date, end_date)
