# utils/get_minute_candle_data.py

import datetime
import pandas as pd
import os, sys
import json
import time
import requests
import dotenv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kis_api import KISApiHandler

MARKET_OPEN = "090000"
MARKET_CLOSE = "153000"

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
        token = self.api.get_access_token() or self.api.access_token

        url = f"{self.api.url_base}/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
        headers = {
            "authorization": f"Bearer {token}",
            "appkey": self.api.app_key,
            "appsecret": self.api.app_secret,
            "tr_id": "FHKST03010230",     # 실전
            "content-type": "application/json",
            "accept": "application/json",
            "custtype": "P",
        }

        # 앵커 시각을 여러 개로 나눠 전체 시간대를 커버
        anchors = ["105900", "125900", "145900", "153000"]

        dfs = []
        last_err = None

        for hhmmss in anchors:
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code,
                "FID_INPUT_HOUR_1": hhmmss,
                "FID_INPUT_DATE_1": ymd,
                "FID_PW_DATA_INCU_YN": "Y",
                "FID_FAKE_TICK_INCU_YN": " ",
            }
            try:
                r = requests.get(url, headers=headers, params=params, timeout=12)
                if r.status_code >= 500:
                    time.sleep(0.3)
                    continue
                r.raise_for_status()
                data = r.json()

                if data.get("rt_cd") != "0":
                    last_err = data.get("msg1", "rt_cd != 0")
                    continue

                rows = data.get("output2", []) or []
                if not rows:
                    last_err = "empty output2"
                    continue

                df = pd.DataFrame(rows)
                if df.empty:
                    continue

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
                dfs.append(df)

            except Exception as e:
                last_err = str(e)
                time.sleep(0.3)
                continue

        if not dfs:
            print(f"[{stock_code}] {ymd} 실패: {last_err}")
            return pd.DataFrame()

        # ---- 누적 병합 + 정리 ----
        out = pd.concat(dfs, ignore_index=True)

        # 당일만 남기기
        if "stck_bsop_date" in out.columns:
            out = out[out["stck_bsop_date"] == ymd]
        elif "date" in out.columns:
            out = out[out["date"] == ymd]

        # 표준 컬럼명 통일(다시 한 번 보정)
        rename = {
            "stck_bsop_date": "date",
            "stck_cntg_hour": "time",
            "stck_oprc": "open",
            "stck_hgpr": "high",
            "stck_lwpr": "low",
            "stck_prpr": "close",
            "cntg_vol": "volume",
        }
        out.rename(columns={k: v for k, v in rename.items() if k in out.columns}, inplace=True)

        # 숫자형 변환
        for col in ["open", "high", "low", "close", "volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # 시간 문자열 정규화 및 장시간대 필터
        if "time" in out.columns:
            out["time"] = out["time"].astype(str).str.zfill(6)
            out = out[(out["time"] >= MARKET_OPEN) & (out["time"] <= MARKET_CLOSE)]

        # 중복 제거 + 정렬
        if {"date", "time"}.issubset(out.columns):
            out.drop_duplicates(subset=["date", "time"], inplace=True)
            out.sort_values(["date", "time"], inplace=True)
            out["datetime"] = pd.to_datetime(out["date"] + " " + out["time"], format="%Y%m%d %H%M%S", errors="coerce")
        out.reset_index(drop=True, inplace=True)

        return out


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
