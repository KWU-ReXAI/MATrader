# MATrader(Multi-Agent Transformer Trader) 🚀

## 실행 방법

### 훈련 (Training)

```bash
python main.py --stock_dir [S2FE 결과 폴더 이름] --output_name [결과 폴더 이름]
```

-   **`--stock_dir`**: `data/S2FE/` 경로에 저장된 S2FE 결과 폴더의 이름을 지정합니다.
-   **`--output_name`**: `output` 경로에 결과를 저장할 폴더의 이름을 지정합니다.
    - `default`: 현재 시각(`%Y%m%d_%H%M%S`)

> **참고**: 세부 하이퍼파라미터는 `main.py` 파일을 참조하세요.

> **참고**: 종목 리스트는 'data/S2FE/[S2FE 결과 폴더 이름]'에 저장하세요.


### 테스트 (Testing)

```bash
python main.py --stock_dir [S2FE 결과 폴더 이름] --output_name [결과 폴더 이름] --test --model_dir [훈련 폴더명]
```

-   **`--stock_dir`**: 훈련에 사용했던 S2FE 결과 폴더의 이름을 동일하게 지정합니다.
-   **`--output_name`**: `output` 경로에 결과를 저장할 폴더의 이름을 지정합니다.
    - `default`: 현재 시각(`%Y%m%d_%H%M%S`)
-   **`--test`**: 모델을 테스트 모드로 실행합니다.
-   **`--model_dir`**: 테스트할 모델(`*.pt`)이 저장된 훈련 폴더의 이름을 지정합니다. 훈련 시 자동으로 생성된 폴더명을 사용하면 됩니다.

## 라이선스 안내
본 프로젝트는 아래 오픈소스 프로젝트의 **Multi-Agent Transformer (MAT)** 알고리즘 코드를 참고 및 일부 수정하여 사용하였습니다.

- [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy/tree/main/onpolicy/algorithms/mat)

원본 코드는 [MIT 라이선스](https://github.com/marlbenchmark/on-policy/blob/main/LICENSE)를 따르며,  
본 프로젝트 또한 해당 라이선스 조건을 준수합니다.

