# Notes

## TODO

[x] 11/06 Dataloader 작성
[x] 11/08 `src/training/train_utils.py`의 데이터 로더가 `src/data/image_datasets.py`의 출력 형식과 맞지 않는 문제 수정
[ ] Image concat 방식 변경
[x] 11/13 OCTA Dataset에 대해 학습
[ ] BerDiff에 prev. output의 thickness/SAUNA 대입
[x] 11/24 BerDiff의 Bernoulli Sampling -> Poisson Sampling
[x] Bernoulli -> Gaussian
[ ] Betti error 확인

## Metrics

### 1. 중첩 기반 지표 (Overlap-based Metrics)

가장 일반적으로 사용되는 지표로, 픽셀 단위의 일치 여부를 측정.

| 지표명 | 공식 | 설명 |
| :--- | :---: | :--- |
| **Precision** | $Precision = \frac{TP}{TP + FP}$ | 양성 예측 중 실제 양성 비율 |
| **Recall** | $Recall = \frac{TP}{TP + FN}$ | 실제 양성 중 예측 성공 비율 |
| **F1 Score** | $F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$ | 정밀도와 재현율의 조화 평균 |
| **IoU** | $IoU = \frac{\|A \cap B\|}{\|A \cup B\|}$ | 예측과 정답의 교집합/합집합 비율 |

---

### 2. 경계 기반 지표 (Boundary-based Metrics)

객체의 윤곽선(Boundary) 정밀도를 평가.

#### 🔍 경계 추출 (Boundary Extraction)

$erosion(mask)$은 $3 \times 3$ `ones` 커널로 AND 연산을 수행하여 얻으며, 최종 경계는 원본과 침식 결과의 XOR 연산으로 추출.

> **Boundary** = $mask \oplus erosion(mask)$

#### 📏 평가 지표

* **Boundary AP (Average Precision)**: 여러 허용 오차($tol$)에서의 경계 정밀도 평균.
  $$\text{Boundary AP} = \frac{1}{N} \sum_{tol=0}^{N} \frac{\text{경계 매칭 수}}{\text{예측 경계 픽셀 수}}$$
* **Boundary Accuracy**: 정답 경계 중 예측 경계와 인접한 픽셀의 비율(재현율).
  $$\text{Boundary Acc} = \frac{\text{정답 경계 중 예측과 가까운 픽셀 수}}{\text{정답 경계 픽셀 수}}$$
* **HD95 (Hausdorff Distance 95)**: 경계점 간 거리의 95번째 백분위수.
  $$\text{HD}_{95} = \text{95th percentile of } \{d(x, Y), d(y, X)\}$$

---

### 3. 위상 기반 지표 (Topological Metrics)

객체의 연결성이나 구조적 형태가 정답과 얼마나 유사한지 평가

* **Betti-0 Error**: 연결된 컴포넌트(객체) 개수의 차이
  $$\text{Betti Error}_0 = | \beta_0^{pred} - \beta_0^{gt} |$$
* **Betti-1 Error**: 구멍(Hole/Loop) 개수의 차이
  $$\text{Betti Error}_1 = | \beta_1^{pred} - \beta_1^{gt} |$$

---

## packages

```bash
uv pip install autorootcwd blobfile torch torchvision numpy \
    pillow mpi4py mpich \
    natsort matplotlib tqdm \
    pandas scikit-image
```
