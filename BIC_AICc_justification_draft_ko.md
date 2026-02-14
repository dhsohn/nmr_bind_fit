# 화학 논문용 통계 정당화 초안

## 주제

화학 모델 비교에서 **BIC를 주기준(primary)**, **AICc를 보조기준(secondary)** 으로 사용하는 방법의 통계적 정당화

---

## 1) 핵심 주장(요약)

본 연구의 모델 선택 목적은 두 가지를 동시에 만족하는 것이다.

1. 화학적으로 해석 가능한 메커니즘/화학양론 모델을 과적합 없이 식별한다(식별 지향).
2. 유한 표본에서의 예측 성능 저하 위험을 점검한다(예측 지향).

이 목적을 분리하면, **BIC를 1차 의사결정 기준**으로 두고 **AICc를 보조 안전장치**로 병기하는 설계가 통계적으로 일관된다.

---

## 2) BIC를 1차 기준으로 두는 근거

### 2.1 이론적 배경

- BIC는

  $$\mathrm{BIC} = -2\ell(\hat\theta) + k\ln(n)$$

  로 정의되며, 표본 수가 증가할수록(\(n\uparrow\)) 복잡도 패널티(\(k\ln n\))가 강해진다.
- 정규성/식별가능성 등 표준 정칙성 가정과 "참모형이 후보군에 포함"된 상황에서, BIC 기반 선택은 **점근적 일관성(consistency)** 을 가진다.
- **참모형 포함 가정이 엄밀히 충족되지 않는 경우에도**, BIC의 \(k\ln n\) 패널티는 AIC의 \(2k\) 패널티보다 강하므로, 불필요한 매개변수 추가(과적합)를 억제하는 보수적 방어 기능은 유지된다. 즉, 일관성 보장이 아니더라도 "보수적 모형 선택 기준"으로서의 실무적 가치는 여전히 유효하다.
- 화학 논문의 메커니즘 비교(예: 1:1 vs 1:2 vs 2:1)처럼 "후보 중 어떤 구조가 더 타당한가"를 묻는 경우, BIC의 보수적 패널티는 과도한 매개변수 추가를 억제하는 장점이 있다.

### 2.1.1 우도와 RSS의 관계

- 본 연구에서 사용하는 잔차가 독립 정규분포(i.i.d. Gaussian)를 따른다고 가정하면, 로그-우도는 다음과 같이 RSS로 표현된다:

  $$-2\ell(\hat\theta) = n\ln\!\left(\frac{\mathrm{RSS}}{n}\right) + n\ln(2\pi) + n$$

- 따라서 동일 \(n\)의 모형 간 비교에서 상수항은 상쇄되며, **BIC·AICc 모두 RSS 기반으로 동치 계산이 가능**하다.
- 이분산(heteroscedastic) 또는 비정규 오차가 의심되는 경우에는 가중최소제곱(WLS) 등 적절한 우도를 별도로 구성해야 하며, 이 경우 RSS 동치 환산이 직접 적용되지 않음에 유의한다.

### 2.2 화학 맥락에서의 실무적 의미

- 실험 노이즈로 인한 미세 RSS 개선에 과민하게 반응하지 않아, "설명 가능한 단순 모형"을 우선한다.
- 메커니즘/화학양론 해석이 핵심인 원고(identification-heavy paper)에서 리뷰어 설득력이 높다.

---

## 3) AICc를 보조기준으로 병기하는 근거

### 3.1 이론적 배경

- AIC 계열은 Kullback-Leibler 위험(예상 정보손실)을 최소화하는 예측 지향 기준이다.
- AICc는

  $$\mathrm{AICc} = -2\ell(\hat\theta) + 2k + \frac{2k(k+1)}{n-k-1}$$

  로, 유한 표본에서 AIC의 하향 편향(복잡 모형 선호)을 보정한다.
- 특히 \(n/k\)가 크지 않은 데이터에서(화학 적정/분광 데이터에서 흔함) AICc 보고는 과적합 위험 점검에 유리하다.

### 3.2 BIC 보완 기능

- BIC가 지나치게 단순한 모형을 고를 가능성(underfitting)을 AICc가 보완한다.
- BIC 선택과 AICc 선택이 일치하면, 식별과 예측 관점 모두에서 강한 일관성 신호가 된다.
- 불일치 시에는 "식별 관점(BIC) vs 예측 관점(AICc)"의 긴장을 투명하게 보고할 수 있다.

---

## 4) 논문에 바로 넣을 수 있는 의사결정 규칙(권장)

아래 규칙을 Methods에 사전 명시하면, "사후적 기준 바꾸기"(criterion shopping) 우려를 줄일 수 있다.

1. 동일 데이터, 동일 오차모형(우도), 동일 전처리에서 모든 후보모형을 적합한다.
2. **최소 BIC 모형을 잠정 작업모형(provisional working model)** 으로 채택한다.
3. AICc를 함께 보고하여 유한 표본 예측 위험을 점검한다.
4. \(\Delta\mathrm{BIC}<2\)이면 모델 구분이 약하다고 명시하고 결론을 약화한다(예: "잠정적").
5. BIC-AICc 불일치 시 아래 의사결정표를 따른다.

| ΔBIC (선택 모형 ↔ 차선 모형) | AICc 지지 일치 여부 | 판정 |
|:---:|:---:|:---|
| < 2 | 일치 | 잠정적: 모형이 가장 유력하나 구분이 약함을 명시 |
| < 2 | **불일치** | 결정 보류: 복수 후보로 보고, 추가 데이터/보조 진단 권장 |
| 2 – 6 | 일치 | 채택: BIC·AICc 모두 동일 모형 지지, 식별·예측 관점 일관 |
| 2 – 6 | **불일치** | 조건부 채택: BIC 모형을 주결론으로 하되, AICc 관점 예측 민감도 분석을 보조자료에 제시 |
| ≥ 6 | 일치 | 강한 채택: 두 기준 모두 강한 지지 |
| ≥ 6 | **불일치** | BIC 모형 우선: 식별 관점 강한 증거. 단, AICc 결과를 보고하여 투명성 확보 |
6. 통계 기준값이 낮더라도 화학적으로 불합리한 파라미터/메커니즘이면 배제 근거를 명시한다.

### 4.1 임계값 해석(권장 표준)

- BIC 해석(자주 쓰이는 실무 기준):
  - \(\Delta\mathrm{BIC}<2\): 구분 약함
  - \(2\le\Delta\mathrm{BIC}<6\): 양의 증거
  - \(6\le\Delta\mathrm{BIC}<10\): 강한 증거
  - \(\Delta\mathrm{BIC}\ge10\): 매우 강한 증거
- AICc 해석(예측 관점):
  - \(\Delta\mathrm{AICc}<2\): 실질적 지지
  - \(4\le\Delta\mathrm{AICc}<7\): 지지 약화
  - \(\Delta\mathrm{AICc}\ge10\): 사실상 지지 없음
- 표본이 작은 경우(경험칙: \(n/k<40\))는 AIC보다 AICc 보고를 우선한다.


---

## 5) Methods 문단 예시(초안)

Model comparison was performed using the Bayesian Information Criterion (BIC) as the primary ranking index and the corrected Akaike Information Criterion (AICc) as a secondary diagnostic. BIC was used for primary selection because our main inferential goal was mechanistic identification with parsimonious model complexity control. AICc was reported to assess finite-sample predictive risk and to reduce small-sample bias of AIC when the sample size was limited relative to the number of estimated parameters. The effective sample size \(n\) was defined as the total number of **finite residual scalars** used in likelihood evaluation across all datasets, titration points, and ppm peaks (i.e., \(\sum \mathbf{1}\{\text{finite residual}\}\)); missing observations were excluded from \(n\). Therefore, the model with the lowest BIC was treated as a provisional working model among tested candidates, while agreement/disagreement with AICc was used to grade the robustness of model discrimination.

국문 버전(필요 시):

후보 모형 비교는 BIC를 1차 순위지표로, AICc를 2차 점검지표로 수행하였다. 본 연구의 1차 추론 목표가 화학 메커니즘 식별이므로, 복잡도 패널티가 더 강한 BIC를 주기준으로 채택하였다. 동시에 표본 수가 제한된 조건에서의 예측 위험과 과적합 가능성을 점검하기 위해 AICc를 병기하였다. 유효 표본 수 \(n\)은 우도 계산에 실제로 사용된 **유한 잔차 스칼라의 총개수**(모든 데이터셋·적정점·ppm 피크에 대해 \(\sum \mathbf{1}\{\text{finite residual}\}\))로 정의하였고, 결측으로 제외된 값은 \(n\)에서 제외하였다. 따라서 최소 BIC 모형을 "후보군 내 잠정 작업모형"으로 정의하고, AICc 일치 여부를 모델 판별의 강건성 판단에 활용하였다.

---

## 6) 결과 보고 템플릿(권장)

표에는 최소한 아래 열을 포함한다.

- Model name
- \(k\) (추정 파라미터 수; 분산항 포함 여부를 각주로 명시)
- \(n\) (유효 표본 수; **모든 유한 잔차 스칼라 수**로 정의, 결측 제외 규칙을 각주로 명시)
- log-likelihood (또는 RSS 기반 동치식 명시)
- BIC, \(\Delta\)BIC
- AICc, \(\Delta\)AICc
- 판정(Primary by BIC / Secondary by AICc)
- 화학적 타당성 점검 결과(Yes/No + 한 줄 근거)

문장 보고 예시:

- "최소 BIC 기준으로 1:1 모형이 잠정 작업모형으로 선택되었으며(\(\Delta\)BIC to next = 3.1), AICc도 동일 모형을 지지하였다."
- "BIC와 AICc가 상이하였고 \(\Delta\)BIC가 1.4로 작아, 모델 판별은 잠정적이며 추가 실험/보조진단이 필요하다."

---

## 7) 리뷰어 코멘트 대응용 한계 명시 문구

다음 문구를 Discussion 또는 Limitations에 포함하면 방어력이 높아진다.

- 정보기준은 후보군 내부의 상대 비교이며, 후보군 밖 "진짜 모형"을 보장하지 않는다.
- BIC의 일관성은 정칙성 가정 및 후보군 적절성에 의존한다.
- AICc의 우수성은 예측 관점의 위험 최소화에 대한 것이며, 메커니즘의 진리성 증명과 동일하지 않다.
- 따라서 통계 기준은 잔차 구조, 파라미터 물리성, 독립적 화학 근거와 함께 해석해야 한다.

---

## 8) 참고문헌(핵심, 검증 쉬운 고전 위주)

1. Akaike H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*.
2. Schwarz G. (1978). Estimating the dimension of a model. *The Annals of Statistics*.
3. Sugiura N. (1978). Further analysis of the data by Akaike's information criterion and the finite corrections. *Communications in Statistics - Theory and Methods*.
4. Hurvich CM, Tsai CL. (1989). Regression and time series model selection in small samples. *Biometrika*.
5. Burnham KP, Anderson DR. (2002). *Model Selection and Multimodel Inference* (2nd ed.). Springer.
6. Claeskens G, Hjort NL. (2008). *Model Selection and Model Averaging*. Cambridge University Press.

---

## 9) 한 줄 결론

"BIC-주기준, AICc-보조기준"은 **식별(메커니즘 타당성)과 예측(유한 표본 강건성)** 을 분리해 동시에 관리하는 전략으로, 화학 모델 비교 문맥에서 충분히 합리화 가능한 설계다.
