# Three-Dimensional Electric Field Reconstruction Based on Boundary Potentials

경계 Dirichlet 전위와 조화 확장을 이용한 3차원 전기장 재현 연구의 시뮬레이션 코드이다.
---

## 저장소 구조

```
01_numerical_analysis.ipynb          # 3D 수치 시뮬레이션 노트북
02_conductive_paper_analysis.ipynb   # 2D PASCO 도전성 종이 실험 분석 노트북
closure_analysis.py                  # 강건 유효부피·패치·양자화 최종 검증 스크립트
generate_field/
  potential_boundary_reconstruction.py  # 핵심 재사용 모듈
  __init__.py
data/
  conductive_paper_measurements_remeasured.csv  # 2D 실측 데이터
results/
  closure_analysis/
    closure_analysis.json             # closure_analysis.py 기준 실행 결과
requirements.txt
```

---

## 연구 구성 (4-layer)

| 계층 | 파일 | 역할 |
|------|------|------|
| 1 | `01_numerical_analysis.ipynb` | 3D 목표 조화장 생성, 표면전하 역산, 패치 해상도, 수치 재현 기본 분석 |
| 2 | `generate_field/potential_boundary_reconstruction.py` | `HarmonicWorldConfig`, `HarmonicFieldModel`, 조화장 생성, Fourier 항, superellipsoid 후보, 강건 유효부피 계산 — 논문 전체가 공유하는 핵심 모듈 |
| 3 | `closure_analysis.py` | 선택 영역, 강건 유효부피, 패치 요구 조건, 전압 양자화, 앙상블 검증 — 논문 후반 수치를 닫는 최종 스크립트 |
| 4 | `02_conductive_paper_analysis.ipynb` | PASCO 도전성 종이 재측정 CSV를 이용한 2D 실험 분석 |

---

## 시뮬레이션 파라미터 (논문 기준)

### 3D 목표장 (공통 설정)

| 파라미터 | 값 |
|----------|-----|
| 도메인 크기 | Lx = 1.30 m, Ly = 0.85 m, Lz = 1.10 m |
| 격자 | 65 × 51 × 57 |
| 중심 | (0.65, 0.425, 0.55) m |
| 기준 스케일 | E₀ = 1.0 |
| 난수 시드 | 12 |
| 다항 스케일 | 0.18 |
| Fourier 스케일 | 0.12 |
| Fourier 항 수 | 5 |

### 고정 다항 계수 (closure_analysis.py / POLY_COEFFS)

| 항 | 계수 |
|----|------|
| u | 0.55 |
| v | −0.30 |
| w | 0.20 |
| uv | 0.08 |
| uw | −0.04 |
| vw | 0.06 |
| u²−v² | 0.10 |
| 2w²−u²−v² | −0.08 |
| u(4w²−u²−v²) | 0.03 |
| v(4w²−u²−v²) | −0.02 |
| w(u²−v²) | 0.05 |
| uvw | 0.04 |
| u(u²−3v²) | −0.03 |
| v(3u²−v²) | 0.02 |
| w(2w²−3u²−3v²) | 0.01 |

### 고정 Fourier 항 (closure_analysis.py / FOURIER_TERMS)

| # | amplitude | α | β | phase_u | phase_v | z_parity |
|---|-----------|---|---|---------|---------|----------|
| 0 | 0.10 | 2.0 | 1.0 | 0.0 | 0.7 | even |
| 1 | −0.07 | 3.0 | 2.0 | 0.8 | 1.6 | odd |
| 2 | 0.05 | 1.0 | 4.0 | 1.2 | 0.2 | even |

---

## 주요 결과 (`results/closure_analysis/closure_analysis.json`)

모든 수치는 `closure_analysis.py` 를 실행하여 생성된 참조 출력 기준입니다 (`TAU_MAIN = 2×10⁻³`).

### 선택 형상 (candidate 211, 원본 격자 25×21×23)

| 지표 | 값 |
|------|----|
| volume_fraction | 0.40 |
| η_τ (eta_tau) | 0.3973 |
| ε_E95 (epsilon_E95) | 1.587 × 10⁻³ |
| C_ω | 2.494 |
| α₁ / α₂ | 1.25 / 1.15 |
| p (superellipsoid) | 4.0 |
| rotation_z | π/6 rad |
| 후보 박스 (Lx/Ly/Lz) | 1.061 / 0.638 / 0.718 m |

### 고해상도 exact-boundary 검증 (격자 53×39×38)

| 지표 | 값 |
|------|----|
| ε_E95 | **5.82 × 10⁻⁵** |
| E_relative_l2 | **3.53 × 10⁻⁵** |
| local_pass_fraction | 1.0 |
| mean_cosine_similarity | 0.99999999957 |

### τ 민감도 분석

| τ_E,local | 실현 가능 후보 수 | 선택 volume_fraction | 선택 η_τ | 선택 ε_E95 |
|-----------|-------------------|----------------------|-----------|------------|
| 1×10⁻³ | 0 (fallback) | 0.28 | 0.222 | 1.28 × 10⁻³ |
| **2×10⁻³** | **62** | **0.40** | **0.396** | **1.72 × 10⁻³** |
| 5×10⁻³ | 211 | 0.75 | 0.749 | 3.43 × 10⁻³ |

### 패치 해상도 요구 조건 (고해상도 형상 기준)

| 패치 (y×z) | 패치 값 수 | E_relative_l2 | ε_E95 |
|-------------|------------|----------------|-------|
| 12×12 | 864 | 0.664 | 0.793 |
| 16×16 | 1 536 | 0.474 | 0.636 |
| 20×20 | 2 400 | 0.318 | 0.574 |
| 24×24 | 3 456 | 0.222 | 0.541 |
| 28×28 | 4 704 | 0.146 | 0.277 |
| 32×32 | 6 144 | 0.176 | 0.293 |
| 36×36 | 7 776 | 0.195 | 0.288 |

> ε_E95 < τ = 2×10⁻³ 를 만족하는 패치 해상도는 이 형상에서 도달하기 어려우며, 이는 고해상도 exact-boundary 결과(ε_E95 = 5.82×10⁻⁵)와의 차이가 패치화에 기인함을 보인다.

### 경계 잡음 이득 (0.1% RMS 잡음, n_trials = 6)

| 지표 | 값 |
|------|----|
| 기준선 ε_E95 | 7.03 × 10⁻⁵ |
| 잡음 ε_E95 (평균) | 5.93 × 10⁻² |
| **G_noise,E95** | **59.2** |
| 기준선 E_relative_l2 | 4.09 × 10⁻⁵ |
| 잡음 E_relative_l2 (평균) | 4.97 × 10⁻² |
| G_noise,E_l2 | 49.7 |

### 전압 양자화 민감도

| 양자화 방식 | 전압 간격 (V) | ε_E95 | E_relative_l2 | local_pass_fraction |
|-------------|--------------|-------|----------------|----------------------|
| 8-bit | 3.75 × 10⁻³ | 0.0297 | 0.0132 | 0.739 |
| 10-bit | 9.35 × 10⁻⁴ | 7.57 × 10⁻³ | 3.32 × 10⁻³ | 0.861 |
| **12-bit** | **2.34 × 10⁻⁴** | **1.96 × 10⁻³** | **8.20 × 10⁻⁴** | **0.951** |
| 10 mV step | 0.010 | 0.0830 | 0.0347 | 0.612 |

### 앙상블 검증 (선택 형상, 고해상도 격자)

| 모델 | ε_E95 | E_relative_l2 | local_pass_fraction |
|------|-------|----------------|----------------------|
| base-12 | 5.82 × 10⁻⁵ | 3.53 × 10⁻⁵ | 1.0 |
| seed-21 | 5.21 × 10⁻⁴ | 3.23 × 10⁻⁴ | 1.0 |
| seed-35 | 4.27 × 10⁻⁴ | 2.85 × 10⁻⁴ | 1.0 |
| seed-47 | 3.88 × 10⁻⁴ | 3.09 × 10⁻⁴ | 1.0 |
| seed-63 | 3.83 × 10⁻⁴ | 2.12 × 10⁻⁴ | 1.0 |
| seed-88 | 4.80 × 10⁻⁴ | 3.01 × 10⁻⁴ | 1.0 |

### 물리 검증

| 지표 | 값 |
|------|----|
| 목표장 div(E) RMS (정규화) | 2.55 × 10⁻⁴ |
| 목표장 curl(E) RMS (정규화) | 1.64 × 10⁻⁴ |
| 재현장 ∇²φ RMS (정규화) | 4.41 × 10⁻³ |
| 최대 원리 위반 | 0.0 |

---

## 2D 도전성 종이 실험 (`02_conductive_paper_analysis.ipynb`)

- **데이터**: `data/conductive_paper_measurements_remeasured.csv`
- **분석 전극**: E0, E1, E2, E3, E12, E13, E14, E15 (8개)
- **측정점**: x = 4, 7, 10, 13, 16, 19, 22, 25 cm × y = 3, 6, 9, 12, 15, 18 cm → 48점
- **피팅 방법**: NNLS (비음수 최소 제곱), Positive-OMP, Positive-Lasso

---

## 실행 순서

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 핵심 모듈 (직접 실행 불필요, import됨)
#    generate_field/potential_boundary_reconstruction.py

# 3. 3D 수치 시뮬레이션 노트북
jupyter notebook 01_numerical_analysis.ipynb

# 4. 최종 검증 스크립트 (outputs/closure_analysis/ 에 JSON + 그래프 저장)
python closure_analysis.py

# 5. 2D 실험 분석 노트북
jupyter notebook 02_conductive_paper_analysis.ipynb
```

> **참고**: `cupy` (GPU 가속)는 선택 사항입니다. 없으면 NumPy/CPU로 자동 전환됩니다.  
> `closure_analysis.py`의 shape sweep은 수백 후보를 탐색하므로 CPU에서 수 분이 소요될 수 있습니다.

---

## 기준 출력 파일

`results/closure_analysis/closure_analysis.json`에 `closure_analysis.py`의 참조 실행 결과가 수록되어 있습니다.  
이 파일의 수치가 논문 표/본문 수치의 직접 출처입니다.

---

## 환경

- Python ≥ 3.11
- numpy ≥ 2.0, scipy ≥ 1.14, pandas ≥ 2.2, matplotlib ≥ 3.9
- plotly ≥ 6.0, numba ≥ 0.61
- (선택) cupy-cuda12x ≥ 13.0
