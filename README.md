# 6-DOF Manipulator Simulator

FANUC M-10iA 기반의 6축 직렬 매니퓰레이터 시뮬레이터입니다.  
Python + Tkinter + Matplotlib으로 구현되어 별도의 무거운 프레임워크 없이 실행됩니다.

---

## 기능 (Features)

| 기능 | 설명 |
|------|------|
| DH 파라미터 편집 | FANUC M-10iA 기본값 제공, GUI 테이블에서 실시간 수정 |
| 3D 시각화 | 링크·관절·좌표계 표시 (마우스 회전/이동/확대) |
| 순기구학 (FK) | Standard DH 규약, 모듈화된 kinematics 패키지 |
| 역기구학 (IK) | Damped Least Squares (적응형 댐핑 포함) |
| 키보드 조작 | 엔드이펙터 기준 6자유도 이동/회전 |
| 관절·월드 좌표 표시 | 실시간 관절 슬라이더 + EE 위치/RPY 표시 |
| 속도 조절 | 슬라이더로 0.1× ~ 5× 범위 조절 |
| 외력 분석 | EE 외력 입력 → J^T·F로 관절 부하 계산 및 표시 |
| 작업공간 시각화 | 몬테카를로 샘플링 (25,000점), 3D 산점도 오버레이 |
| 특이점 감지·회피 | 조작성 지수, 조건수 실시간 모니터링 + 가변 댐핑 회피 |

---

## 설치 (Installation)

### 요구사항
- Python 3.10 이상
- tkinter (Python 기본 포함, Windows/macOS 자동 설치됨)

### 패키지 설치

```bash
cd manipulator_sim
pip install -r requirements.txt
```

---

## 실행 (Run)

```bash
cd manipulator_sim
python main.py
```

---

## 키보드 조작 (Keyboard Controls)

> 시뮬레이션 탭에서 3D 뷰를 클릭한 뒤 키를 누르세요.

| 키 | 동작 |
|----|------|
| `A` / `D` | EE X축 이동 (-/+) |
| `W` / `S` | EE Y축 이동 (+/-) |
| `R` / `F` | EE Z축 이동 (+/-) |
| `Q` / `E` | Roll 회전 (-/+) |
| `I` / `K` | Pitch 회전 (+/-) |
| `J` / `L` | Yaw 회전 (-/+) |

> 마우스 조작 (3D 뷰)
> - **좌클릭 드래그**: 회전
> - **우클릭 드래그 / 스크롤**: 확대·축소
> - **중클릭 드래그**: 이동(Pan)

---

## 속도 조절

우측 패널의 **Speed** 슬라이더로 이동 속도를 0.1× ~ 5× 범위에서 조절합니다.

---

## DH 파라미터 편집

1. 상단 탭에서 **DH Parameters** 선택
2. 테이블에서 값을 직접 수정
3. **Apply DH Parameters** 클릭 → 즉시 반영
4. **Reset to FANUC M-10iA** 클릭 → 기본값 복원

### FANUC M-10iA 기본 DH 파라미터 (Standard DH)

| Joint | a (mm) | d (mm) | α (°) | θ_offset (°) | θ_min (°) | θ_max (°) |
|-------|--------|--------|-------|---------------|-----------|-----------|
| J1 | 150 | 520 | -90 | 0 | -170 | 170 |
| J2 | 640 | 0 | 0 | -90 | -100 | 135 |
| J3 | 200 | 0 | -90 | 90 | -120 | 270 |
| J4 | 0 | 700 | 90 | 0 | -190 | 190 |
| J5 | 0 | 0 | -90 | 0 | -120 | 120 |
| J6 | 0 | 115 | 0 | 0 | -360 | 360 |

변환 행렬 정의: **T_i = Rz(θ) · Tz(d) · Tx(a) · Rx(α)**

---

## 외력 분석

우측 패널 **External Wrench at EE** 섹션:
1. Fx, Fy, Fz (N) 및 Mx, My, Mz (N·m) 입력
2. **Apply Force** 클릭
3. **Joint Torques** 섹션에서 τ = J^T · F 결과 확인 (토크 값 + 정격 대비 %)

---

## 작업공간 시각화

1. 우측 패널 **Workspace Visualisation** 섹션에서 **Compute Workspace** 클릭
2. 백그라운드 스레드로 25,000개 관절 조합 샘플링 (수초 소요)
3. 완료 후 **Show** 버튼으로 3D 뷰에 점군 오버레이

---

## 특이점 (Singularity)

하단 상태바에 실시간 표시:

| 색상 | 의미 |
|------|------|
| 🟢 초록 | 정상 — 조작성 지수(w), 최솟값(sv) 표시 |
| 🟠 주황 | 특이점 근접 경고 |
| 🔴 빨강 | 특이점 진입 |

**회피 전략**: 가변 댐핑 DLS (Nakamura & Hanafusa, 1986)  
특이값(σ_min)이 임계치 이하로 내려가면 댐핑 계수λ²를 자동으로 증가시켜  
큰 관절 속도 발생을 억제합니다.

---

## 코드 구조 (Modularity)

```
manipulator_sim/
├── kinematics/               ← 기구학 엔진 (실제 제어기 이관 가능)
│   ├── dh_params.py          #  DH 파라미터 데이터 클래스
│   ├── forward_kinematics.py #  순기구학 (DH 행렬, RPY 변환)
│   ├── jacobian.py           #  기하학적 야코비안
│   ├── inverse_kinematics.py #  역기구학 (DLS, 적응형 댐핑)
│   └── singularity.py        #  특이점 감지 및 가변 댐핑 회피
├── dynamics/
│   └── statics.py            #  정적 힘 분석 (τ = J^T · F)
├── workspace/
│   └── sampler.py            #  몬테카를로 작업공간 샘플링
├── gui/
│   └── main_window.py        #  Tkinter GUI
└── main.py                   #  실행 진입점
```

`kinematics/` 패키지는 GUI에 대한 의존성이 전혀 없습니다.  
실제 제어 시스템에 해당 폴더만 복사하여 그대로 사용할 수 있습니다.

---

## 라이선스

MIT License
