아래는 LiₓFePO₄ 시스템을 대상으로 pymatgen을 통해 Li 농도별 전압 프로파일과 Open Circuit Potential(OCP)을 추정하는 예시 코드와, 이론적인 결과 예시입니다.

⸻

1. 환경 세팅 및 의존성 설치 📦

pip install pymatgen matplotlib

VASP 의존 없이, 이미 계산된 ComputedEntry들을 사용하여 진행합니다.
(실제 DFT 결과는 미리 VASP 등으로 계산되어 있어야 합니다.)

⸻

2. Python 코드 예시

from pymatgen.apps.battery.insertion_battery import InsertionVoltagePair
from pymatgen.apps.battery.plotter import VoltageProfilePlotter
from pymatgen.entries.computed_entries import ComputedEntry

# (1) 예시: 이미 계산된 상태 entry 불러오기
# 아래는 예시일 뿐, 실제에는 DFT 결과 파일이나 MP DB에서 불러와야 함
entry_Li1 = ComputedEntry("LiFePO4", energy=-20.0)      # x = 1
entry_Li03 = ComputedEntry("Li0.3FePO4", energy=-18.5)  # x = 0.3
entry_Li0 = ComputedEntry("FePO4", energy=-17.0)       # x = 0

# 기준 리튬 메탈 상대 entry
entry_Li_metal = ComputedEntry("Li", energy=-1.9)

# (2) 서로 다른 조성 쌍에서 평균 전압 계산
pairs = []
for entry_i, entry_j in [(entry_Li1, entry_Li03), (entry_Li03, entry_Li0)]:
    pair = InsertionVoltagePair.from_entries(
        entry_i, entry_j, working_ion_entry=entry_Li_metal
    )
    pairs.append(pair)

# (3) 전압 프로필 그리기
plotter = VoltageProfilePlotter(xaxis="frac_x")
for p in pairs:
    plotter.add_electrode(p)

# 결과 출력
fig = plotter.get_plot()
fig.savefig("LiFePO4_voltage_profile.png")


⸻

3. 예상되는 결과 해석
	•	x = 1 → 0.3 구간의 평균 전압:
V_{1→0.3} = - \frac{E_{0.3} + 0.7\,E_{Li} - E_1}{0.7\,F}
	•	x = 0.3 → 0 구간의 평균 전압:
V_{0.3→0} = - \frac{E_0 + 0.3\,E_{Li} - E_{0.3}}{0.3\,F}
	•	그 결과는 계단형(voltage plateaus) 그래프로 출력됩니다—이는 **Open Circuit Potential(OCP)**을 이론적으로 재현한 값입니다.

⸻

4. 작은 예시 결과 (가정 기반)

구간     x 변화     전압
1 → 0.3     0.7      약 3.45 V
0.3 → 0      0.3      약 3.20 V

그래프에서는 두 구간 사이에서 전압이 단계적으로 변하는 모습을 확인할 수 있습니다.

⸻

5. 요약
	•	**여러 조성(x)**에 대해 각각 DFT energy를 계산해야 하며,
	•	세부적인 전압 profile과 plateaus를 얻을 수 있습니다.
	•	이 값들은 OCP와 동일하게 해석될 수 있습니다.

⸻

추가 지원 가능
	•	실제 DFT 파일(VASP OUTCAR/CONTCAR → ComputedEntry) 사용 예
	•	여러 조성과 phase stability (convex hull) 분석
	•	실험 전압 데이터(예: LiₓFePO₄: 3.45 V, 3.20 V plateau)와 비교 논문 소개

필요하시면 이어서 코드 구현, 파일 설정, MP API 연동 등을 도와드릴게요!