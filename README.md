# 악천후 환경에서 기존 단안 3D 객체 검출 모델들의 한계 분석 

## Project Summary

본 프로젝트는 악천후(안개, 비) 환경에서 기존 단안 3D 객체 검출(Monocular 3D Object Detection) 모델들의 성능 한계를 분석하는 것을 목표로 한다. 단안 3D 객체 검출은 단일 RGB 이미지로 객체의 위치, 크기, 방향을 추정하는 기술로, 자율주행 및 로봇 비전 분야에서 널리 활용되고 있다. 하지만 악천후 상황에서는 영상 품질 저하로 인해 성능이 크게 감소하는 문제가 존재한다.

본 연구에서는 최신 모델인 MonoDTR와 MonoWAD의 성능을 다양한 날씨 조건에서 비교 분석하였으며, 단순한 데이터 확장이 오히려 성능 저하를 일으킬 수 있음을 확인하였다. 이를 통해 날씨별 학습 전략의 중요성을 실험적으로 검증하였다.

## Code Instruction
### 데이터셋
  - 기본 데이터셋: KITTI 3D Object Detection Benchmark
  - Foggy 이미지: DORN 기반 깊이 추정 + Koschmieder 법칙 적용
  - Rainy 이미지: 물리 기반 시뮬레이션 (rain mask 활용)
    -> rain_mask.py: 기존 이미지에 rain mask를 적용하여 rainy 이미지를 생성
    -> 입력: clear image & rain mask
  - 모든 변환 이미지는 원본 KITTI label과 1:1 매칭된다
### 학습 환경
  - GPU: NVIDIA RTX 3090 (1개)
  - epoch: 120
  - 평가 지표: AP_3D (Average Precision at 40 recall points, AP_40)
  - 대상 클래스: 자동차 (Easy, Moderate, Hard)
### 학습 설정
  - MonoDTR
    - Clear only
    - Clear + Foggy
    - Clear + Rainy
    - Clear + Foggy + Rainy
  - MonoWAD
    - Clear + Foggy
    - Clear + Rainy
    - random choice (Clear/Foggy 또는 Clear/Rainy를 랜덤하게 선택)
### 주요 분석 코드
- compare_fft.py
  -> clear/foggy/rainy 이미지들의 FFT 결과 및 Power Spectral Density(PSD) curve 시각화
  -> 입력: clear, foggy, rainy images
- max_activation.py
  -> 각 이미지들의 feature map(.npy files) 파일로부터 max activation map을 시각화 (heatmap)
  -> 입력: backbone을 통과한 feature maps
- noise_comparison.py
 -> MonoWAD 에서는 (foggy-clear), (rainy-clear) 처럼 feature map의 차이를 Diffusion 모델의 noise로 정의하기 때문에 이를 분석하고, t-SNE 시각화 및 여러 분석 그래프를 생성한다
  -> 임력: noise feature maps (.npy file)
  
## Demo
### feature map과 Frequency Domain 분석 (+PSD curve)
![Max Activation과 frequency domain](./out/psd_compare.png)
![PSD 곡선](./out/psd_compare.png)
### Noise 분석
![Noise 분석](./out/psd_compare.png)

## Conclusion and Future Work
### Conclusion
본 연구를 통해 날씨 도메인을 단순히 통합 학습할 경우 성능 저하가 발생할 수 있음을 확인하였다. MonoDTR는 clear 단일 도메인 학습 시 가장 높은 성능을 보였으며, MonoWAD는 clear + foggy 조합에서 가장 우수한 성능을 기록하였다.

PSD 분석과 Noise 분석 결과, 날씨 조건별로 feature 분포가 상이하며 diffusion 기반 deweathering 프로세스는 이를 효과적으로 처리하기 어렵다는 한계를 확인하였다.
### Future work
날씨 도메인별로 채널 수준의 가중치(channel-level weighting)를 적용하는 방법 연구
diffusion 이외의 방법을 활용하여 빠른 추론 속도와 우수한 성능을 달성할 수 있는 단안 3D 객체 검출 모델 개발
  - Transformer 기반 de-weathering 기법
  - VLM(Visual-Language Model) 기반 de-weathering 기법

