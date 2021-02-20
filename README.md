# personal_master
석사과정으로 있을 때 진행한 프로젝트 모음

## util_torch

### my_tensorboard.py
하이퍼파라미터 튜닝이 가능하도록 구성한 텐서보드 모듈 

### torch_find_lr.py

에폭 코스에 대해, 작은 학습율로 시작하고 각 미니 배치마다 높은 학습율로 증가시켜서 에폭의 끝에 높은 비율을 만들어 냅니다. 
각 학습율에 대해 loss를 계산하고 가장 크게 감소하는 학습율을 선택합니다. 

여기서 진행하는 것은 배치마다 반복하여 평소대로 거의 훈련합니다. 모델을 통해 인풋을 전달하고나서 배치로부터 loss를 얻습니다. 현재까지 best_loss를 기록하고 새로운 loss를 비교합니다. 새로운 loss가 best_loss의 4배 이상이면 함수를 중단하고 지금까지의 값을 반환합니다(loss가 발산할 가능성). 그렇지 않다면, loss와 현재 학습율의 로그값을 기록해서 학습 속도를 다음 단계로 갱신하여 반복문 끝의 최대 속도로 이동합니다. 

학습율과 loss의 슬라이싱해서 반환한 이유는 처음 값들과 마지막 값들은 좋은 정보를 주지 않는 경향이 있기 때문입니다. 

fas.ai의 라이브러리에서 구현은 가중화된 smoothing을 포함하고 있어서 플랏의 smooth line을 그릴 수 있습니다. 마지막으로, 이 함수는 실제로 모델을 훈련시키고 optimizer의 학습율 설정을 방해하기 때문에, 미리 모델을 저장하고 다시 로드하여`find_lr()`를 호출하기 전에 상태로 되돌리고 또한 선택한 optimizer를 다시 초기화해야 한다는 것을 기억하세요.

이 모듈을 사용하여 찾아낸 학습률을 이용해서 transfer learning에서는 각 층마다 다른 학습률로 훈련시킬 수 있습니다. 

### torch_datasplit.py
training set, validation set, test set을 나눕니다. 


### torch_datasplit_directory.py
training set, validation set, test set을 디렉토리 단위로 나누고 미리 저장합니다. 따라서 load가 빠르고 split의 랜덤성을 하지 않아도 됩니다. 




