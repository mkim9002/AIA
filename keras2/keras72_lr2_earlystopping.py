# 실습 얼리 스타핑 적용 하려면 어떻게 하나?
#1. 최소값을 넣을 변수를 하나 준비.
#2. 다음 에포의 값과 최소값을 비교. 최소값이 변경되면 그 변수에 최소값 넣어주고, 카운트 변수 초기화
#3. 갱신이 안되면 카운트 변수 ++1
# 카운트 변수가 내가 원하는 얼리 스타핑 갯수에 도달하면 for 문을 stop



x = 10
y = 10
w = 1111
lr = 0.001
epochs = 1000

early_stopping_count = 0  # 얼리 스타핑 카운트 변수
min_loss = float('inf')  # 최소 손실값 변수

for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y)**2  # mse
    
    print('Loss:', round(loss, 4), '\tPredict:', round(hypothesis, 4))
    
    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2
    
    if up_loss >= down_loss:
        w = w - lr
    else:
        w = w + lr
    
    # 얼리 스타핑 조건 확인
    if loss < min_loss:
        min_loss = loss
        early_stopping_count = 0
    else:
        early_stopping_count += 1
    
    if early_stopping_count >= 3:  # 원하는 얼리 스타핑 갯수 (3)에 도달하면 종료
        print('Early Stopping')
        break

        