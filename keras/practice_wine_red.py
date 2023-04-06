import pandas as pd

# 데이터가 저장된 CSV 파일을 불러옵니다.
data = pd.read_csv(path +'관광펜션업.csv', encoding='cp949')

# '소재지전체주소' 열에서 '휴업' 또는 '폐업'이 포함된 행을 찾습니다.
closed = data[data['영업상태명'].str.contains('휴업|폐업|영업|취소')]

# 찾은 행의 개수를 출력합니다.
print(f"영업,휴업,취소 또는 폐업인 가게 수: {len(closed)}")

# 찾은 행의 지역 정보를 출력합니다.
for location in closed['소재지전체주소']:
    print(location)
    
    
 ##############################################################################   
    import pandas as pd

# 데이터가 저장된 CSV 파일을 불러옵니다.
data = pd.read_csv(path +'관광펜션업.csv', encoding='cp949')

# '소재지전체주소' 열에서 첫번째 단어만 추출하여 새로운 '지역' 열을 생성합니다.
data['지역'] = data['소재지전체주소'].str.split().str.get(0)

# '지역' 열에서 '휴업|폐업|영업|취소'이 포함된 행을 찾습니다.
closed = data[data['영업상태명'].str.contains('휴업|폐업|영업|취소')]

# 찾은 행의 개수를 출력합니다.
print(f"영업,휴업,취소 또는 폐업인 가게 수: {len(closed)}")

# 찾은 행의 지역 정보를 출력합니다.
for location in closed['지역']:
    print(location)
    
 ############################################################################
import pandas as pd

# 데이터가 저장된 CSV 파일을 불러옵니다.
data = pd.read_csv(path +'관광펜션업.csv', encoding='cp949')

# '소재지전체주소' 열에서 첫번째 단어만 추출하여 새로운 '지역' 열을 생성합니다.
data['지역'] = data['소재지전체주소'].str.split().str.get(0)

# '지역'과 '영업상태명' 열만 추출합니다.
subset = data[['지역', '영업상태명']]

# '지역'과 '영업상태명' 열의 조합별로 개수를 세어 데이터프레임으로 저장합니다.
counts = pd.crosstab(subset['지역'], subset['영업상태명'])

# 비율을 계산합니다.
percentages = counts.apply(lambda x: x/x.sum(), axis=1)

# 결과를 출력합니다.
print(percentages)

#############################################################################
    