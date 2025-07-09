import pandas as pd
import os

### 지역별 2015~2022 air 데이터 모두 합치는 코드

path = '../../data/air_data/raw_data'
output_path = '../../data/air_data/preprocessed_data'
os.makedirs(output_path, exist_ok=True)  # 결과 저장 디렉토리 생성
# path에서 2015~2022까지의 지역별 데이터 합치기
# location은 2015년_ 으로 시작하는 파일들 찾아서
locations = [f.split('_')[1] for f in os.listdir(path) if f.startswith('2015년')]
print(locations)
# 파일 이름 형식은 X년_지역이름_merged.csv
# 예시: 2015년_서울_merged.csv
# 각 지역별로 파일을 읽어서 하나의 DataFrame으로 합치고 .parquet으로 저장 파일 이름은 air_{location}.parquet
# location별로 하나의 파일로 저장하게 돼야함. 모든 지역의 파일을 한 번에 합치는게 아니라.
years = range(2015, 2023)  # 2015년부터 2022년까지
for location in locations:
    dfs = []
    for year in years:
        file = f'{year}년_{location}_merged.csv'
        df = pd.read_csv(os.path.join(path, file))
        dfs.append(df)
    # 모든 지역의 DataFrame을 하나로 합치기
    combined_df = pd.concat(dfs, ignore_index=True)
    # 측정일시를 문자열로 변환
    combined_df['측정일시'] = combined_df['측정일시'].astype(str)

    # 날짜와 시간 column 한번에 생성 (벡터화 연산)
    combined_df['date'] = pd.to_datetime(combined_df['측정일시'].str[:8], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    combined_df['time'] = combined_df['측정일시'].str[8:10] + ':00'

    # PM25와 초미세먼지 column 통일
    combined_df.loc[combined_df['PM25'].isna() & combined_df['초미세먼지'].notna(), '초미세먼지'] = pd.NA

    # .parquet 파일로 저장
    combined_df.to_parquet(os.path.join(output_path,f'air_{location}.parquet'))