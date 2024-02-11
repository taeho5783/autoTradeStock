## 종목 불러오기
import pandas as pd
import time

"""KRX로부터 상장기업 목록 파일을 읽어와서 데이터프레임으로 반환"""
url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method='\
    'download&searchType=13'
krx = pd.read_html(url, header=0)[0]
krx = krx[['종목코드', '회사명']]
krx = krx.rename(columns={'종목코드': 'code', '회사명': 'company'})
krx.code = krx.code.map('{:06d}'.format)
krx = krx.set_index('code')


codes = list(krx.index)
companys = [krx.loc[code,'company'] for code in codes]

# print(companys)

def backtest(종목명, k, start='2020-01-01', end_date='2021-02-14'):
    df=mk.get_daily_price(종목명, start, end_date)
    df.set_index(df['date'].apply(lambda x:pd.to_datetime(x)),inplace=True)
    df['diff_ratio'] = df['diff'] / df['close'].shift(1)
    df['open_gap'] = df['open']-df['close'].shift(1)
    
    df['변동폭'] = df['high']-df['low']
    df['목표가'] = df['open'] + df['변동폭'].shift(1)*k
    df['MA3_yes'] = df.close.rolling(window=3).mean().shift(1)
    df['내일시가'] = df.open.shift(-1)
    df['MA20'] = df['close'].rolling(window=20).mean() 
    df['stddev'] = df['close'].rolling(window=20).std() 
    df['upper'] = df['MA20'] + (df['stddev'] * 2)
    df['lower'] = df['MA20'] - (df['stddev'] * 2)
    df['PB'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])
    df['PB_yes'] = df['PB'].shift(1)
    df['bandwidth'] = (df['upper'] - df['lower']) / df['MA20'] * 100 # ①  
    cond = ( df['high'] > df['목표가'] ) & ( df['목표가'] > df['MA3_yes'] ) & ( df['PB_yes'] < 0.6 )
    
    df.loc[cond,'수익률'] = df.loc[cond,'내일시가']/df.loc[cond,'목표가']*0.9975 - 0.006 #0.9975 수수료, 0.002 슬리피지
    df=df[19:]
    #print(df.dropna().수익률.cumprod().iloc[-1])
    #df.수익률.plot.hist(bins=20)
    #return df #일반
    return df['수익률'] #return 데이터프레임용

def 돌려보기(codes, start_date, end_date):
    def load_data(code, k, start='2020-01-01'):
        
        df=mk.get_daily_price(code, start, end_date)
        df['변동폭'] = df['high'] - df['low']
        df['목표가'] = df['open'] + df['변동폭'].shift(1)*k
        df['어제종가'] = df['close'].shift(1)
        df['내일시가'] = df['open'].shift(-1)
        df['어제거래량'] = df['volume'].shift(1)
        df['그제거래량'] = df['volume'].shift(2)
        df['시가-어제종가'] = df['open']-df['어제종가']
        df['MA3_yes'] = df['close'].rolling(window=3).mean().shift(1)
        df['MA20'] = df['close'].rolling(window=20).mean() 
        df['stddev'] = df['close'].rolling(window=20).std() 
        df['upper'] = df['MA20'] + (df['stddev'] * 2)
        df['lower'] = df['MA20'] - (df['stddev'] * 2)
        df['PB'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])
        df['PB_yes'] = df['PB'].shift(1)
        df['bandwidth'] = (df['upper'] - df['lower']) / df['MA20'] * 100 # ①
        df['TP'] = (df['high'] + df['low'] + df['close']) / 3
        df[19:]
        return df

    start = time.time()
    a=[]
    b=[]
    c=[]
    d=[]
    e=[]
    f=[]
    g=[]
    h=[]
    i=[]
    j=[]
    k=[]
    l=[]
    m=[]
    n=[]
    name=[]
    #name2=[]
    iteration=0
    for code in codes:
        iteration=iteration+1    
        if iteration%500==0: #알림용
            print(iteration)
        try:
            df = load_data(code, 0.5, start=start_date)
            기간 = df.shape[0] # 기간수
            cond = ( df['high'] > df['목표가'] ) & ( df['목표가'] > df['MA3_yes'] ) & (df['PB_yes']<0.6)# 구매조건
            df=df[cond]

            df['수익률'] = df['내일시가']/df['목표가']*0.9975 - 0.006 #0.9975 수수료, 0.002 슬리피지
            df['승패'] = np.where(df['수익률']>1, 1, 0)
            df=df.iloc[:-2]

            조건만족횟수 = df.shape[0] # 조건만족 수
            조건만족비율 = 조건만족횟수/기간
            조건승률 = df['승패'].value_counts()[1] / len(df['승패'])
            #최근승률 = df[-1:].승패.value_counts()[1]/ len(df[-10:].승패)
            보유수익률 = (df['close'][-1]/df['close'][0]*0.9975-0.006-1)
            돌파수익률 = (df.수익률.cumprod()[-1]-1)
            최대수익률 = (df.loc[df.수익률.idxmax()].수익률-1)
            평균수익률 = df.수익률.mean()-1
            중앙수익률 = df.수익률.median()-1
            수익률표준편차 = df.std()['수익률']
            최대손실률 = (df.loc[df.수익률.idxmin()].수익률-1)
            기간수익률 = df.수익률.cumprod().iloc[-1]
            돌파비율 = 돌파수익률-보유수익률
            N = (df.index[-1] - df.index[0]).days / 252
            M = df.dropna().shape[0]
            CAGR = (기간수익률 ** (1/N))-1
            기하평균수익률 = (기간수익률 **(1/M))-1
            name.append(code)
            #name2.append(krx.loc[code])
            a.append(조건만족횟수)
            b.append(조건만족비율)
            c.append(조건승률)
            #d.append(최근승률)
            e.append(보유수익률)
            f.append(돌파수익률)
            g.append(최대수익률)
            h.append(평균수익률)
            i.append(중앙수익률)
            j.append(최대손실률)
            k.append(CAGR)
            l.append(기하평균수익률)
            m.append(수익률표준편차)
            n.append(돌파비율)
        except :
            pass
        df=pd.DataFrame({"종목이름":name,"조건만족횟수":a,"조건만족비율":b,"조건승률":c,"보유수익률":e,"돌파수익률":f,"평균수익률":h,"수익률표준편차":m,"중앙수익률":i,"최대수익률":g,"최대손실률":j,"돌파비율":n,"CAGR":k,"기하수익률":l})
    print("완료 소요시간 :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    df['돌파-보유'] = df['돌파수익률'] - df['보유수익률']
    df['랭크_돌파'] = df['돌파수익률'].rank(ascending=False) 
    df['랭크_중앙'] = df['중앙수익률'].rank(ascending=False) 

    #k는 그룹수
    #몫 = df.shape[0]//4 +1=225
    def 그룹화(x):
        return (x//df.shape[0]//4)+1
    df['랭크_돌파'] = df['랭크_돌파'].apply(lambda x:그룹화(x))    
    df['랭크_중앙'] = df['랭크_중앙'].apply(lambda x:그룹화(x))
    
    return df

def returns_(returns):
    import random
    returns_=pd.DataFrame()
    returns_['min']= returns.min(axis=1)
    returns_['mean']= returns.mean(axis=1)
    returns_['median']= returns.median(axis=1)
    returns_['max']= returns.max(axis=1)
    returns_['count'] = returns.count(axis=1)
    returns_['승패'] = returns_['mean'].map(lambda x:1 if x>1 else 0 )
    returns_['count_rev'] = returns_['count'].map(lambda x : x/5 if 0<x<5 else 1) #5종목 이내일 때는 1/5
    returns_['mean_rev'] = returns_['mean']**(returns_['count_rev'])
    for date in list(returns_.index):
        codes_date = returns.loc[date].dropna()
        if len(codes_date) >=5:
            codes_rand = random.sample(range(0,len(codes_date)),5)
            returns_.loc[date, 'mean_rand5'] = codes_date[codes_rand].mean()
        else:
            returns_.loc[date, 'mean_rand5'] = returns_.loc[date,'mean']**(returns_.loc[date,'count_rev'])

    returns_['count_rev2'] = returns_['count'].map(lambda x : x/2 if 0<x<2 else 1) #5종목 이내일 때는 1/5
    returns_['mean_rev2'] = returns_['mean']**(returns_['count_rev2'])
    for date in list(returns_.index):
        codes_date = returns.loc[date].dropna()
        if len(codes_date) >=2:
            codes_rand = random.sample(range(0,len(codes_date)),2)
            returns_.loc[date, 'mean_rand2'] = codes_date[codes_rand].mean()
        else:
            returns_.loc[date, 'mean_rand2'] = returns_.loc[date,'mean']**(returns_.loc[date,'count_rev2'])
    for date in list(returns_.index):
        codes_date = returns.loc[date].dropna()
        if len(codes_date) >=1:
            codes_rand = random.sample(range(0,len(codes_date)),1)
            #returns_.loc[date, 'mean_rand1'] = codes_date[codes_rand].mean()
            returns_.loc[date, 'mean_rand1'] = codes_date[random.sample(list(codes_date.index),1)[0]]
        else:
            returns_.loc[date, 'mean_rand1'] = returns_.loc[date,'mean']



    #print("5종목 평균수익률 : {:.2f}".format(returns_['mean_rev'].dropna().cumprod().iloc[-1]))
    #print("2종목 평균수익률 : {:.2f}".format(returns_['mean_rev2'].dropna().cumprod().iloc[-1]))
    print("1종목 임의평균수익률 : {:.2%}".format(returns_['mean_rand1'].dropna().cumprod().iloc[-1]))
    print("2종목 임의평균수익률 : {:.2%}".format(returns_['mean_rand2'].dropna().cumprod().iloc[-1]))
    print("5종목 임의평균수익률 : {:.2%}".format(returns_['mean_rand5'].dropna().cumprod().iloc[-1]))

    #print("1종목 중앙수익률 : {:.2f}".format(returns_['median'].dropna().cumprod().iloc[-1]))
    returns_[['mean_rand1','mean_rand2','mean_rand5']].cumprod().loc[:].plot()
    return returns_

codes = companys
start_date='2019-12-01'#30일 전으로 가야함 20일 이동평균을 활용하기 위해, 
end_date = '2020-12-31'
df = 돌려보기(codes, start_date, end_date)

df

cond = (df['조건승률']>0.53 ) &(df['기하수익률']>0.005) & (df['최대손실률']>-0.07) & (df['조건만족횟수']>3) & (df['랭크_돌파']<3) & (df['랭크_중앙']<3)
print(df[cond].shape[0])
print("평균승률 : {:.2f}".format(df[cond].조건승률.mean()))
print("평균수익률 : {:.2f}".format(df[cond].평균수익률.mean()))
print("중앙수익률 : {:.2f}".format(df[cond].중앙수익률.mean()))
print("기하수익률 : {:.2f}".format(df[cond].기하수익률.mean()))
print("돌파수익률 : {:.2f}".format(df[cond].돌파수익률.mean()))
new_codes = list(df[cond]['종목이름'])
df[cond].sort_values(by='기하수익률',ascending=False).head()

start= '2019-12-01' #한달전으로 세팅
end = '2020-12-31'
returns = pd.DataFrame()
for code in new_codes:
    df2 = backtest(code,k=0.5,start=start, end_date=end)
    returns[code] = df2
    #time.sleep(0.01)
returns.set_index(returns.reset_index()['date'].apply(lambda x:pd.to_datetime(x)),inplace=True)
returns.set_index(returns.index.strftime("%Y-%m-%d"),inplace=True)

print("{} ~ {} 수익률 : {:.2%}".format(start, end, returns.mean(axis=1).dropna().cumprod().iloc[-1]))
a=returns_(returns)

