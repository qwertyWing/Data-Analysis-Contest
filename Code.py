import pandas as pd
import numpy as np

df = pd.read_csv('dd/MEASURED_WAVE_TRAIN.csv') #측정데이터(train)
df2 = pd.read_csv('dd/REFERENCE_WAVE.csv') #정상 데이터
ans = pd.read_csv('dd/MEASURED_WAVE_TEST.csv') #test label 없음

df2['QUALITY'] = 0
df2.columns=['REFERENCE_ID','MEASURED_WAVE','QUALITY']
df = df.replace({'QUALITY' : 'GOOD'}, 0)
df1 = df.replace({'QUALITY' : 'BAD'}, 1)

df8 = df2['MEASURED_WAVE'].str.split(', ')
df5 = df8.apply(lambda x: pd.Series(x))
df5.columns=['T'+str(i) for i in range(0,200)]
df2.drop(labels=['MEASURED_WAVE'],axis=1,inplace=True)
df5.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
rev = pd.concat([df2,df5], axis = 1)

df2 = df1['MEASURED_WAVE'].str.split(', ')
df2 = df2.apply(lambda x: pd.Series(x))
df2.columns=['T'+str(i) for i in range(0,200)]
df1.drop(labels=['MEASURED_WAVE'],axis=1,inplace=True)
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
df3 = pd.concat([df1, df2], axis = 1)

df4 = df3.drop(labels=['REFERENCE_ID','QUALITY'],axis=1)
df4.reset_index(drop=True, inplace=True)
df4 = df4.astype('float64')
df4.columns=['Z'+str(i) for i in range(0,200)]
kur = pd.DataFrame(np.array(df4.kurt(axis='columns')).T)
kur.columns = ['kurt']
ske = pd.DataFrame(np.array(df4.skew(axis='columns')).T)
ske.columns = ['skew']
mea = pd.DataFrame(np.array(df4.mean(axis='columns')).T)
mea.columns = ['mean']
var = pd.DataFrame(np.array(df4.var(axis='columns')).T)
var.columns = ['var']
qu1 = pd.DataFrame(np.array(df4.quantile(q=0.75, axis='columns')).T)
qu1.columns = ['q1']
qu4 = pd.DataFrame(np.array(df4.quantile(q=0.25, axis='columns')).T)
qu4.columns = ['q4']
qu2 = pd.DataFrame(np.array(df4.min( axis='columns')).T)
qu2.columns=['min']
qu3 = pd.DataFrame(np.array(df4.max( axis='columns')).T)
qu3.columns=['max']
for i in range(0,199,10):
  qu3['W'+str(i)] = (df4['Z'+str(i+9)] - df4['Z'+str(i)]) / 10
df4 = pd.concat([ske,qu2,qu3], axis = 1)

ans1 = ans['MEASURED_WAVE'].str.split(', ')
ans1 = ans1.apply(lambda x: pd.Series(x))
ans1.columns=['T'+str(i) for i in range(0,200)]
ans.reset_index(drop=True, inplace=True)
ans1.reset_index(drop=True, inplace=True)
ans2 = pd.concat([ans, ans1], axis = 1)
ans2.drop(labels=['MEASURED_WAVE','INDEX'],axis=1,inplace=True)

ans3 = ans2.drop(labels=['REFERENCE_ID'],axis=1)
ans3.reset_index(drop=True, inplace=True)
ans3 = ans3.astype('float64')
ans3.columns=['Z'+str(i) for i in range(0,200)]
kur = pd.DataFrame(np.array(ans3.kurt(axis='columns')).T)
kur.columns = ['kurt']
ske = pd.DataFrame(np.array(ans3.skew(axis='columns')).T)
ske.columns = ['skew']
mea = pd.DataFrame(np.array(ans3.mean(axis='columns')).T)
mea.columns = ['mean']
var = pd.DataFrame(np.array(ans3.var(axis='columns')).T)
var.columns = ['var']
qu1 = pd.DataFrame(np.array(ans3.quantile(q=0.75, axis='columns')).T)
qu1.columns = ['q1']
qu4 = pd.DataFrame(np.array(ans3.quantile(q=0.25, axis='columns')).T)
qu4.columns = ['q4']
qu2 = pd.DataFrame(np.array(ans3.min(axis='columns')).T)
qu2.columns=['min']
qu3 = pd.DataFrame(np.array(ans3.max(axis='columns')).T)
qu3.columns=['max']
for i in range(0,199,10):
  qu3['W'+str(i)] = (ans3['Z'+str(i+9)] - ans3['Z'+str(i)]) / 10
ans3 = pd.concat([ske,qu2,qu3], axis = 1)

t = list(df3['REFERENCE_ID'].unique())
df9 = pd.DataFrame()
for i in range(len(t)):
  d1 = df3[df3['REFERENCE_ID']==t[i]]
  d3 = rev[rev['REFERENCE_ID']==t[i]]
  d1.drop(labels=['REFERENCE_ID','QUALITY'],axis=1,inplace=True)
  d3.drop(labels=['REFERENCE_ID','QUALITY'],axis=1,inplace=True)
  df8 = pd.DataFrame()
  for i in range(len(d1.index)):
    df8 = df8.append(d3, ignore_index = True)
  df8.index = d1.index
  d1 = d1.astype('float64')
  df8 = df8.astype('float64')
  d2 = d1.sub(df8)
  df9 = pd.concat([df9,d2])

df10 = df9.sort_index(ascending=True)
kur = pd.DataFrame(np.array(df10.sum(axis='columns')).T)
kur.columns = ['sum1']
ske = pd.DataFrame(np.array(df10.mean(axis='columns')).T)
ske.columns = ['mean1']
v = pd.DataFrame(np.array(df10.std(axis='columns')).T)
v.columns = ['var1']
#df10 = df10.abs()
s = pd.DataFrame(np.array(df10.quantile(q=0.25, axis='columns')).T)
s.columns = ['q1']
t = pd.DataFrame(np.array(df10.quantile(q=0.75, axis='columns')).T)
t.columns = ['q4']
hello = df3[['REFERENCE_ID', 'QUALITY']]

x_data = pd.concat([hello, df10, df4, kur, ske, v, t, s], axis=1)

t = list(ans2['REFERENCE_ID'].unique())
df9 = pd.DataFrame()
for i in range(len(t)):
  d1 = ans2[ans2['REFERENCE_ID']==t[i]]
  d3 = rev[rev['REFERENCE_ID']==t[i]]
  d1.drop(labels=['REFERENCE_ID'],axis=1,inplace=True)
  d3.drop(labels=['REFERENCE_ID','QUALITY'],axis=1,inplace=True)
  df8 = pd.DataFrame()
  for i in range(len(d1.index)):
    df8 = df8.append(d3, ignore_index = True)
  df8.index = d1.index
  d1 = d1.astype('float64')
  df8 = df8.astype('float64')
  d2 = d1.sub(df8)
  df9 = pd.concat([df9,d2])

df9 = df9.sort_index(ascending=True)

v = pd.DataFrame(np.array(df9.std(axis='columns')).T)
v.columns = ['var1']
s = pd.DataFrame(np.array(df9.quantile(q=0.25, axis='columns')).T)
s.columns = ['q1']
t = pd.DataFrame(np.array(df9.quantile(q=0.75, axis='columns')).T)
t.columns = ['q4']
#df9 = df9.abs()
kur = pd.DataFrame(np.array(df9.sum(axis='columns')).T)
kur.columns = ['sum1']
ske = pd.DataFrame(np.array(df9.mean(axis='columns')).T)
ske.columns = ['mean1']
df9 = pd.concat([df9, ans3, kur, ske, v, t, s], axis=1)

df9.reset_index(drop=True, inplace=True)
x_data.reset_index(drop=True, inplace=True)
df9['sub'] = df9['max'] - df9['min']
x_data['sub'] = x_data['max'] - x_data['min']
x_data.drop(labels=['min'], axis=1, inplace=True)
df9.drop(labels=['min'], axis=1, inplace=True)

x_data.drop(labels=['REFERENCE_ID'], axis=1, inplace=True)

y_train = x_data.iloc[:, 0:1]
X_train = x_data.drop(['QUALITY'], axis=1)

from sklearn.preprocessing import RobustScaler

robust = RobustScaler().fit(X_train)
X_train = robust.transform(X_train)
X_test = robust.transform(df9)


from imblearn.combine import *


svm = SMOTETomek(random_state=1, n_jobs=-1)
X_train, y_train = svm.fit_resample(X_train, y_train)

# from sklearn.neural_network import MLPClassifier
#
#
# lgbm_wrapper = MLPClassifier(hidden_layer_sizes=(210, 360, 300, 150, 50), activation='relu', \
#                            solver='lbfgs', verbose=2 ,max_iter=50 \
#                            )  # 객체 생성
#
# lgbm_wrapper.fit(X_train, y_train)
#
#
#
#
# from collections import Counter
#
# Counter(lgbm_wrapper.predict(X_test))
# an = pd.read_csv('dd/ANSWER_FORM.csv')
# df = pd.DataFrame((lgbm_wrapper.predict(X_test)).T)
# df.columns = ['QUALITY']
# df = df.replace({'QUALITY': 1}, 'BAD')
# df = df.replace({'QUALITY': 0}, 'GOOD')
# an.drop(labels=['QUALITY'], axis=1, inplace=True)
# df1 = pd.concat([an, df], axis=1)
# df1.to_csv('test.csv', index=False)


# import keras
from keras.optimizers import Adam
# from keras.optimizers import adam
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D, Dense
from keras.applications.resnet import ResNet50
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping

callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

model = Sequential()

model.add(Dense(210, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(450, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
# model.add(Dense(50, activation='relu', input_dim=X_train.shape[1]))
# model.add(Dense(16, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    #loss='binary_crossentropy',
    loss='mse',
    metrics=['accuracy']
)

model.summary()
model.fit(X_train, y_train, epochs=45)
model.save("dfff.h5")

mod = load_model('dfff.h5')
a = mod(X_test)

from collections import Counter
print((a))
pred_th = [ 1 if x > 0.5 else 0 for x in a[:,0]]
print(Counter(pred_th))
#Counter({0: 2841, 1: 2513})

an = pd.read_csv('dd/ANSWER_FORM.csv')
df = pd.DataFrame(np.array([pred_th]).T)
df.columns =['QUALITY']
df = df.replace({'QUALITY' : 1}, 'BAD')
df = df.replace({'QUALITY' : 0}, 'GOOD')
an.drop(labels=['QUALITY'],axis=1,inplace=True)
df1 = pd.concat([an, df], axis = 1)
df1.to_csv('test.csv',index = False)