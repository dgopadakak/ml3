import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1
# Считываем набор данных
df = pd.read_csv('train_2.csv', sep=',', decimal='.', header=0)

# Первые 5 строк
print(df.head())

# Печатаем форму(?) данных
print(df.shape)

# Вывели типы столбцов и ненулевые значения
print(df.info())

# Вывели описание(сводную статистику)
print(df.describe())

# Мы можем видеть, что набор данных содержит 30471 экземпляр и 292 столбца.
# Есть некоторые недостающие значения, обозначаемые NA.
# Нам нужно будет обработать эти недостающие значения на этапе предварительной обработки данных.

# 2

# Визуализируем распределение количеств комнат
sns.histplot(df['num_room'], kde=True)
plt.show()

# Визуализируем распределение длин маршрутов до метро
sns.histplot(df['metro_km_walk'], kde=True)
plt.show()

# Визуализируем корреляцию между переменными
sns.heatmap(df.corr())
plt.show()

# 3

# Из первичного анализа мы можем видеть, что набор данных содержит усредненные(по часам) отклики датчиков от 5
# химических датчиков оксида металла.
# Целевыми переменными являются усредненные(по часам) концентрации CO, неметановых углеводородов,
# бензола(С6H6), общих оксидов азота (NOx) и диоксида азота (NO2),
# измеряемые совместно расположенным эталонным сертифицированным анализатором.
# Набор данных также содержит недостающие значения, обозначенные -200,
# которые нам нужно будет обработать на этапе предварительной обработки данных.

# 4

# Удаляем столбцы даты и времени
df.drop(['Date', 'Time'], axis=1, inplace=True)
df = df.drop('C6H6(GT)', axis=1)

# Заменяем отсутствующие значения 0
# Сначала заполним это недостающее значение значением 0,
# потому что, если заполним средним значением,
# значение не будет репрезентативным,
# потому что оно содержит значение -200.
df = df.applymap(lambda x: x if x >= 0 else 0)

# Заменяем значения 0 на среднее значение/медиану

for i in df[df['CO(GT)'] == 0].index:
    df.loc[i, 'CO(GT)'] = df['CO(GT)'].mean()
for i in df[df['PT08.S1(CO)'] == 0].index:
    df.loc[i, 'PT08.S1(CO)'] = df['PT08.S1(CO)'].mean()
for i in df[df['NMHC(GT)'] == 0].index:
    df.loc[i, 'NMHC(GT)'] = df['NMHC(GT)'].mean()
for i in df[df['PT08.S2(NMHC)'] == 0].index:
    df.loc[i, 'PT08.S2(NMHC)'] = df['PT08.S2(NMHC)'].mean()
for i in df[df['NOx(GT)'] == 0].index:
    df.loc[i, 'NOx(GT)'] = df['NOx(GT)'].median()
for i in df[df['PT08.S3(NOx)'] == 0].index:
    df.loc[i, 'PT08.S3(NOx)'] = df['PT08.S3(NOx)'].mean()
for i in df[df['NO2(GT)'] == 0].index:
    df.loc[i, 'NO2(GT)'] = df['NO2(GT)'].mean()
for i in df[df['PT08.S4(NO2)'] == 0].index:
    df.loc[i, 'PT08.S4(NO2)'] = df['PT08.S4(NO2)'].median()
for i in df[df['PT08.S5(O3)'] == 0].index:
    df.loc[i, 'PT08.S5(O3)'] = df['PT08.S5(O3)'].median()
for i in df[df['T'] == 0].index:
    df.loc[i, 'T'] = df['T'].mean()
for i in df[df['RH'] == 0].index:
    df.loc[i, 'RH'] = df['RH'].mean()
for i in df[df['AH'] == 0].index:
    df.loc[i, 'AH'] = df['AH'].mean()

# 5

# У этого набора данных нет класса.
# Поэтому используем кластеризацию k-средних для заполнения класса(прост и достаточно точен)
# Выбрано 2 кластера, потому что в описании данных указано,
# что датчик был расположен на поле в значительно загрязненной зоне.
# Таким образом, кластер 0 представляет собой ОЧЕНЬ сильно загрязненный,
# а кластер 1 представляет собой сильно загрязненный.

km = KMeans(n_clusters=2, random_state=1)
new = df._get_numeric_data()
km.fit(new)
predict = km.predict(new)
df['Class'] = pd.Series(predict, index=df.index)

# Разделяем набор данных на объекты и целевую переменную
X = df.drop('Class', axis=1)
y = df.loc[:, 'Class'].values

# 6

# Выделяем данные на обучающие и тестовые наборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

param_grid = {'n_estimators': [10, 50, 100, 500],
              'max_depth': [5, 10, None]}

# Создай random forest regressor
rf = RandomForestRegressor(random_state=0)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)

# Обучение
grid_search.fit(X_train, y_train)

# 7

# Определяем диапазон размеров тестовой выборки
train_sizes = np.linspace(0.1, 1.0, 10)

train_sizes, train_scores, validation_scores = learning_curve(
    rf, X_train, y_train, train_sizes=train_sizes, cv=5, scoring='neg_mean_squared_error')

train_scores_mean = -train_scores.mean(axis=1)
validation_scores_mean = -validation_scores.mean(axis=1)

plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, validation_scores_mean, label='Validation error')
plt.xlabel('Number of training samples')
plt.ylabel('Mean squared error')
plt.title('Learning curve')
plt.legend()
plt.show()

# Определяем диапазон значений параметра
max_depth_range = range(1, 21)

train_scores, validation_scores = validation_curve(
    rf, X_train, y_train, param_name='max_depth', param_range=max_depth_range, cv=5, scoring='neg_mean_squared_error')

train_scores_mean = -train_scores.mean(axis=1)
validation_scores_mean = -validation_scores.mean(axis=1)

plt.plot(max_depth_range, train_scores_mean, label='Training error')
plt.plot(max_depth_range, validation_scores_mean, label='Validation error')
plt.xlabel('max_depth')
plt.ylabel('Mean squared error')
plt.title('Validation curve')
plt.legend()
plt.show()

# 8

# Определяем конечную модель с max_depth=7
model = DecisionTreeRegressor(max_depth=7)

# Обучите модель на всей обучающей выборке
model.fit(X_train, y_train)

# делаем предсказания
y_pred = model.predict(X_test)

# Рассчитываем среднеквадратичную ошибку и R-квадрат оценки
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error:', mse)
print('R-squared score:', r2)
