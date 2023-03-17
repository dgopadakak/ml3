import math
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


def nan_to_zero(x):
    if not isinstance(x, str):
        if math.isnan(x):
            return 0
    return x


def no_yes_to_false_true(x):
    if isinstance(x, str):
        if x == 'no':
            return False
        elif x == 'yes':
            return True
    return x


def eco_to_num(x):
    if isinstance(x, str):
        if x == 'no data':
            return 0
        if x == 'poor':
            return 1
        if x == 'satisfactory':
            return 2
        if x == 'good':
            return 3
        if x == 'excellent':
            return 4
    return x


# 1 ################################
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

# 2 ################################

# Визуализируем распределение количеств комнат
sns.histplot(df['num_room'], kde=True)
plt.show()

# Визуализируем распределение длин маршрутов до метро
sns.histplot(df['metro_km_walk'], kde=True)
plt.show()

# Визуализируем распределение цен квартир
sns.histplot(df['price_doc'], kde=True)
plt.show()

# Визуализируем корреляцию между переменными
sns.heatmap(df.corr())
plt.show()

# 3 ################################

# Из первичного анализа мы можем видеть, что набор данных содержит характеристики
# и сопутствующие данные проданных квартир
# Целевыми переменными являются количество комнат, расстояние до метро пешком и цена.
# Набор данных также содержит недостающие значения, обозначенные NA,
# которые нам нужно будет обработать на этапе предварительной обработки данных.

# 4 ################################

# Удаляем столбец даты
df.drop(['timestamp'], axis=1, inplace=True)
df.drop(['product_type'], axis=1, inplace=True)
df.drop(['sub_area'], axis=1, inplace=True)
# df = df.drop('C6H6(GT)', axis=1)                                   закомментил, так как не понял зачем это

# Заменяем отсутствующие значения 0
df = df.applymap(lambda x: nan_to_zero(x))
df = df.applymap(lambda x: no_yes_to_false_true(x))
df = df.applymap(lambda x: eco_to_num(x))

# Заменяем значения 0 на среднее значение

tags = ['max_floor', 'material', 'build_year', 'num_room', 'kitch_sq', 'state', 'hospital_beds_raion', 'school_quota',
        'cafe_sum_500_min_price_avg', 'cafe_sum_500_max_price_avg', 'cafe_avg_price_500', 'cafe_sum_1000_min_price_avg',
        'cafe_sum_1000_max_price_avg', 'cafe_avg_price_1000', 'cafe_sum_1500_min_price_avg',
        'cafe_sum_1500_max_price_avg', 'cafe_avg_price_1500', 'cafe_sum_2000_min_price_avg',
        'cafe_sum_2000_max_price_avg', 'cafe_avg_price_2000', 'cafe_sum_3000_min_price_avg',
        'cafe_sum_3000_max_price_avg', 'cafe_avg_price_3000', 'cafe_sum_5000_min_price_avg',
        'cafe_sum_5000_max_price_avg', 'cafe_avg_price_5000']
for tag in tags:
    for i in df[df[tag] == 0].index:
        df.loc[i, tag] = df[tag].mean()

# 5 ################################

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

# 6 ################################

# Выделяем данные на обучающие и тестовые наборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

param_grid = {'n_estimators': [10, 50, 100, 500],
              'max_depth': [5, 10, None]}

# Создай random forest regressor
rf = RandomForestRegressor(random_state=0)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)

# Обучение
grid_search.fit(X_train, y_train)

# 7 ################################

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

# 8 ################################

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
