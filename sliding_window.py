import pmdarima as pm
import numpy as np
from pmdarima import model_selection
from matplotlib import pyplot as plt
import pandas as pd
from pprint import pprint as pp
import statsmodels.api as sm
import PyQt5

def get_standard_deviation_dict(start_size: int, train_sales: np.ndarray, forecast_horizon: int):
    test_dict = {}
    counter = 1
    temp = start_size
    count_iter = len(train_sales) - forecast_horizon - temp
    if forecast_horizon == 1:

        for _ in range(count_iter):
            print(f"ТЕКУЩАЯ ИТЕРАЦИЯ {counter} из {count_iter}")
            print(f"Размер глубины прогноза {start_size}")
            mod_value = predict_forecast_horizon(train_sales[0:start_size], forecast_horizon)
            print(f"Прогноз{mod_value}")
            c_s_d_o_s = calculate_standard_deviation_one_step(train_sales[start_size], mod_value)
            print(f"Ошибка{c_s_d_o_s}")
            test_dict[c_s_d_o_s] = start_size
            start_size += 1
            counter += 1

        return test_dict

    if forecast_horizon > 1:
        for _ in range(count_iter):
            print(f"ТЕКУЩАЯ ИТЕРАЦИЯ {counter} из {count_iter}")
            print(f"Размер глубины прогноза {start_size}")
            mod_values = predict_forecast_horizon(train_sales[0:start_size], forecast_horizon)
            print(f"Прогноз{mod_values}")

            c_s_d_m_o_s = calculate_standard_deviation_more_one_step(
                train_sales[start_size:start_size + forecast_horizon],
                mod_values)
            print(f"Ошибка{c_s_d_m_o_s}")
            test_dict[c_s_d_m_o_s] = start_size  # trrain_sales выбрать правльные индксы для слайса
            start_size += 1
            counter += 1
        return test_dict


def show_standard_deviation_one_step(dct_s_d: dict, best_size: int):
    y = np.array(list(dct_s_d.keys()))
    x = np.array(list(dct_s_d.values()))

    plt.plot(x, y)
    mask = y.min()
    plt.scatter(best_size, mask, color='orange', s=40, marker='o')

    plt.title('Standard deviation')
    plt.show()


def calculate_standard_deviation_one_step(real_value, model_predict_value):
    return (real_value - float(model_predict_value)) ** 2


def calculate_standard_deviation_more_one_step(real_values, model_predict):
    result = 0
    for i in range(len(real_values)):
        result += (real_values[i] - float(model_predict[i])) ** 2
    return result / len(real_values)


def predict_forecast_horizon(sales_data: np.ndarray, forecast_h: int):
    '''
    Подбор коэффициентов p d a P D Q для модели Sarima.
    Используя Ментод решателя bfgs' Бройдена-Флетчера-Гольдфарба-Шанно для по оптимизации AIC
    Используя Расширенный тест Дики-Фуллера для нахождения несезонной разности d
    Используя тест Осборна, Чуи, Смита и Бирченхолла (OCSB) для нахождения сезонной разности D
    '''
    smodel = pm.auto_arima(sales_data, start_p=1, start_q=1,
                           test="adf",
                           max_p=10, max_q=10,
                           error_action='ignore',
                           d=None, D=None,
                           trace=True,
                           suppress_warnings=True,
                           maxiter=50,
                           with_intercept=False,
                           seasonal=True, m=12)
    return smodel.predict(forecast_h)


def predict_and_parameters(sales_data: np.ndarray, forecast_h: int):
    '''
    Подбор коэффициентов p d a P D Q для модели Sarima.
    Используя Ментод решателя bfgs' Бройдена-Флетчера-Гольдфарба-Шанно для по оптимизации AIC
    Используя Расширенный тест Дики-Фуллера для нахождения несезонной разности d
    Используя тест Осборна, Чуи, Смита и Бирченхолла (OCSB) для нахождения сезонной разности D
    '''
    smodel = pm.auto_arima(sales_data, start_p=1, start_q=1,
                           test="adf",
                           max_p=10, max_q=10,
                           error_action='ignore',
                           d=None, D=None,
                           trace=True,
                           suppress_warnings=True,
                           maxiter=50,
                           with_intercept=False,
                           seasonal=True, m=12)
    s_par = smodel.to_dict()["seasonal_order"]
    par = smodel.to_dict()["order"]
    dct_with_predict_and_par = {}
    dct_with_predict_and_par["order"] = par
    dct_with_predict_and_par["seasonal_orders"] = s_par
    dct_with_predict_and_par["predict"] = smodel.predict(forecast_h)

    return dct_with_predict_and_par


def get_window_size(dct_for_detected_size):
    return dct_for_detected_size[min(dct_for_detected_size.keys())]


def sliding_window(window_sliding_size: int, train_sales: np.ndarray, forecast_horizon: int):
    test_dict = {}
    counter = 1
    step = 0
    count_iter = len(train_sales) - forecast_horizon - window_sliding_size
    if forecast_horizon == 1:
        for _ in range(count_iter):
            print(f"ТЕКУЩАЯ ИТЕРАЦИЯ {counter} из {count_iter}")
            print(f"Размер глубины прогноза {window_sliding_size}")
            dict_with_par_spar_predict = predict_and_parameters(train_sales[step:window_sliding_size + step],
                                                                forecast_horizon)
            mod_value = dict_with_par_spar_predict["predict"]
            test_dict[calculate_standard_deviation_one_step(
                train_sales[step + window_sliding_size],
                mod_value)] = [dict_with_par_spar_predict["order"], dict_with_par_spar_predict["seasonal_orders"]]
            print(f"Значения которые берутся для прогноза{train_sales[step:window_sliding_size]}")
            print(f"Значения прогноза:{mod_value}")
            print(f"Размер{len(mod_value)}")
            print(
                f"Реальные значения:{train_sales[step + window_sliding_size:step + window_sliding_size + forecast_horizon]}")
            print(f"Размер{train_sales[step + window_sliding_size:step + window_sliding_size + forecast_horizon]}")

            step += 1
            counter += 1

    if forecast_horizon > 1:
        for _ in range(count_iter):
            print(f"ТЕКУЩАЯ ИТЕРАЦИЯ {counter} из {count_iter}")
            print(f"Размер глубины прогноза {window_sliding_size}")
            dict_with_par_spar_predict = predict_and_parameters(train_sales[step:window_sliding_size + step],
                                                                forecast_horizon)
            mod_values = dict_with_par_spar_predict["predict"]
            test_dict[calculate_standard_deviation_more_one_step(
                train_sales[step + window_sliding_size:step + window_sliding_size + forecast_horizon],
                mod_values)] = [dict_with_par_spar_predict["order"], dict_with_par_spar_predict["seasonal_orders"]]
            print(f"Значения которые берутся для прогноза{train_sales[step:window_sliding_size]}")
            print(f"Значения прогноза:{mod_values}")
            print(f"Размер{len(mod_values)}")
            print(
                f"Реальные значения:{train_sales[step + window_sliding_size:step + window_sliding_size + forecast_horizon]}")
            print(f"Размер{train_sales[step + window_sliding_size:step + window_sliding_size + forecast_horizon]}")

            step += 1
            counter += 1

    return test_dict
    # do list отдельную фкнцию при получении прогноза так же возвращать p d q P D Q
    # В словарь запихихивать в ключи значение rmse а в значиение словаря строкой параметры модели


def get_best_model_par(dct_after_train):
    return dct_after_train[min(dct_after_train.keys())]


# ОСНОВНАЯ ПРОГРАММА

FORECAST_DEPTH = 36  # Необходимо минимум брать 3 сезона для старта поиска размера скользящего окна, т.к
# алгоритм не может верно определить сезонную составляющую

FORECAST_HORIZON = 12  # Горизонт прогноза будет выбирать пользователь от 1 до 12 тк.
# выборка для теста финальной модели насчитывает 12 значений

SIZE_VALID_DATA = 12  # Размер выборки для финального теста модели

# Загрузка данных из файла формата csv
df_tree = pd.read_csv("Tree.csv", parse_dates=[0])
print(df_tree)
rd = sm.tsa.seasonal_decompose (df_tree['Value'].values, period = 12) # Период временного ряда равен 12
rd.plot()
plt.show()
ghjhgjh
value_array = df_tree["Value"].to_numpy()
value_array_size = value_array.shape[0]
print(value_array)
print(value_array_size)
size_for_search_and_train = value_array_size - SIZE_VALID_DATA
train, test = model_selection.train_test_split(value_array, train_size=size_for_search_and_train)
print(train)
print(len(train))
print(type(train))

dct_for_s_d = get_standard_deviation_dict(FORECAST_DEPTH, train, FORECAST_HORIZON)
print("Квадрат стандртных отклонений при разных размерах окна: ")
pp(dct_for_s_d)

window_size = get_window_size(dct_for_s_d)
print(f"Размер окна:{window_size}")

show_standard_deviation_one_step(dct_for_s_d, best_size=window_size)
plt.show()




print(f"Размер RMSE при пробеге окном размером {window_size}, и параметры модели:")
dict_after_train = sliding_window(window_size, train, FORECAST_HORIZON)
pp(dict_after_train)

best_model_param = get_best_model_par(dict_after_train)
print(best_model_param)

sarima_model = sm.tsa.statespace.SARIMAX(train,
                                         order=best_model_param[0],
                                         seasonal_order=best_model_param[1],
                                         enforce_stationarity=False,
                                         enforce_invertibility=False,
                                         )
res_sarima_model = sarima_model.fit()
print(res_sarima_model.summary())
res_sarima_model.plot_diagnostics(figsize=(15, 12))
plt.show()

# Do list проверить алгоритм при горизонте прогноза = 1 нужно чтобы совпадало при первом окне
# Получение объекта предсказания и доверительный 95% интервал
print("Размер выборки для теста", test.shape[0])
print("Размер выборки для обучения", train.shape[0])
x = np.arange(start=train.shape[0], stop=len(value_array))
x2 = np.arange(train.shape[0])

plt.plot(x2, train.shape[0], color="green")
plt.plot(x, test, color="red")
plt.plot(x, res_sarima_model.predict(n_periods=test.shape[0]))

plt.title('Actual test samples vs. forecasts')
plt.show()

pred = res_sarima_model.get_prediction(88, dynamic=False)  # prediction
pred_ci = pred.conf_int(0.05)  # Доверительный интервал
"""
model_predict = pred.predicted_mean()
print(model_predict)
"""
print(pred_ci)
'''
ax = y['1990':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='green', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.show()
'''

#DO LIST 1)проверить при горизонте = 1, 2)получение прогноза и доверительного интервала 3)построение графиков
#DO LIST DAY 1)описание алгоритма и блоксхема 2)Разработка интерфейса
