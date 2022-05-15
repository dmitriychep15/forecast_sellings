from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem
from doplom1_3 import Ui_MainWindow
import csv
import pandas as pd
import sys
import statsmodels.api as sm
from matplotlib import pyplot as plt


class My_Window(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_Window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Мой код

        # Переменные
        self.name_current_file = "test_test_Tree.csv"
        self.current_table_data_size = self.get_size_file()
        self.ui.table_data.setRowCount(self.current_table_data_size)
        self.ui.table_data.setColumnCount(2)
        self.ui.table_data.setHorizontalHeaderLabels(
            ("Дата", "Число продаж")
        )
        self.status_accsess_btn_add_new_record = False
        self.ui.add_new_record.setEnabled(self.status_accsess_btn_add_new_record)
        # Сигналы
        self.ui.type_vr.currentTextChanged.connect(self.change_current_text_file)
        self.ui.visualization.clicked.connect(self.display_data)
        self.ui.delete_firs_record.clicked.connect(self.del_first_rec)
        self.ui.add_new_record.clicked.connect(self.add_rec)
        self.ui.input_value_to_add.textChanged.connect(self.change_accses_for_btn_add)

        #тест вывода stl
        self.ui.show_stl.clicked.connect(self.test)

        # Мой код

    # Слоты

    def change_current_text_file(self, text):
        name_v_r = text
        if name_v_r == "Дерево":
            self.name_current_file = "test_test_Tree.csv"
        elif name_v_r == "Алюминий":
            self.name_current_file = "Aluminum.csv"
        else:
            self.name_current_file = "Plastic.csv"
        print(self.name_current_file)

    def count_lines(self, filename, chunk_size=1 << 13):
        with open(filename) as file:
            return sum(chunk.count('\n')
                       for chunk in iter(lambda: file.read(chunk_size), ''))

    def get_size_file(self):
        result = self.count_lines(self.name_current_file) - 1
        print(result)
        return result

    def display_data(self):
        with open(self.name_current_file) as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = list(reader)
            print(data)

        row = 0
        number = 0
        for entry in data:
            col = 0

            for item in entry:
                cellinfo = QTableWidgetItem(item)
                # Только для чтения
                cellinfo.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.ui.table_data.setItem(row, col, cellinfo)
                col += 1

            row += 1

    def del_first_rec(self):
        df_tree = pd.read_csv(self.name_current_file, parse_dates=[0])
        df_tree.drop(index=0, inplace=True)
        self.current_table_data_size -= 1
        self.ui.table_data.setRowCount(self.current_table_data_size)
        df_tree.to_csv(self.name_current_file, index=False)
        self.display_data()

    def next_observation(self, date):
        """Если нужно будет оставить только месяца то убрать в строке преобразования"""
        lst = [int(i) for i in date.split("-")]
        if lst[1] == 12:
            lst[0] += 1
            lst[1] = 1
        else:
            lst[1] += 1
        if lst[1] < 10:
            result = str(lst[0]) + "-" + "0" + str(lst[1]) + "-" + str(lst[2])
        else:
            result = "-".join(map(str, lst))

        return result
#Доконичть функцию
    def change_accses_for_btn_add(self, text):
        if text == "":
            self.status_accsess_btn_add_new_record = True

    def add_rec(self):
        df = pd.read_csv("test_test_Tree.csv")
        last_date = df.iloc[-1]["Date"]
        next_date = self.next_observation(last_date)
        # Дописать обработку если не число

        df.loc[len(df.index)] = [next_date, self.ui.input_value_to_add.text()]
        self.current_table_data_size += 1
        self.ui.table_data.setRowCount(self.current_table_data_size)
        df.to_csv(self.name_current_file, index=False)
        self.display_data()

    def test(self):
        df_tree = pd.read_csv("Tree.csv", parse_dates=[0],index_col=["Date"])
        rd = sm.tsa.seasonal_decompose(df_tree['Value'].values, period=12)  # Период временного ряда равен 12
        rd.plot()
        plt.show()


# Слоты
# Мой код


def application():
    app = QtWidgets.QApplication(sys.argv)
    win = My_Window()
    win.show()
    sys.exit(app.exec_())


application()
