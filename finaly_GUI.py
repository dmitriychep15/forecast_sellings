from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox
from diploma import Ui_MainWindow
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
        self.current_file_name = "test_test_Tree.csv"
        self.next_file_name = ""
        self.current_table_data_size = self.count_lines(self.current_file_name) - 1
        self.ui.table_data.setRowCount(self.current_table_data_size)
        self.ui.table_data.setColumnCount(2)
        self.ui.table_data.setHorizontalHeaderLabels(
            ("Дата", "Число продаж")
        )


        self.previous_data_state = None

        self.enable_btn_visualize_vr_data = True

        self.error_type = QMessageBox()
        self.error_type.setWindowTitle("Ошибка - некорректный формат")
        self.error_type.setText("Некорретный формат введенных данных")
        self.error_type.setInformativeText("Требуется ввести целое число")
        self.error_type.setIcon(QMessageBox.Warning)
        self.error_type.setStandardButtons(QMessageBox.Cancel|QMessageBox.Ok)

        self.error_many_recs = QMessageBox()
        self.error_many_recs.setWindowTitle("Ошибка - слишком много записей")
        self.error_many_recs.setText("Ошибка - выбрано много записей")
        self.error_many_recs.setInformativeText("Выберите одну запись для удаления")
        self.error_many_recs.setIcon(QMessageBox.Warning)
        self.error_many_recs.setStandardButtons(QMessageBox.Cancel|QMessageBox.Ok)

        self.error_no_rec = QMessageBox()
        self.error_no_rec.setWindowTitle("Ошибка - запись не найдена")
        self.error_no_rec.setText("Ошибка - записи под таким номером не существует")
        self.error_no_rec.setInformativeText("Введите другое значение")
        self.error_no_rec.setIcon(QMessageBox.Warning)
        self.error_no_rec.setStandardButtons(QMessageBox.Cancel|QMessageBox.Ok)

        self.error_invalid_sell_value = QMessageBox()
        self.error_invalid_sell_value.setWindowTitle("Ошибка - некорретное число")
        self.error_invalid_sell_value.setText("Ошибка - число продаж не может быть отрицательным")
        self.error_invalid_sell_value.setInformativeText("Введите значение больше нуля")
        self.error_invalid_sell_value.setIcon(QMessageBox.Warning)
        self.error_invalid_sell_value.setStandardButtons(QMessageBox.Cancel|QMessageBox.Ok)

        # Сигналы
        self.ui.cmb_box_type_vr.currentTextChanged.connect(self.change_current_text_file)
        self.ui.btn_visualize_vr_data.clicked.connect(self.display_data)
        self.ui.btn_add_new_record.clicked.connect(self.add_rec)
        self.ui.btn_edit_record.clicked.connect(self.edit_rec)
        self.ui.btn_delete_record.clicked.connect(self.delete_rec)

        #тест вывода stl
        self.ui.btn_show_stl.clicked.connect(self.test)

    # Слоты

    def change_current_text_file(self, text):
        name_v_r = text
        self.next_file_name = name_v_r

        if name_v_r == "Дерево":
            self.current_file_name = "test_test_Tree.csv"
        elif name_v_r == "Алюминий":
            self.current_file_name = "Aluminum.csv"
        else:
            self.current_file_name = "Plastic.csv"
        print(self.current_file_name)

    def count_lines(self, filename, chunk_size=1 << 13):
        with open(filename) as file:
            return sum(chunk.count('\n')
                       for chunk in iter(lambda: file.read(chunk_size), ''))

    def display_data(self):
        with open(self.current_file_name) as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = list(reader)
            print(data)

        row = 0
        # number = 0
        for entry in data:
            col = 0

            for item in entry:
                cellinfo = QTableWidgetItem(item)
                # Только для чтения
                cellinfo.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.ui.table_data.setItem(row, col, cellinfo)
                col += 1

            row += 1

    def delete_rec(self):
        recs_to_delete = 0
        if self.ui.value_to_delete.text():
            recs_to_delete += 1
        if self.ui.chBox_del_first_rec.isChecked():
            recs_to_delete += 1
        if self.ui.chBox_del_last_rec.isChecked():
            recs_to_delete += 1
        if recs_to_delete > 1:
            self.error_many_recs.exec_()
            return None
        last_rec_index = self.current_table_data_size - 1
        def del_process(ind):
            df_tree = pd.read_csv(self.current_file_name, parse_dates=[0])
            df_tree.drop(index=ind, inplace=True)
            self.current_table_data_size -= 1
            self.ui.table_data.setRowCount(self.current_table_data_size)
            df_tree.to_csv(self.current_file_name, index=False)
            self.display_data()

        if self.ui.value_to_delete.text():
            try:
                int(self.ui.value_to_delete.text())
            except ValueError:
                self.error_type.exec_()
                return None
            self.ui.chBox_del_first_rec.isEnabled = False
            self.ui.chBox_del_last_rec.isEnabled = False
            ind = int(self.ui.value_to_delete.text()) - 1
            del_process(ind=ind)

        elif self.ui.chBox_del_first_rec.isChecked():
            print('hi')
            self.ui.chBox_del_last_rec.isEnabled = False
            del_process(ind=0)

        elif self.ui.chBox_del_last_rec.isChecked():
            self.ui.chBox_del_first_rec.isEnabled = False
            del_process(ind=last_rec_index)

    def edit_rec(self):
        try:
            int(self.ui.rec_to_edit.text())
            int(self.ui.value_to_edit.text())
        except ValueError:
            self.error_type.exec_()
            print('hi')
            return None
       
        if self.ui.rec_to_edit.text() and self.ui.value_to_edit.text():
            rec_ind = int(self.ui.rec_to_edit.text()) - 1
            rec_value = self.ui.value_to_edit.text()
            df = pd.read_csv(self.current_file_name)
            if rec_ind <= 0:
                self.error_no_rec.exec_()
                return None
            elif int(rec_value) < 0:
                self.error_invalid_sell_value.exec_()
                return None
            try:
                df.loc[rec_ind] = [df.iloc[rec_ind]["Date"], rec_value]
                df.to_csv(self.current_file_name, index=False)
                self.display_data()
                print('hello')
            except IndexError:
                self.error_no_rec.exec_()
                return None


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

    def add_rec(self):
        if self.ui.value_to_add.text():
            try:
                int(self.ui.value_to_add.text())
            except ValueError:
                self.error_type.exec_()
                return None

            df = pd.read_csv(self.current_file_name)
            last_date = df.iloc[-1]["Date"]
            next_date = self.next_observation(last_date)

            df.loc[len(df.index)] = [next_date, self.ui.value_to_add.text()]
            self.current_table_data_size += 1
            self.ui.table_data.setRowCount(self.current_table_data_size)
            df.to_csv(self.current_file_name, index=False)
            self.display_data()

    def test(self):
        df_tree = pd.read_csv(self.current_file_name, parse_dates=[0],index_col=["Date"])
        rd = sm.tsa.seasonal_decompose(df_tree['Value'].values, period=12)  # Период временного ряда равен 12
        rd.plot()
        plt.show()


def application():
    app = QtWidgets.QApplication(sys.argv)
    win = My_Window()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    application()
