import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QLabel, QLineEdit, QPushButton, QComboBox,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QTextEdit
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
import sys
from datetime import datetime
from xgboost import XGBClassifier
from model import read_dtaset
from xgboost.callback import TrainingCallback


class CustomTrainingCallback(TrainingCallback):
    def __init__(self, status_dialog):
        self.status_dialog = status_dialog

    def after_iteration(self, model, epoch, evals_log):
        if evals_log:
            for metric, values in evals_log.items():
                for dataset, value in values.items():
                    message = f"[Epoch {epoch + 1}] {dataset}-{metric}: {value[-1]}"
                    self.status_dialog.log_message(message)
        return False


class TrainingStatusDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Training Status")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        layout.addWidget(self.text_edit)
        self.setLayout(layout)

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.text_edit.append(f"{timestamp} - {message}")


class XGBoostApp(QWidget):
    def __init__(self):
        super().__init__()

        self.status_dialog = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.model = None
        self.setWindowTitle("XGBoost Hyperparameter Tuning")
        self.setGeometry(100, 100, 600, 700)

        # Initialize training step index
        self.current_step = 0

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(20)

        # App Title
        title_label = QLabel("XGBoost App")
        title_label.setFixedHeight(30)
        title_font = QFont("Arial", 16, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("border: none;")
        main_layout.addWidget(title_label)

        # Train Panel
        train_panel = QVBoxLayout()
        train_panel.setContentsMargins(10, 20, 10, 10)
        train_panel.setSpacing(10)

        # Dataset File Selection
        dataset_layout = QHBoxLayout()
        dataset_path_label = QLabel("Dataset Path:")
        dataset_path_label.setFont(QFont("Arial", 12))
        dataset_path_label.setFixedHeight(30)
        dataset_path_label.setStyleSheet("border: none;")
        dataset_path_label.setFixedWidth(150)
        self.dataset_path_line = QLineEdit(self)
        self.dataset_path_line.setFixedHeight(30)
        self.dataset_path_line.setText("Enter dataset path...")  # Initial value

        dataset_button = QPushButton('...', self)
        dataset_button.setFixedSize(30, 30)
        dataset_button.setStyleSheet(self.common_button_style())
        dataset_button.clicked.connect(self.select_dataset)

        dataset_layout.addWidget(dataset_path_label)
        dataset_layout.addWidget(self.dataset_path_line)
        dataset_layout.addWidget(dataset_button)
        train_panel.addLayout(dataset_layout)

        # Model Save Path Selection
        model_save_layout = QHBoxLayout()
        model_save_path_label = QLabel("Model Save Directory:")
        model_save_path_label.setFont(QFont("Arial", 12))
        model_save_path_label.setFixedHeight(30)
        model_save_path_label.setStyleSheet("border: none;")
        model_save_path_label.setFixedWidth(150)
        self.model_save_path_line = QLineEdit(self)
        self.model_save_path_line.setFixedHeight(30)
        self.model_save_path_line.setText("Enter save directory...")  # Initial value

        model_save_button = QPushButton('...', self)
        model_save_button.setFixedSize(30, 30)
        model_save_button.setStyleSheet(self.common_button_style())
        model_save_button.clicked.connect(self.select_model_save_path)

        model_save_layout.addWidget(model_save_path_label)
        model_save_layout.addWidget(self.model_save_path_line)
        model_save_layout.addWidget(model_save_button)
        train_panel.addLayout(model_save_layout)

        # Hyperparameter Inputs - Row 1
        hyperparams_layout1 = QHBoxLayout()
        max_depth_label = QLabel("Max Depth:")
        max_depth_label.setFont(QFont("Arial", 12))
        max_depth_label.setFixedHeight(30)
        max_depth_label.setStyleSheet("border: none;")
        max_depth_label.setFixedWidth(150)
        self.max_depth_line = QLineEdit(self)
        self.max_depth_line.setFixedHeight(30)
        self.max_depth_line.setFixedWidth(100)
        self.max_depth_line.setText("6")  # Initial value set to 6

        learning_rate_label = QLabel("Learning Rate:")
        learning_rate_label.setFont(QFont("Arial", 12))
        learning_rate_label.setFixedHeight(30)
        learning_rate_label.setStyleSheet("border: none;")
        learning_rate_label.setFixedWidth(150)
        self.learning_rate_line = QLineEdit(self)
        self.learning_rate_line.setFixedHeight(30)
        self.learning_rate_line.setFixedWidth(100)
        self.learning_rate_line.setText("0.1")  # Initial value set to 0.3
        self.learning_rate_line.textChanged.connect(self.check_learning_rate)

        hyperparams_layout1.addWidget(max_depth_label)
        hyperparams_layout1.addWidget(self.max_depth_line)
        hyperparams_layout1.addWidget(learning_rate_label)
        hyperparams_layout1.addWidget(self.learning_rate_line)
        train_panel.addLayout(hyperparams_layout1)

        # Hyperparameter Inputs - Row 2
        hyperparams_layout2 = QHBoxLayout()
        subsample_label = QLabel("Subsample:")
        subsample_label.setFont(QFont("Arial", 12))
        subsample_label.setFixedHeight(30)
        subsample_label.setStyleSheet("border: none;")
        subsample_label.setFixedWidth(150)
        self.subsample_line = QLineEdit(self)
        self.subsample_line.setFixedHeight(30)
        self.subsample_line.setFixedWidth(100)
        self.subsample_line.setText("1.0")  # Initial value set to 1.0

        colsample_bytree_label = QLabel("Colsample by Tree:")
        colsample_bytree_label.setFont(QFont("Arial", 12))
        colsample_bytree_label.setFixedHeight(30)
        colsample_bytree_label.setStyleSheet("border: none;")
        colsample_bytree_label.setFixedWidth(150)
        self.colsample_bytree_line = QLineEdit(self)
        self.colsample_bytree_line.setFixedHeight(30)
        self.colsample_bytree_line.setFixedWidth(100)
        self.colsample_bytree_line.setText("0.8")  # Initial value set to 1.0

        hyperparams_layout2.addWidget(subsample_label)
        hyperparams_layout2.addWidget(self.subsample_line)
        hyperparams_layout2.addWidget(colsample_bytree_label)
        hyperparams_layout2.addWidget(self.colsample_bytree_line)
        train_panel.addLayout(hyperparams_layout2)

        # Evaluation Metrics Dropdown
        eval_metrics_layout = QHBoxLayout()
        eval_metrics_label = QLabel("Evaluation Metrics:")
        eval_metrics_label.setFont(QFont("Arial", 12))
        eval_metrics_label.setFixedHeight(30)
        eval_metrics_label.setStyleSheet("border: none;")
        eval_metrics_label.setFixedWidth(150)
        self.eval_metrics_combo = QComboBox(self)
        self.eval_metrics_combo.setFixedHeight(30)
        self.eval_metrics_combo.addItems(["auc", "error", "logloss"])  # Default selected is "auc"

        eval_metrics_layout.addWidget(eval_metrics_label)
        eval_metrics_layout.addWidget(self.eval_metrics_combo)
        train_panel.addLayout(eval_metrics_layout)

        # Train Button
        train_button = QPushButton("Train", self)
        train_button.setFixedHeight(30)
        train_button.setFixedWidth(140)
        train_button.setStyleSheet(self.common_button_style())
        train_button.clicked.connect(self.start_training)
        train_panel.addWidget(train_button, alignment=Qt.AlignRight)

        # Customize train panel with rounded corners and distinct background color
        train_container = QWidget()
        train_container.setLayout(train_panel)
        train_container.setStyleSheet("""  
            QWidget {  
                border: 1px solid black;  
                border-radius: 8px;  
                background-color: #eef5fb;  
            }  
        """)
        main_layout.addWidget(train_container)

        # Inference Panel
        infer_panel = QVBoxLayout()
        infer_panel.setContentsMargins(10, 20, 10, 10)
        infer_panel.setSpacing(10)

        # Inference Data Selection
        inference_layout = QHBoxLayout()
        input_data_label = QLabel("Input Data Path:")
        input_data_label.setFont(QFont("Arial", 12))
        input_data_label.setFixedHeight(30)
        input_data_label.setStyleSheet("border: none;")
        input_data_label.setFixedWidth(150)
        self.infer_path_line = QLineEdit(self)
        self.infer_path_line.setFixedHeight(30)
        self.infer_path_line.setText("Enter input data path...")  # Initial value
        infer_button = QPushButton('...', self)
        infer_button.setFixedSize(30, 30)
        infer_button.setStyleSheet(self.common_button_style())
        infer_button.clicked.connect(self.select_inference_data)

        inference_layout.addWidget(input_data_label)
        inference_layout.addWidget(self.infer_path_line)
        inference_layout.addWidget(infer_button)
        infer_panel.addLayout(inference_layout)

        # Model Path Selection
        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("Model Path:")
        model_path_label.setFont(QFont("Arial", 12))
        model_path_label.setFixedHeight(30)
        model_path_label.setStyleSheet("border: none;")
        model_path_label.setFixedWidth(150)
        self.model_path_line = QLineEdit(self)
        self.model_path_line.setFixedHeight(30)
        self.model_path_line.setText("Enter model path...")  # Initial value
        model_button = QPushButton('...', self)
        model_button.setFixedSize(30, 30)
        model_button.setStyleSheet(self.common_button_style())
        model_button.clicked.connect(self.select_model_path)

        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(self.model_path_line)
        model_path_layout.addWidget(model_button)
        infer_panel.addLayout(model_path_layout)

        # Inference Button
        inference_button = QPushButton("Inference", self)
        inference_button.setFixedHeight(30)
        inference_button.setFixedWidth(140)
        inference_button.setStyleSheet(self.common_button_style())
        inference_button.clicked.connect(self.perform_inference)
        infer_panel.addWidget(inference_button, alignment=Qt.AlignRight)

        # Table Widget for displaying results
        self.results_table = QTableWidget(12, 9)
        headers = [f'Column {i+1}' for i in range(9)]
        headers.append('result')
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        infer_panel.addWidget(self.results_table)

        # Customize inference panel with rounded corners and distinct background color
        infer_container = QWidget()
        infer_container.setLayout(infer_panel)
        infer_container.setStyleSheet("""  
            QWidget {  
                border: 1px solid black;  
                border-radius: 8px;  
                background-color: #eef5fb;  
            }  
        """)
        main_layout.addWidget(infer_container)

        # Setting the layout for the whole window
        self.setLayout(main_layout)

    def common_button_style(self):
        return """  
            QPushButton {  
                background-color: black;  
                color: white;  
                border-radius: 5px;  
            }  
            QPushButton:hover {  
                background-color: #2E86C1;  
            }  
            QPushButton:pressed {  
                background-color: #1B4F72;  
            }  
        """

    def select_dataset(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Select Dataset CSV")
        if file_path:
            self.dataset_path_line.setText(file_path)

    def select_model_save_path(self):
        file_dialog = QFileDialog(self)
        dir_path = file_dialog.getExistingDirectory(self, "Select Model Save Directory")
        if dir_path:
            self.model_save_path_line.setText(dir_path)

    def check_learning_rate(self):
        try:
            learning_rate = float(self.learning_rate_line.text())
            if not (0 < learning_rate <= 0.5):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Learning rate must be between 0 and 0.5.")
            self.learning_rate_line.clear()

    def select_inference_data(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Select Input Data CSV")
        if file_path:
            self.infer_path_line.setText(file_path)

    def select_model_path(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Select Model File")
        if file_path:
            self.model_path_line.setText(file_path)

    def start_training(self):

        self.status_dialog = TrainingStatusDialog()
        self.status_dialog.show()
        # Set up the custom callback for logging
        custom_callback = CustomTrainingCallback(self.status_dialog)

        self.model = XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=int(self.max_depth_line.text()),
            learning_rate=float(self.learning_rate_line.text()),
            subsample=float(self.subsample_line.text()),
            colsample_bytree=float(self.colsample_bytree_line.text()),
            eval_metric=self.eval_metrics_combo.itemText(self.eval_metrics_combo.currentIndex()),
            callbacks=[custom_callback]
        )

        self.X_train, self.X_val, self.y_train, self.y_val = read_dtaset(self.dataset_path_line.text())

        # Train the model with the custom callback  
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False  # Disable default prints
        )

        current_datetime = datetime.now()
        timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
        filename = f"model_{timestamp}.json"

        self.model.save_model(self.model_save_path_line.text() + '/' + filename)

    def perform_inference(self):

        path_infer_data = self.infer_path_line.text()

        if not path_infer_data.endswith('.csv'):
            QMessageBox.warning("File Format Error", "The selected file is not a CSV.")
            return

        data = pd.read_csv(path_infer_data)

        infer_data = data[data.iloc[:, 0].str.len() == 1]

        infer_data['WAX_wane'] = infer_data['WAX_wane'].apply(lambda x: 1 if 'wax' in x.lower() else 0).astype(int)

        feed_data = infer_data[['WAX_wane', 'Column 1', 'Column 2', 'Column 3', 'Column 4', 'Column 5', 'Column 6', 'Column 7', 'Column 8']]
        feed_data = feed_data.reset_index(drop=True)

        self.model.load_model(self.model_path_line.text())
        feed_data['result'] = self.model.predict_proba(feed_data)[:, 1]

        # Display feed_data in the table
        self.results_table.setRowCount(feed_data.shape[0])
        self.results_table.setColumnCount(feed_data.shape[1])

        for row_idx, row in feed_data.iterrows():
            for col_idx, value in enumerate(row):
                self.results_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = XGBoostApp()
    window.show()
    sys.exit(app.exec())
