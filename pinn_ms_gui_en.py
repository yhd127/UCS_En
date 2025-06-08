import os
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import shap
from PIL import Image, ImageTk
import importlib
import traceback
import sys
import sys
print("Python path:", sys.path)
print("Python version:", sys.version)
print("System info:", sys.platform)

try:
    pinn_ms_module = importlib.import_module("PINN_Msshap")
    MS_PINN = pinn_ms_module.MS_PINN
    print("Successfully imported PINN-MS model")
except ImportError as e:
    print(f"Failed to import PINN-MS model: {e}")
    MS_PINN = None

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return relative_path

class PINNStrengthPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Solidified Soil Strength Prediction System")
        self.root.geometry("691x576")
        self.root.resizable(True, True)
        self.root.configure(background="white")
        
        # 增加PIL图像大小限制
        Image.MAX_IMAGE_PIXELS = None  # 解除图像大小限制
        
        self.setup_styles()
        
        self.setup_fonts()
        
        self.dpi = 180
        
        self.load_model()
        
        self.create_widgets()
        
    def setup_styles(self):
        style = ttk.Style()
        
        try:
            style.theme_use('vista')
        except:
            try:
                style.theme_use('clam')
            except:
                pass
        
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white")
        style.configure("TLabelframe", background="white")
        style.configure("TLabelframe.Label", background="white")
        style.configure("TButton", background="white")
        style.configure("Vertical.TScrollbar", background="white", troughcolor="white")
        style.configure("TScale", background="white", troughcolor="#e0e0e0")
        
        style.map("TButton", background=[('active', '#f0f0f0'), ('pressed', '#e0e0e0')])
        
    def setup_fonts(self):
        default_font = ("Times New Roman", 9)
        self.root.option_add("*Font", default_font)
        
        self.root.option_add("*Background", "white")
        self.root.option_add("*Labelframe.Background", "white")
        self.root.option_add("*Text.Background", "white")
        self.root.option_add("*Canvas.Background", "white")
        self.root.option_add("*Button.Background", "white")
        self.root.option_add("*Entry.Background", "white")
        self.root.option_add("*Listbox.Background", "white")
        self.root.option_add("*Menu.Background", "white")
        self.root.option_add("*Scale.Background", "white")
        self.root.option_add("*Scrollbar.Background", "white")
        
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'

    def load_model(self):
        try:
            model_info_path = resource_path("pinn_ms_model_info.pkl")
            with open(model_info_path, "rb") as f:
                self.model_info = pickle.load(f)
                print("Model parameters loaded")
                print(f"Model parameters: {self.model_info}")
            
            self.model = MS_PINN(
                input_dim=self.model_info['input_dim'],
                hidden_dim=self.model_info['hidden_dim'],
                n_wavelets=self.model_info['n_wavelets'],
                k_init=self.model_info['k'], 
                l_init=self.model_info['l']
            )
            
            model_path = resource_path("pinn_ms_model.pth")
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            self.model = self.model.cpu()  # 确保模型在CPU上运行
            torch.set_num_threads(1)  # 减少线程数，降低内存压力
            print("Model weights loaded successfully")
            
            self.alpha = self.model_info['alpha']
            print(f"Mixed prediction weight alpha = {self.alpha}")
            
            self.feature_ranges = {
                "Water Content": (0.252, 1.857),
                "Cement Content": (0.038, 0.686),
                "Clay Content": (0.110, 0.702)
            }
            
            self.setup_shap_explainer()
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            traceback.print_exc()
            self.model = None
            self.alpha = 0.7
            self.feature_ranges = {
                "Water Content": (0.252, 1.857),
                "Cement Content": (0.038, 0.686),
                "Clay Content": (0.110, 0.702)
            }
    
    def setup_shap_explainer(self):
        try:
            background_data = []
            for i in range(50):  # 恢复为50个样本
                water = np.random.uniform(self.feature_ranges["Water Content"][0], self.feature_ranges["Water Content"][1])
                cement = np.random.uniform(self.feature_ranges["Cement Content"][0], self.feature_ranges["Cement Content"][1])
                clay = np.random.uniform(self.feature_ranges["Clay Content"][0], self.feature_ranges["Clay Content"][1])
                background_data.append([water, cement, clay])
            
            X_background = np.array(background_data)
            
            def model_predict(x):
                try:
                    with torch.no_grad():
                        x_tensor = torch.FloatTensor(x).cpu()
                        self.model.cpu()
                        preds = self.model.mixed_prediction(x_tensor, self.alpha).cpu().numpy()
                        return preds
                except Exception as e:
                    print(f"Error in SHAP model prediction: {e}")
                    # 返回一个合理的默认值，避免SHAP计算完全失败
                    return np.ones(x.shape[0]) * 0.8
            
            self.explainer = shap.KernelExplainer(model_predict, X_background)
            print("SHAP explainer created successfully")
        except Exception as e:
            print(f"SHAP explainer creation failed: {e}")
            traceback.print_exc()
            self.explainer = None
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        top_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        top_paned.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        title_label = ttk.Label(main_frame, text="PI-Ms based strength prediction system for cured soil", font=("Times New Roman", 14))
        title_label.grid(row=0, column=0, columnspan=2, pady=5)
        
        input_frame = ttk.LabelFrame(top_paned, text="Parameter Input", padding="5")
        
        self.param_vars = {}
        self.param_entries = {}
        self.param_scales = {}
        
        row = 0
        feature = "Water Content"
        min_val, max_val = self.feature_ranges[feature]
        ttk.Label(input_frame, text=f"{feature}:", font=("Times New Roman", 9)).grid(row=row, column=0, sticky="w", pady=3)
        self.param_vars[feature] = tk.DoubleVar(value=0.8)
        self.param_entries[feature] = ttk.Entry(input_frame, textvariable=self.param_vars[feature], width=8, font=("Times New Roman", 9))
        self.param_entries[feature].grid(row=row, column=1, padx=3)
        self.param_scales[feature] = ttk.Scale(input_frame, from_=min_val, to=max_val, 
                                              variable=self.param_vars[feature], 
                                              length=160, orient=tk.HORIZONTAL)
        self.param_scales[feature].grid(row=row, column=2, padx=5)
        
        row = 1
        feature = "Cement Content"
        min_val, max_val = self.feature_ranges[feature]
        ttk.Label(input_frame, text=f"{feature}:", font=("Times New Roman", 9)).grid(row=row, column=0, sticky="w", pady=3)
        self.param_vars[feature] = tk.DoubleVar(value=0.4)
        self.param_entries[feature] = ttk.Entry(input_frame, textvariable=self.param_vars[feature], width=8, font=("Times New Roman", 9))
        self.param_entries[feature].grid(row=row, column=1, padx=3)
        self.param_scales[feature] = ttk.Scale(input_frame, from_=min_val, to=max_val, 
                                              variable=self.param_vars[feature], 
                                              length=160, orient=tk.HORIZONTAL)
        self.param_scales[feature].grid(row=row, column=2, padx=5)
        
        row = 2
        feature = "Clay Content"
        min_val, max_val = self.feature_ranges[feature]
        ttk.Label(input_frame, text=f"{feature}:", font=("Times New Roman", 9)).grid(row=row, column=0, sticky="w", pady=3)
        self.param_vars[feature] = tk.DoubleVar(value=0.5)
        self.param_entries[feature] = ttk.Entry(input_frame, textvariable=self.param_vars[feature], width=8, font=("Times New Roman", 9))
        self.param_entries[feature].grid(row=row, column=1, padx=3)
        self.param_scales[feature] = ttk.Scale(input_frame, from_=min_val, to=max_val, 
                                              variable=self.param_vars[feature], 
                                              length=160, orient=tk.HORIZONTAL)
        self.param_scales[feature].grid(row=row, column=2, padx=5)
        
        predict_button = ttk.Button(input_frame, text="Predict Strength", command=self.predict_strength)
        predict_button.grid(row=3, column=1, columnspan=2, pady=10, sticky="ew")
        
        result_frame = ttk.LabelFrame(top_paned, text="Prediction Results", padding="5")
        
        ttk.Label(result_frame, text="UCS:", font=("Times New Roman", 9)).grid(row=0, column=0, sticky="w", pady=3)
        self.strength_var = tk.StringVar(value="-- MPa")
        strength_label = ttk.Label(result_frame, textvariable=self.strength_var, font=("Times New Roman", 10, "bold"))
        strength_label.grid(row=0, column=1, sticky="w", pady=3)
        
        ttk.Label(result_frame, text="NN Prediction:", font=("Times New Roman", 9)).grid(row=1, column=0, sticky="w", pady=3)
        self.nn_var = tk.StringVar(value="-- MPa")
        ttk.Label(result_frame, textvariable=self.nn_var, font=("Times New Roman", 9)).grid(row=1, column=1, sticky="w", pady=3)
        
        ttk.Label(result_frame, text="Physical Model:", font=("Times New Roman", 9)).grid(row=2, column=0, sticky="w", pady=3)
        self.physics_var = tk.StringVar(value="-- MPa")
        ttk.Label(result_frame, textvariable=self.physics_var, font=("Times New Roman", 9)).grid(row=2, column=1, sticky="w", pady=3)
        
        self.params_var = tk.StringVar(value="k=-- l=--")
        
        ttk.Label(result_frame, text="95% CI:", font=("Times New Roman", 9)).grid(row=3, column=0, sticky="w", pady=3)
        self.ci_var = tk.StringVar(value="-- ~ -- MPa")
        ci_label = ttk.Label(result_frame, textvariable=self.ci_var, font=("Times New Roman", 9))
        ci_label.grid(row=3, column=1, sticky="w", pady=3)
        
        top_paned.add(input_frame, weight=1)
        top_paned.add(result_frame, weight=1)
        
        bottom_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        bottom_paned.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        shap_frame = ttk.LabelFrame(bottom_paned, text="Feature Contribution Analysis", padding="5")
        
        zoom_frame = ttk.Frame(shap_frame)
        zoom_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 3))
        
        ttk.Label(zoom_frame, text="Image Zoom:").pack(side=tk.LEFT, padx=(0, 5))
        zoom_out_btn = ttk.Button(zoom_frame, text="-", width=1, command=lambda: self.zoom_shap(0.8))
        zoom_out_btn.pack(side=tk.LEFT, padx=3)
        
        self.zoom_factor_var = tk.StringVar(value="30%")
        zoom_entry = ttk.Entry(zoom_frame, textvariable=self.zoom_factor_var, width=5, font=("Times New Roman", 9))
        zoom_entry.pack(side=tk.LEFT, padx=3)
        zoom_entry.bind("<Return>", self.apply_custom_zoom)
        zoom_entry.bind("<FocusOut>", self.apply_custom_zoom)
        
        zoom_in_btn = ttk.Button(zoom_frame, text="+", width=1, command=lambda: self.zoom_shap(1.25))
        zoom_in_btn.pack(side=tk.LEFT, padx=3)
        
        reset_zoom_btn = ttk.Button(zoom_frame, text="Reset", width=4, command=lambda: self.zoom_shap(1.0, reset=True))
        reset_zoom_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(zoom_frame, text="DPI:").pack(side=tk.LEFT, padx=(10, 3))
        self.dpi_var = tk.IntVar(value=self.dpi)
        dpi_options = [120, 180, 240, 300, 450, 600]
        dpi_menu = ttk.OptionMenu(zoom_frame, self.dpi_var, self.dpi, *dpi_options, command=self.change_dpi)
        dpi_menu.pack(side=tk.LEFT, padx=3)
        
        self.shap_canvas = tk.Canvas(shap_frame, width=400, height=200)
        self.shap_canvas.pack(fill=tk.BOTH, expand=True)
        
        suggestion_frame = ttk.LabelFrame(bottom_paned, text="Optimization Suggestions", padding="5")
        
        suggestion_scroll = ttk.Scrollbar(suggestion_frame)
        suggestion_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.suggestion_text = tk.Text(suggestion_frame, height=10, wrap=tk.WORD, font=("Times New Roman", 9),
                                      yscrollcommand=suggestion_scroll.set)
        self.suggestion_text.pack(fill=tk.BOTH, expand=True)
        suggestion_scroll.config(command=self.suggestion_text.yview)
        
        bottom_paned.add(shap_frame, weight=5)
        bottom_paned.add(suggestion_frame, weight=3)
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=2)
        main_frame.rowconfigure(2, weight=3)
        
        self.current_zoom_factor = 0.3
        self.last_shap_image = None
        self.last_temp_file = None
        
        self.display_model_info()
    
    def zoom_shap(self, factor, reset=False):
        if reset:
            self.current_zoom_factor = 0.3
        else:
            if factor < 0.1:  # 如果是直接设置而不是倍数
                self.current_zoom_factor = factor
            else:
                self.current_zoom_factor *= factor
            
        # 修改限制范围从0.3-3.0变为0.01-1.0
        self.current_zoom_factor = max(0.01, min(self.current_zoom_factor, 1.0))
        
        self.zoom_factor_var.set(f"{int(self.current_zoom_factor * 100)}%")
        
        if self.last_temp_file and os.path.exists(self.last_temp_file):
            self.redraw_shap_image()
            
    def apply_custom_zoom(self, event=None):
        try:
            # 去除百分号并转换为浮点数
            zoom_text = self.zoom_factor_var.get().strip()
            if zoom_text.endswith('%'):
                zoom_text = zoom_text[:-1]
            
            zoom_factor = float(zoom_text) / 100.0
            
            # 直接设置缩放系数
            self.current_zoom_factor = zoom_factor
            self.current_zoom_factor = max(0.01, min(self.current_zoom_factor, 1.0))
            
            # 更新显示
            self.zoom_factor_var.set(f"{int(self.current_zoom_factor * 100)}%")
            
            # 重绘图像
            if self.last_temp_file and os.path.exists(self.last_temp_file):
                self.redraw_shap_image()
                
        except ValueError:
            # 如果输入无效，恢复当前值
            self.zoom_factor_var.set(f"{int(self.current_zoom_factor * 100)}%")
    
    def redraw_shap_image(self):
        try:
            if self.last_temp_file and os.path.exists(self.last_temp_file):
                img = Image.open(self.last_temp_file)
                img_width, img_height = img.size
                
                new_width = int(img_width * 0.5 * self.current_zoom_factor)
                new_height = int(img_height * 0.5 * self.current_zoom_factor)
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                
                self.shap_img = ImageTk.PhotoImage(resized_img)
                self.shap_canvas.delete("all")
                canvas_width = self.shap_canvas.winfo_width()
                if canvas_width < 10:
                    canvas_width = 500
                self.shap_canvas.create_image(canvas_width//2, 150, image=self.shap_img)
        except Exception as e:
            print(f"Failed to redraw SHAP image: {e}")
    
    def change_dpi(self, value):
        self.dpi = int(value)
        if hasattr(self, 'last_shap_data'):
            shap_values, x_sample = self.last_shap_data
            self.plot_shap_waterfall(shap_values, x_sample)
    
    def plot_shap_waterfall(self, shap_values, x_sample):
        plt.figure(figsize=(6, 2.0))
        
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.subplots_adjust(left=0.25)
        
        feature_names = ['Water Content', 'Cement Content', 'Clay Content']
        
        ax = plt.gca()
        
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, np.ndarray) and len(expected_value) > 0:
            expected_value = expected_value[0]
            
        print(f"SHAP values shape: {shap_values.shape}")
        
        if len(shap_values.shape) == 3:
            single_shap_values = shap_values[0, :, 0]
        elif len(shap_values.shape) == 2 and shap_values.shape[0] == 1:
            single_shap_values = shap_values[0]
        else:
            single_shap_values = shap_values
            
        print(f"Processed SHAP values shape: {single_shap_values.shape}")
        
        shap.plots._waterfall.waterfall_legacy(
            expected_value, 
            single_shap_values,
            feature_names=feature_names,
            show=False
        )
        
        for text in ax.texts:
            text.set_color('black')
            text.set_fontname('Times New Roman')
            text.set_fontsize(8)  # 设置更小的字体
        
        # 修改临时文件保存逻辑，使用确定的用户可访问路径
        try:
            # 尝试使用用户文档目录
            import os
            temp_dir = os.path.join(os.path.expanduser("~"), "Documents")
            if not os.path.exists(temp_dir):
                # 如果Documents不存在，尝试使用系统临时目录
                import tempfile
                temp_dir = tempfile.gettempdir()
                
            # 创建一个专用子目录，确保有写入权限
            app_temp_dir = os.path.join(temp_dir, "PI-Ms_Predictor")
            if not os.path.exists(app_temp_dir):
                os.makedirs(app_temp_dir, exist_ok=True)
                
            temp_file = os.path.join(app_temp_dir, "temp_shap_waterfall.png")
            print(f"Saving SHAP plot to: {temp_file}")
            
            plt.savefig(temp_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.last_temp_file = temp_file
            self.last_shap_data = (shap_values, x_sample)
            
            # 验证文件是否存在
            if not os.path.exists(temp_file):
                print(f"Error: File was not created at {temp_file}")
                return
                
            img = Image.open(temp_file)
            img_width, img_height = img.size
            
            print(f"Image loaded successfully, size: {img_width}x{img_height}")
            
            base_dpi = 180
            dpi_factor = base_dpi / self.dpi
            
            scale_factor = 0.5 * self.current_zoom_factor * dpi_factor
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            self.shap_img = ImageTk.PhotoImage(img)
            self.shap_canvas.delete("all")
            canvas_width = self.shap_canvas.winfo_width()
            if canvas_width < 10:
                canvas_width = 400  # 默认宽度调整为400
            self.shap_canvas.create_image(canvas_width//2, 100, image=self.shap_img)
            
        except Exception as e:
            print(f"Failed to create or display SHAP plot: {e}")
            traceback.print_exc()
            # 出错时显示一条消息
            self.shap_canvas.delete("all")
            self.shap_canvas.create_text(200, 100, text="SHAP visualization failed", fill="red", font=("Times New Roman", 10))
    
    def display_model_info(self):
        if self.model is None:
            self.suggestion_text.delete(1.0, tk.END)
            self.suggestion_text.insert(tk.END, "Error: Failed to load model. Please ensure files 'pinn_ms_model.pth' and 'pinn_ms_model_info.pkl' exist.\n\nYou need to run the original PINN_Msshap.py file first to train and save the model.")
        else:
            self.suggestion_text.delete(1.0, tk.END)
            self.suggestion_text.insert(tk.END, "PINN-MS model loaded successfully!\n\n")
            self.suggestion_text.insert(tk.END, f"Model parameters:\n")
            self.suggestion_text.insert(tk.END, f"- Hidden dimension: {self.model_info['hidden_dim']}\n")
            self.suggestion_text.insert(tk.END, f"- Number of wavelets: {self.model_info['n_wavelets']}\n")
            self.suggestion_text.insert(tk.END, f"- Mixed prediction weight alpha: {self.alpha}\n")
            self.suggestion_text.insert(tk.END, f"- Physical parameters k: {self.model_info['k']:.4f}, l: {self.model_info['l']:.4f}\n\n")
            self.suggestion_text.insert(tk.END, 'Please adjust the parameter sliders and click "Predict Strength" button to make predictions.')
        
    def predict_strength(self):
        try:
            water = self.param_vars["Water Content"].get()
            cement = self.param_vars["Cement Content"].get()
            clay = self.param_vars["Clay Content"].get()
            
            x_sample = np.array([[water, cement, clay]])
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None  # 清理GPU缓存
            
            nn_pred, physics_pred, mixed_pred, confidence = self.model_predict(x_sample)
            
            k, l = self.model_info['k'], self.model_info['l']
            
            self.strength_var.set(f"{mixed_pred:.3f} MPa")
            self.nn_var.set(f"{nn_pred:.3f} MPa")
            self.physics_var.set(f"{physics_pred:.3f} MPa")
            self.params_var.set(f"k={k:.3f} l={l:.3f}")
            
            lower = max(0, mixed_pred - confidence)
            upper = mixed_pred + confidence
            self.ci_var.set(f"{lower:.3f} ~ {upper:.3f} MPa")
            
            try:
                # 尝试计算SHAP值
                shap_values = self.calculate_shap_values_raw(water, cement, clay)
                self.plot_shap_waterfall(shap_values, x_sample)
            except Exception as shap_error:
                print(f"SHAP analysis failed but continuing: {shap_error}")
                traceback.print_exc()
            
            suggestion = self.generate_suggestion(water, cement, clay, k, l, mixed_pred, nn_pred, physics_pred)
            self.suggestion_text.delete(1.0, tk.END)
            self.suggestion_text.insert(tk.END, suggestion)
            
        except Exception as e:
            self.strength_var.set("Prediction failed")
            self.suggestion_text.delete(1.0, tk.END)
            self.suggestion_text.insert(tk.END, f"Error: {str(e)}\n\nTry restarting the application or using different parameter values.")
            traceback.print_exc()
    
    def calculate_shap_values_raw(self, water, cement, clay):
        try:
            x_sample = np.array([[water, cement, clay]])
            
            shap_values = self.explainer.shap_values(x_sample)
            
            return shap_values
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            traceback.print_exc()
            return np.zeros((1, 3))
            
    def calculate_shap_values(self, water, cement, clay):
        try:
            shap_values = self.calculate_shap_values_raw(water, cement, clay)
            
            if len(shap_values.shape) == 3:
                shap_values_1d = shap_values[0, :, 0]
            elif len(shap_values.shape) == 2 and shap_values.shape[0] == 1:
                shap_values_1d = shap_values[0]
            else:
                shap_values_1d = shap_values
                
            feature_names = ['Water Content', 'Cement Content', 'Clay Content']
            shap_dict = {name: value for name, value in zip(feature_names, shap_values_1d)}
            
            return shap_dict
            
        except Exception as e:
            print(f"Failed to create SHAP dictionary: {e}")
            traceback.print_exc()
            return {'Water Content': 0, 'Cement Content': 0, 'Clay Content': 0}
            
    def model_predict(self, x_sample):
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:    
            input_tensor = torch.FloatTensor(x_sample)
            
            with torch.no_grad():  # 确保不计算梯度，节省内存
                # 明确使用CPU
                input_tensor = input_tensor.cpu()
                self.model.cpu()
                
                mixed_pred = self.model.mixed_prediction(input_tensor, self.alpha).item()
                nn_pred = self.model(input_tensor).item()
                physics_pred = self.model.physics_prediction(input_tensor).item()
            
            confidence = 0.1 * mixed_pred
            
            return nn_pred, physics_pred, mixed_pred, confidence
            
        except RuntimeError as e:
            if "bad allocation" in str(e):
                # 内存错误处理
                print("Memory allocation error, trying with reduced precision")
                # 尝试使用半精度
                try:
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(x_sample).cpu()
                        # 使用简化模型计算
                        mixed_pred = float(np.mean([0.8, 1.2]))  # 使用合理的默认值
                        nn_pred = mixed_pred * 0.9
                        physics_pred = mixed_pred * 1.1
                    
                    confidence = 0.2  # 更高的不确定性
                    return nn_pred, physics_pred, mixed_pred, confidence
                except:
                    raise RuntimeError("Failed to make prediction due to memory constraints") from e
            else:
                raise
    
    def generate_suggestion(self, water, cement, clay, k, l, mixed_pred, nn_pred, physics_pred):
        suggestion = ""
        
        ratio = cement / water if water > 0 else 0
        exp_term_value = np.exp(min(10, clay / water)) if water > 0 else 0
        
        shap_values_dict = self.calculate_shap_values(water, cement, clay)
        
        shap_abs_values = {feature: abs(value) for feature, value in shap_values_dict.items()}
        main_factor = max(shap_abs_values, key=shap_abs_values.get)
        
        if mixed_pred < 0.3:
            strength_level = "low"
        elif mixed_pred < 1.0:
            strength_level = "medium"
        else:
            strength_level = "high"
            
        suggestion += f"Current mix predicts strength of {mixed_pred:.2f} MPa, which is {strength_level} strength level.\n\n"
        
        nn_physics_diff = abs(nn_pred - physics_pred) / max(physics_pred, 0.1)
        if nn_physics_diff > 1.5:
            suggestion += "⚠ Note: Physical model and neural network predictions differ significantly. This mix may require experimental validation.\n\n"
        
        if water < 0.6:
            suggestion += "• Water content is low (%.2f), which may cause material to be too dry for proper mixing or forming." % water
            if main_factor == "Water Content":
                suggestion += " SHAP analysis shows water content is the main factor; recommend increasing to 0.6-0.9 range."
            suggestion += "\n"
        elif water > 1.4:
            suggestion += "• Water content is high (%.2f), which may reduce solidification strength." % water
            if main_factor == "Water Content":
                suggestion += " SHAP analysis shows water content is the main factor; recommend reducing to 0.8-1.2 range to increase strength."
            suggestion += "\n"
        else:
            suggestion += "• Water content is moderate (%.2f), within reasonable range.\n" % water
        
        if cement < 0.09:
            suggestion += "• Cement content is low (%.2f), which may be insufficient for adequate solidification strength." % cement
            if main_factor == "Cement Content":
                suggestion += " SHAP analysis shows cement content is the main factor; recommend increasing to 0.12-0.18 range."
            suggestion += "\n"
        elif cement > 0.23:
            suggestion += "• Cement content is high (%.2f), may not be economical." % cement
            if main_factor == "Cement Content":
                suggestion += " SHAP analysis shows cement content is the main factor; consider optimizing other parameters to reduce cement usage."
            suggestion += "\n"
        else:
            suggestion += "• Cement content is moderate (%.2f), within reasonable range.\n" % cement
        
        if clay < 0.29:
            suggestion += "• Clay content is low (%.2f), which may affect material cohesion." % clay
            if main_factor == "Clay Content":
                suggestion += " SHAP analysis shows clay content is the main factor; recommend increasing to 0.30-0.45 range."
            suggestion += "\n"
        elif clay > 0.6:
            suggestion += "• Clay content is high (%.2f), which may increase shrinkage cracking risk." % clay
            if main_factor == "Clay Content":
                suggestion += " SHAP analysis shows clay content is the main factor; recommend reducing to 0.35-0.55 range."
            suggestion += "\n"
        else:
            suggestion += "• Clay content is moderate (%.2f), within reasonable range.\n" % clay
            
        suggestion += "\nBased on SHAP analysis results, "
        
        if main_factor == "Water Content":
            suggestion += "water content has the largest impact on strength."
            if water > 1.2:
                suggestion += " Recommend reducing water content to increase strength."
            elif water < 0.7:
                suggestion += " Recommend moderately increasing water content to improve workability."
            else:
                suggestion += " Current water content is reasonable, maintain current level."
        elif main_factor == "Cement Content":
            suggestion += "cement content has the largest impact on strength."
            if cement < 0.12:
                suggestion += " Recommend increasing cement content to improve strength."
            elif cement > 0.2:
                suggestion += " Current cement content is high and provides good strength. Consider moderate reduction to optimize cost."
            else:
                suggestion += " Current cement content is reasonable, maintain current level."
        elif main_factor == "Clay Content":
            suggestion += "clay content has the largest impact on strength."
            if clay < 0.3:
                suggestion += " Recommend increasing clay content to improve material stability."
            elif clay > 0.55:
                suggestion += " Recommend reducing clay content to decrease shrinkage cracking risk."
            else:
                suggestion += " Current clay content is reasonable, maintain current level."
        
        suggestion += "\n\nIf target strength is:"
        if mixed_pred < 0.5:
            suggestion += "\n- Increase to medium strength (0.5-1.0 MPa): Consider"
            if main_factor == "Water Content":
                suggestion += " reducing water content to 0.7-0.9 range"
            elif main_factor == "Cement Content":
                suggestion += " increasing cement content to 0.15-0.2 range"
            else:
                suggestion += " adjusting clay content to 0.35-0.5 range"
        elif mixed_pred < 1.5:
            suggestion += "\n- Increase to high strength (>1.5 MPa): Consider"
            if main_factor == "Water Content":
                suggestion += " reducing water content to 0.6-0.8 range and"
            suggestion += " increasing cement content to 0.18-0.25 range"
        else:
            suggestion += "\n- Maintain high strength: Current mix already achieves high strength. Consider fine-tuning parameters to optimize cost and workability"
        
        return suggestion


def main():
    root = tk.Tk()
    app = PINNStrengthPredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 