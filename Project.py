import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from scipy import ndimage
root = tk.Tk()
root.title(" Image Processing ")
root.geometry("1450x850")
BG_MAIN       = "#070B14"   
BG_PANEL      = "#0D1526"  
BG_CARD       = "#111D35"   
ACCENT_BLUE   = "#4F8EF7"   
ACCENT_CYAN   = "#00D4FF"   
ACCENT_GREEN  = "#00E5A0"   
ACCENT_PURPLE = "#7B5EEA"  
ACCENT_PINK   = "#FF7EC7"   
ACCENT_ORANGE = "#FF8C42"   
ACCENT_YELLOW = "#FFD700"   
ACCENT_RED    = "#FF4D6A"   
TEXT_WHITE    = "#E8F0FF"   
TEXT_GRAY     = "#5A7299"   
BORDER_COLOR  = "#1E3058"   
root.configure(bg=BG_MAIN)
original_img  = None
processed_img = None
def display_image(img, label):
    if img is None:
        return
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((420, 320), Image.LANCZOS)
    img_tk  = ImageTk.PhotoImage(img_pil)
    label.config(image=img_tk)
    label.image = img_tk
def update_image(result):
    global processed_img
    processed_img = result
    display_image(result, processed_label)
def browse_image():
    global original_img, processed_img
    path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
    if path:
        original_img  = cv2.imread(path)
        processed_img = original_img.copy()
        display_image(original_img,  original_label)
        display_image(processed_img, processed_label)
def save_image():
    if processed_img is None:
        return
    path = filedialog.asksaveasfilename(defaultextension=".jpg")
    if path:
        cv2.imwrite(path, processed_img)
        messagebox.showinfo("Saved", "Image saved successfully!")
def reset_image():
    global processed_img
    if original_img is not None:
        processed_img = original_img.copy()
        display_image(processed_img, processed_label)
def addition():
    result = cv2.add(processed_img, np.full(processed_img.shape, 50, dtype=np.uint8))
    update_image(result)
def subtraction():
    result = cv2.subtract(processed_img, np.full(processed_img.shape, 50, dtype=np.uint8))
    update_image(result)
def division():
    result = np.uint8(np.clip(processed_img / 2, 0, 255))
    update_image(result)
def complement():
    result = 255 - processed_img
    update_image(result)
def change_red():
    result = processed_img.copy()
    result[:, :, 2] = cv2.add(result[:, :, 2], 80)
    update_image(result)
def swap_rg():
    result = processed_img.copy()
    result[:, :, [1, 2]] = result[:, :, [2, 1]]
    update_image(result)
def eliminate_red():
    result = processed_img.copy()
    result[:, :, 2] = 0
    update_image(result)
def histogram_stretching():
    gray      = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    min_val   = np.min(gray)
    max_val   = np.max(gray)
    stretched = ((gray - min_val) / (max_val - min_val + 1e-6)) * 255
    update_image(np.uint8(stretched))
def histogram_equalization():
    gray      = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    update_image(equalized)
def average_filter():
    result = cv2.blur(processed_img, (5, 5))
    update_image(result)
def laplacian_filter():
    gray   = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    result = cv2.Laplacian(gray, cv2.CV_64F)
    update_image(np.uint8(np.absolute(result)))
def max_filter():
    kernel = np.ones((5, 5), np.uint8)
    update_image(cv2.dilate(processed_img, kernel))
def min_filter():
    kernel = np.ones((5, 5), np.uint8)
    update_image(cv2.erode(processed_img, kernel))
def median_filter():
    update_image(cv2.medianBlur(processed_img, 5))
def mode_filter():
    if len(processed_img.shape) == 3:
        channels = []
        for c in range(3):
            ch = processed_img[:, :, c].astype(np.float64)
            mode_ch = ndimage.generic_filter(ch,lambda x: np.bincount(x.astype(np.int32)).argmax(),size=5)
            channels.append(mode_ch.astype(np.uint8))
        result = cv2.merge(channels)
    else:
        result = ndimage.generic_filter(processed_img.astype(np.float64),lambda x: np.bincount(x.astype(np.int32)).argmax(),size=5).astype(np.uint8)
    update_image(result)
def salt_pepper_noise():
    noisy       = processed_img.copy()
    row, col, _ = noisy.shape
    prob        = 0.02
    num_salt    = int(np.ceil(prob * row * col))
    coords_salt = [np.random.randint(0, i, num_salt) for i in (row, col)]
    noisy[coords_salt[0], coords_salt[1], :] = 255
    coords_pepper = [np.random.randint(0, i, num_salt) for i in (row, col)]
    noisy[coords_pepper[0], coords_pepper[1], :] = 0
    update_image(noisy)
def average_restoration():
    update_image(cv2.blur(processed_img, (5, 5)))
def median_restoration():
    update_image(cv2.medianBlur(processed_img, 5))
def outlier_method():
    img_float = processed_img.astype(np.float32)
    blurred   = cv2.blur(img_float, (5, 5))
    diff      = np.abs(img_float - blurred)
    threshold = 50
    result    = np.where(diff > threshold, blurred, img_float).astype(np.uint8)
    update_image(result)
def gaussian_noise():
    row, col, ch = processed_img.shape
    gauss = np.random.normal(0, 25, (row, col, ch))
    noisy = np.clip(processed_img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    update_image(noisy)
def image_averaging():
    acc = np.zeros_like(processed_img, dtype=np.float32)
    for ksize in [3, 5, 7, 9]:
        acc += cv2.blur(processed_img, (ksize, ksize)).astype(np.float32)
    update_image(np.uint8(acc / 4))
def gaussian_average_restoration():
    update_image(cv2.blur(processed_img, (5, 5)))
def sobel_detector():
    gray   = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    update_image(np.uint8(np.clip(cv2.magnitude(sobelx, sobely), 0, 255)))
def dilation():
    gray   = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    update_image(cv2.dilate(gray, kernel))
def erosion():
    gray   = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    update_image(cv2.erode(gray, kernel))
def opening():
    gray   = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    update_image(cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel))
def internal_boundary():
    gray   = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    update_image(gray - cv2.erode(gray, kernel))
def external_boundary():
    gray   = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    update_image(cv2.dilate(gray, kernel) - gray)
def morphological_gradient():
    gray   = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    update_image(cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel))
HOVER_MAP = {ACCENT_RED:    "#FF1A3A",ACCENT_PINK:   "#FF50A0",ACCENT_GREEN:  "#00BF86",ACCENT_CYAN:   "#00AACC",ACCENT_BLUE:   "#2B6FD4",ACCENT_PURPLE: "#5A3DC8",ACCENT_ORANGE: "#E06A20",ACCENT_YELLOW: "#CCB000",}
BG_MAP = {ACCENT_RED:    "#200010",ACCENT_PINK:   "#200018",ACCENT_GREEN:  "#003D2A",ACCENT_CYAN:   "#00142B",ACCENT_BLUE:   "#0A1F40",ACCENT_PURPLE: "#16103A",ACCENT_ORANGE: "#280F00",ACCENT_YELLOW: "#2B2200",}
BORDER_MAP = {ACCENT_RED:    "#400020",ACCENT_PINK:   "#3D0030",ACCENT_GREEN:  "#005A3C",ACCENT_CYAN:   "#003D5C",ACCENT_BLUE:   "#1A3A70",ACCENT_PURPLE: "#2D226E",ACCENT_ORANGE: "#4A2000",ACCENT_YELLOW: "#3D3000",}
def on_enter(e, btn, hover_fg, hover_bg):
    btn.config(fg=hover_fg, bg=hover_bg)
def on_leave(e, btn, normal_fg, normal_bg):
    btn.config(fg=normal_fg, bg=normal_bg)
def make_button(parent, text, command, color=ACCENT_BLUE, width=26):
    fg     = color
    bg     = BG_MAP.get(color, BG_CARD)
    hover_fg = HOVER_MAP.get(color, "#FFFFFF")
    hover_bg = BG_MAP.get(color, BG_CARD)
    btn = tk.Button(parent, text=text, command=command,bg=bg, fg=fg,font=("Consolas", 10, "bold"),width=width, relief="flat", bd=0,highlightbackground=BORDER_MAP.get(color, BORDER_COLOR),highlightthickness=1,activebackground=hover_bg,activeforeground=hover_fg,cursor="hand2", pady=6)
    btn.bind("<Enter>", lambda e, b=btn, hf=hover_fg, hb=hover_bg: on_enter(e, b, hf, hb))
    btn.bind("<Leave>", lambda e, b=btn, nf=fg, nb=bg: on_leave(e, b, nf, nb))
    return btn
title_bar = tk.Frame(root, bg=BG_PANEL, bd=0)
title_bar.pack(fill="x")
tk.Frame(title_bar, bg=BG_PANEL, height=1).pack(fill="x")  
inner_title = tk.Frame(title_bar, bg=BG_PANEL)
inner_title.pack(fill="x", padx=20, pady=10)
for dot_color in ["#FF4D6A", "#FFD700", "#00E5A0"]:
    tk.Label(inner_title, text="●", bg=BG_PANEL, fg=dot_color,font=("Consolas", 10)).pack(side="left", padx=2)
tk.Label(inner_title,text="  IMAGE PROCESSING",bg=BG_PANEL, fg=ACCENT_CYAN,font=("Consolas", 20, "bold")).pack(side="left", padx=10)
tk.Frame(title_bar, bg=BORDER_COLOR, height=1).pack(fill="x")
top_frame = tk.Frame(root, bg=BG_MAIN)
top_frame.pack(pady=10)
for txt, cmd, col in [("  Browse Image", browse_image, ACCENT_GREEN),("  Reset",        reset_image,  ACCENT_YELLOW),("  Save Image",   save_image,   ACCENT_CYAN),]:
    make_button(top_frame, txt, cmd, color=col, width=18).pack(side="left", padx=6)
img_frame = tk.Frame(root, bg=BG_MAIN)
img_frame.pack(pady=6)
def make_img_card(parent, label_text, accent):
    card = tk.Frame(parent, bg=BG_CARD, bd=0,
                    highlightbackground=BORDER_COLOR, highlightthickness=1)
    card.grid(row=0, column=parent.grid_size()[0], padx=18)
    tk.Label(card, text=label_text, bg=BG_CARD, fg=accent,
             font=("Consolas", 10, "bold")).pack(pady=(8, 2))
    img_lbl = tk.Label(card, bg="#050810", width=420, height=320)
    img_lbl.pack(padx=6, pady=(0, 8))
    return img_lbl
original_label  = make_img_card(img_frame, "  ◈  ORIGINAL  ",  ACCENT_CYAN)
processed_label = make_img_card(img_frame, "  ◈  PROCESSED  ", ACCENT_PURPLE)
style = ttk.Style()
style.theme_use("clam")
style.configure("TNotebook",background=BG_MAIN,borderwidth=0,tabmargins=[0, 0, 0, 0])
style.configure("TNotebook.Tab",background=BG_PANEL,foreground=TEXT_GRAY,font=("Consolas", 10, "bold"),padding=[14, 7],borderwidth=0)
style.map("TNotebook.Tab",background=[("selected", BG_CARD)],foreground=[("selected", ACCENT_CYAN)])
notebook = ttk.Notebook(root, style="TNotebook")
notebook.pack(fill="both", expand=True, padx=14, pady=(4, 10))
def create_tab(name):
    frame = tk.Frame(notebook, bg=BG_PANEL)
    notebook.add(frame, text=f"  {name}  ")
    inner = tk.Frame(frame, bg=BG_PANEL)
    inner.pack(expand=True)
    return inner
point_tab     = create_tab("Point Ops")
color_tab     = create_tab("Color Ops")
hist_tab      = create_tab("Histogram")
linear_tab    = create_tab("Linear Filters")
nonlinear_tab = create_tab("Non-Linear Filters")
restore_tab   = create_tab("Restoration")
edge_tab      = create_tab("Edge Detection")
morph_tab     = create_tab("Morphology")
boundary_tab  = create_tab("Boundary")
def add_btn(tab, text, cmd, color=ACCENT_BLUE):
    make_button(tab, text, cmd, color=color).pack(pady=4, padx=20)
add_btn(point_tab, "  Addition",    addition,    ACCENT_RED)
add_btn(point_tab, "  Subtraction", subtraction, ACCENT_RED)
add_btn(point_tab, "  Division",    division,    ACCENT_RED)
add_btn(point_tab, "  Complement",  complement,  ACCENT_RED)
add_btn(color_tab, "  Change Red",    change_red,    ACCENT_PINK)
add_btn(color_tab, "  Swap R → G",    swap_rg,       ACCENT_PINK)
add_btn(color_tab, "  Eliminate Red", eliminate_red, ACCENT_PINK)
add_btn(hist_tab, "  Histogram Stretching",   histogram_stretching,   ACCENT_CYAN)
add_btn(hist_tab, "  Histogram Equalization", histogram_equalization, ACCENT_CYAN)
add_btn(linear_tab, "  Average Filter",   average_filter,   ACCENT_PURPLE)
add_btn(linear_tab, "  Laplacian Filter", laplacian_filter, ACCENT_PURPLE)
add_btn(nonlinear_tab, "  Maximum Filter", max_filter,    ACCENT_ORANGE)
add_btn(nonlinear_tab, "  Minimum Filter", min_filter,    ACCENT_ORANGE)
add_btn(nonlinear_tab, "  Median Filter",  median_filter, ACCENT_ORANGE)
add_btn(nonlinear_tab, "  Mode Filter",    mode_filter,   ACCENT_ORANGE)
add_btn(restore_tab, "  Salt & Pepper Noise",         salt_pepper_noise,            ACCENT_YELLOW)
add_btn(restore_tab, "  Average Restoration (S&P)",   average_restoration,          ACCENT_YELLOW)
add_btn(restore_tab, "  Median Restoration (S&P)",    median_restoration,           ACCENT_YELLOW)
add_btn(restore_tab, "  Outlier Method (S&P)",        outlier_method,               ACCENT_YELLOW)
add_btn(restore_tab, "  Gaussian Noise",              gaussian_noise,               ACCENT_YELLOW)
add_btn(restore_tab, "  Image Averaging (Gaussian)",  image_averaging,              ACCENT_YELLOW)
add_btn(restore_tab, "  Average Filter (Gaussian)",   gaussian_average_restoration, ACCENT_YELLOW)
add_btn(edge_tab, "  Sobel Detector", sobel_detector, ACCENT_GREEN)
add_btn(morph_tab, "  Dilation", dilation, ACCENT_GREEN)
add_btn(morph_tab, "  Erosion",  erosion,  ACCENT_GREEN)
add_btn(morph_tab, "  Opening",  opening,  ACCENT_GREEN)
add_btn(boundary_tab, "  Internal Boundary",      internal_boundary,      ACCENT_BLUE)
add_btn(boundary_tab, "  External Boundary",      external_boundary,      ACCENT_BLUE)
add_btn(boundary_tab, "  Morphological Gradient", morphological_gradient, ACCENT_BLUE)
root.mainloop()
