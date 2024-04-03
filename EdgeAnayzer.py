import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from circle_fit import taubinSVD
import os


def prepare_data(path):
    x = []
    y = []
    with open(path, 'r') as file:
        for line in file:
            # replace "," with "."
            line_dec = line.replace(",", ".")
            # Split the line by tabs
            items = line_dec.strip().split('\t')
            x.append(float(items[0]))
            y.append(float(items[1]))
    x_raw = np.asarray(x)
    y_raw = np.asarray(y)
    y_smooth = savgol_filter(y_raw, 3, 2)
    return x_raw, y_smooth

def select_path(button):
    global file_list
    # Clear Treeviw and Plot
    file_tree.delete(*file_tree.get_children())
    axs1.clear()

    file_list = []
    # Check if folder or File was selected and create file list
    if button == "folder":
        folder_path = filedialog.askdirectory()
        if folder_path:
            files = os.listdir(folder_path)
            for i in range(len(files)):
                file_list.append(os.path.join(folder_path, files[i]))
    elif button == "file":
        file_path = filedialog.askopenfilename()
        if file_path:
            file_list.append(file_path)

    for file in file_list:
        # print file list in treeview
        file_tree.insert('', 'end', values=(os.path.basename(file), 'k.A.'))
        x_raw, y_raw = prepare_data(file)
        # plot raw profiles
        axs1.set_xlabel('x [mm]')
        axs1.set_ylabel('y [mm]')
        axs1.set_title('raw profile')
        axs1.plot(x_raw, y_raw)
        canvas1.draw()
        axs2.clear()
        axs2.set_xlabel('x [mm]')
        axs2.set_ylabel('y [mm]')
        axs2.set_title('cleaned edge')
        canvas2.draw()
    # update Info
    info = os.path.dirname(file_list[0])
    info_label.configure(text="Selected folder: " + str(info))

def clear_selection():
    global file_list
    file_list = None
    del file_list
    file_tree.delete(*file_tree.get_children())
    axs1.clear()
    axs2.clear()
    axs1.set_xlabel('x [mm]')
    axs1.set_ylabel('y [mm]')
    axs1.set_title('raw profile')
    axs2.set_xlabel('x [mm]')
    axs2.set_ylabel('y [mm]')
    axs2.set_title('cleaned edge')
    canvas1.draw()
    canvas2.draw()
    info_label.configure(text="Selection cleared! Select file or folder to start... ")

def fit_calculation(export):
    try:
        file_list
        # Clear Treeview and Plot
        file_tree.delete(*file_tree.get_children())
        cut_value = float(entry_cut_value.get())               # Get the integer value from the Entry field
        no_flank = cut_value - float(entry_linear_area.get())  # get linear model of profile flanks
        # initialize result list
        result_file = []
        result_radius = []
        for file in file_list:
            axs1.clear() # see every plot seperatly
            axs2.clear()
            x_raw, y_raw = prepare_data(file)
            # determin transition points
            x_min_limit, x_max_limit, x_shift, y_shift, x_tip, y_tip, x_lin_left, y_lin_left, x_lin_right, y_lin_right, \
                x_relief_left, y_relief_left, x_relief_right, y_relief_right, x_left, y_left, x_right, y_right, \
                    tp_error = edge_detection(cut_value, no_flank, x_raw, y_raw)
            # fit calculaiton
            x_edge = x_shift[(x_shift > x_relief_left) & (x_shift < x_relief_right)]
            y_edge = y_shift[(x_shift > x_relief_left) & (x_shift < x_relief_right)]

            # Error catching
            # check if transition points exist
            if tp_error == 1:
                err_msg = 'Transition points not found'
                print(err_msg)
                radius = np.nan
                center = np.nan
            else:
                # transition points must be on the correct side of the tip point
                if x_relief_left < x_tip and x_relief_right > x_tip:
                    # ratio of distances between transition and tip point less than factor X --> force symmetry
                    if max((x_tip - x_relief_left), (x_relief_right - x_tip)) / min((x_tip - x_relief_left),(x_relief_right - x_tip)) < 2:
                        print('Transition points good')
                        radius, center = circle_fit(x_edge, y_edge)
                    else:
                        # distances too different
                        err_msg = 'Transition points uneven'
                        print(err_msg)
                        radius = np.nan
                        center = np.nan
                else:
                    # transition point on wrong side
                    err_msg = 'Transition points out of boundary'
                    print(err_msg)
                    radius = np.nan
                    center = np.nan

                # center of circle must be under the tip
                if np.isnan(radius):
                    pass
                else:
                    if y_tip > center[1]:
                        if x_relief_left < center[0] and x_relief_right > center[0]:
                            pass
                        else:
                            err_msg = 'Center point out of boundary'
                            print(err_msg)
                            radius = np.nan
                            center = np.nan
                    else:
                        err_msg = 'Center point out of boundary'
                        print(err_msg)
                        radius = np.nan
                        center = np.nan
            # check if radius makes sense regarding the distance of transition points
            # large radii occur is edge is very flat
            if radius / (x_relief_right - x_relief_left) > 10:
                err_msg = 'Radius too large'
                print(err_msg)
                radius = np.nan
                center = np.nan

            # print file list in treeview and append results
            if np.isnan(radius):
                file_tree.insert('', 'end', values=(os.path.basename(file),'nan'))
                result_file.append(os.path.basename(file))
                result_radius.append('nan')
            else:
                file_tree.insert('', 'end', values=(os.path.basename(file), round(radius*1000)))
                result_file.append(os.path.basename(file))
                result_radius.append(round(radius*1000))

            # Plotting
            # Plot scaled data on the first plot window
            axs1.set_xlabel('x [mm]')
            axs1.set_ylabel('y [mm]')
            axs1.set_title('raw profile')
            axs1.plot(x_raw, y_raw)
            axs1.vlines(x_min_limit, ymax=max(y_raw), ymin=min(y_raw), linestyles='dashed')
            axs1.vlines(x_max_limit, ymax=max(y_raw), ymin=min(y_raw), linestyles='dashed')
            canvas1.draw()  # Redraw canvas with new plot

            # Plot scaled data on the second plot window
            axs2.set_xlabel('x [mm]')
            axs2.set_ylabel('y [mm]')
            axs2.set_title('cleaned edge')
            axs2.plot(x_shift, y_shift)
            axs2.plot(x_lin_left, y_lin_left, 'k--', x_lin_right, y_lin_right, 'k--')
            axs2.plot(x_edge, y_edge, 'kx')
            axs2.plot(x_left, y_left, 'ko')
            axs2.plot(x_right, y_right, 'ko')
            axs2.plot(x_tip, y_tip, 'rx', markersize=10)
            axs2.plot(x_relief_left, y_relief_left, 'bx', markersize=10)
            axs2.plot(x_relief_right, y_relief_right, 'bx', markersize=10)
            if np.isnan(radius):
                axs2.text(0.5, 0, f'no calculation: {err_msg} ', ha='left', va='bottom', color='red')
            else:
                circle_finale = plt.Circle((center[0], center[1]), radius, color='b', fill=False)
                axs2.plot(center[0], center[1], 'k+')
                axs2.add_patch(circle_finale)
                axs2.text(center[0], center[1], f'radius: {round(radius, 2)}', ha='right', va='bottom', color='red')
            canvas2.draw()  # Redraw canvas with new plot
            root.update()

        # file export ?
        if export == 1:
            not_nan_indexes = [index for index, element in enumerate(result_radius) if element != 'nan']
            filtered_file_list = [file_list[i] for i in not_nan_indexes]
            filtered_result_file = [result_file[i] for i in not_nan_indexes]
            filtered_result_radius = [result_radius[i] for i in not_nan_indexes]
            save_path = result_exporter(filtered_file_list, filtered_result_file, filtered_result_radius)
            # update Info
            info_label.configure(text="Results are exported to" + str(save_path))
        else:
            pass
    except NameError:
        info_label.configure(text = "No file or folder selected")

def circle_fit(x_edge, y_edge):
    xy_edge = np.stack((x_edge, y_edge), axis=1)
    list_of_pairs = [tuple(row) for row in xy_edge]
    xc, yc, r, sigma = taubinSVD(list_of_pairs)
    center = (xc, yc)
    return r, center

def three_point_calculation(export):
    try:
        file_list
        # Clear Treeviw and Plot
        file_tree.delete(*file_tree.get_children())
        cut_value = float(entry_cut_value.get())               # Get the integer value from the Entry field
        no_flank = cut_value - float(entry_linear_area.get())  # get linear model of profile flanks
        # initialize result list
        result_file = []
        result_radius = []
        for file in file_list:
            axs1.clear() # see every plot seperatly
            axs2.clear()
            x_raw, y_raw = prepare_data(file)
            # determin transition points
            x_min_limit, x_max_limit, x_shift, y_shift, x_tip, y_tip, x_lin_left, y_lin_left, x_lin_right, y_lin_right, \
                x_relief_left, y_relief_left, x_relief_right, y_relief_right, x_left, y_left, x_right, \
                    y_right = edge_detection(cut_value, no_flank, x_raw, y_raw)
            # 3-Point calculaiton
            center, radius = define_circle((x_relief_left, y_relief_left), (x_relief_right, y_relief_right),(x_tip, y_tip))
            # print file list in treeview and append results
            file_tree.insert('', 'end', values=(os.path.basename(file), round(radius*1000)))
            result_file.append(os.path.basename(file))
            result_radius.append(round(radius*1000))
            # Plotting
            # Plot scaled data on the first plot window
            axs1.set_xlabel('x [mm]')
            axs1.set_ylabel('y [mm]')
            axs1.set_title('raw profile')
            axs1.plot(x_raw, y_raw)
            axs1.vlines(x_min_limit, ymax=max(y_raw), ymin=min(y_raw), linestyles='dashed')
            axs1.vlines(x_max_limit, ymax=max(y_raw), ymin=min(y_raw), linestyles='dashed')
            canvas1.draw()  # Redraw canvas with new plot

            # Plot scaled data on the second plot window
            axs2.set_xlabel('x [mm]')
            axs2.set_ylabel('y [mm]')
            axs2.set_title('cleaned edge')
            axs2.plot(x_shift, y_shift, x_tip, y_tip, 'kx')
            axs2.plot(x_lin_left, y_lin_left, 'k--', x_lin_right, y_lin_right, 'k--')
            axs2.plot(x_relief_left, y_relief_left, 'kx', x_relief_right, y_relief_right, 'kx')
            axs2.plot(x_left, y_left, 'ko')
            axs2.plot(x_right, y_right, 'ko')
            circle_finale = plt.Circle((center[0], center[1]), radius, color='b', fill=False)
            axs2.plot(center[0], center[1], 'k+')
            axs2.add_patch(circle_finale)
            axs2.text(center[0], center[1], f'radius: {round(radius, 2)}', ha='right', va='bottom', color='red')
            canvas2.draw()  # Redraw canvas with new plot
            root.update()

            # file export ?
            if export == 1:
                save_path = result_exporter(file_list, result_file, result_radius)
                # update Info
                info_label.configure(text="Results are exported to" + str(save_path))
            else:
                pass
    except NameError:
        info_label.configure(text = "No file or folder selected")

def result_exporter(file_list, result_file, result_radius):
    save_path = os.path.dirname(file_list[0])
    upper_folder = os.path.dirname(save_path)
    with open(os.path.join(upper_folder, 'Kantenradien.txt'),'w') as file:
        file.write("File\tRadius [µm]\n")  # Writing the header

        # Writing data from both lists into the file
        for file_name, result in zip(result_file, result_radius):
            file.write(f"{file_name}\t{result}\n")
    return upper_folder

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def edge_detection(cut_value,no_flank,x_raw,y_raw):
    max_value_y = max(y_raw)
    middle_value_x = x_raw[(y_raw == max_value_y)]
    x_min_limit = min(middle_value_x) - cut_value
    x_max_limit = max(middle_value_x) + cut_value
    x_cut = x_raw[(x_raw > x_min_limit) & (x_raw < x_max_limit)]
    y_cut = y_raw[(x_raw > x_min_limit) & (x_raw < x_max_limit)]
    x_shift = x_cut - min(x_cut)
    y_shift = y_cut - min(y_cut)

    # get peak of profile
    max10_ind = np.argpartition(y_shift, -10)[-10:]  # average over 10 greatest values
    y_tip = np.mean(y_shift[max10_ind])
    x_tip = np.mean(x_shift[max10_ind])

    # define sections of flanks
    x_left = x_shift[np.where(x_shift < (x_tip-no_flank))]
    y_left = y_shift[np.where(x_shift < (x_tip-no_flank))]
    x_right = x_shift[np.where(x_shift > (x_tip+no_flank))]
    y_right = y_shift[np.where(x_shift > (x_tip+no_flank))]

    # linear modell of flanks
    m_left, b_left = np.polyfit(x_left, y_left, 1)
    m_right, b_right = np.polyfit(x_right, y_right, 1)
    x_intersept = (b_right - b_left)/(m_left - m_right)
    x_lin_left = x_shift[np.where(x_shift < x_intersept)]
    x_lin_right = x_shift[np.where(x_shift > x_intersept)]
    y_lin_left = m_left * x_lin_left + b_left
    y_lin_right = m_right * x_lin_right + b_right

    # find transition points
    # get mean error of flank approximation
    y_cut_left = y_shift[0:len(y_lin_left)]
    x_cut_left = x_shift[0:len(y_lin_left)]
    y_cut_right = y_shift[len(y_shift) - len(y_lin_right):len(y_shift)]
    x_cut_right = x_shift[len(x_shift) - len(y_lin_right):len(x_shift)]
    err_left = mean_absolute_error(y_cut_left, y_lin_left)
    err_right = mean_absolute_error(y_cut_right, y_lin_right)

    # run up left flank and check if real flank reliefs from flank approximation more than mean error
    relief_left = len(y_cut_left) - 1  # initialize with highest possible value --> will cause error catch if no transition point is found
    for i in range(len(y_lin_left)):
        if y_cut_left[i] < (y_lin_left[i] - err_left):
            y_cut_temp = y_cut_left[i:len(y_cut_left)]
            y_flank_temp = y_lin_left[i:len(y_lin_left)]
            check_rest = y_cut_temp > (y_flank_temp - err_left)
            if True in check_rest:
                pass
            else:
                relief_left = i
                break
    # if no left transition point is found, skip search of right transition point and set it so zero
    # --> will cause error catch if no transition point is found
    transition_point_error = 0
    if relief_left == len(y_cut_left) - 1:
        relief_right = 0
        transition_point_error = 1
    else:
        # run down right flank and check if real flank reliefs from flank approximation more than mean error
        for i in range(len(y_lin_right)):
            if y_cut_right[i] > y_lin_right[i] - (err_right):
                relief_right = i - 1
                break
    x_relief_left = x_cut_left[relief_left]
    y_relief_left = y_cut_left[relief_left]
    x_relief_right = x_cut_right[relief_right]
    y_relief_right = y_cut_right[relief_right]

    return x_min_limit, x_max_limit, x_shift, y_shift, x_tip, y_tip, x_lin_left, y_lin_left, x_lin_right, y_lin_right, \
        x_relief_left, y_relief_left, x_relief_right, y_relief_right, x_left, y_left, x_right, y_right, transition_point_error


# GUI Setup
root = tk.Tk()
root.geometry("1410x500")
color = "snow3"
root.configure(bg=color)

### Control Panel ###
frame_control = tk.Frame(root, bg = color)
frame_control.pack(side='left', fill='y')
## Selection Frame
frame_selection = tk.Frame(frame_control, bg = color)
frame_selection.pack(side='top',fill='x')
## File Shower Frame
frame_file_shower = tk.Frame(frame_control, bg=color)
frame_file_shower.pack(side='top', fill='x')
## File Handle Frame
frame_file_handle = tk.Frame(frame_control, bg=color)
frame_file_handle.pack(side='top', fill='x')
## Parameter Frame
frame_parameter_cut = tk.Frame(frame_control, bg=color)
frame_parameter_cut.pack(side='top', fill='x')
frame_parameter_linear = tk.Frame(frame_control, bg=color)
frame_parameter_linear.pack(side='top', fill='x')
frame_parameter_img = tk.Frame(frame_control, bg=color)
frame_parameter_img.pack(side='top', fill='x')
### Elements of selection Frame
select_file_button = tk.Button(frame_selection, text="Select File", command=lambda: select_path("file"))
select_file_button.pack(side='left', padx=5, pady=5)
select_file_label = tk.Label(frame_selection, text="or", bg=color)
select_file_label.pack(side='left', padx=5, pady=5)
select_folder_button = tk.Button(frame_selection, text="Select Folder", command=lambda: select_path("folder"))
select_folder_button.pack(side='left', padx=5, pady=5)
### Elements of File Shower Frame
file_tree = ttk.Treeview(frame_file_shower, columns=('Name', 'Radius [µm]'), show='headings')
file_tree.heading('Name', text='Name')
file_tree.heading('Radius [µm]', text='Radius [µm]')
file_tree.column('Name', width=88)
file_tree.column('Radius [µm]', width=88)
file_tree['height'] = 9
file_tree.pack(side='left', padx=5, pady=5)  # Adjust dimensions here
### Elements of File Handle Frame
clear_button = tk.Button(frame_file_handle, text="Clear Selection", command=clear_selection)
clear_button.pack(side='left', padx=5, pady=5)
export_checker = tk.IntVar()
checkbox = tk.Checkbutton(frame_file_handle, text="Export", variable=export_checker, bg=color)
checkbox.pack(side='left', padx=0, pady=0)
### Elements of Parameter Frame
# Entry cut value
label_cut_value = tk.Label(frame_parameter_cut, text="Cut value [mm]:  ", bg='snow4')
label_cut_value.pack(side='left', padx=5, pady=5)
entry_cut_value = tk.Entry(frame_parameter_cut, width=11)
entry_cut_value.insert(0, "2.0")  # Insert "2" at index 0 initally
entry_cut_value.pack(side='left', padx=5, pady=5)
# Entry linear area
label_linear_area = tk.Label(frame_parameter_linear, text="Linear area [mm]:", bg='snow4')
label_linear_area.pack(side='left', padx=5, pady=5)
entry_linear_area = tk.Entry(frame_parameter_linear, width=11)
entry_linear_area.insert(0, "1.2")  # Insert "2" at index 0 initally
entry_linear_area.pack(side='left', padx=5, pady=5)
# Image
small_image = Image.open("cutvalue_lineararea_graphic.PNG")
small_image = small_image.resize((172, 95))
small_image = ImageTk.PhotoImage(small_image)
small_image_label = tk.Label(frame_parameter_img, image=small_image)
small_image_label.pack(side='left', padx=5, pady=5)
# Calculate
# calc_button_single = tk.Button(frame_control, text="Calculate (3P)", command=lambda: three_point_calculation(export_checker.get()))
# calc_button_single.pack(side = 'left', pady=5)
calc_button_single = tk.Button(frame_control, text="Calculate (fit)", command=lambda: fit_calculation(export_checker.get()))
calc_button_single.pack(side='left', pady=5)


### Plot Panel ###
frame_plot_all = tk.Frame(root, bg=color)
frame_plot_all.pack(side='top', fill='both')
# Figure 1
fig1, axs1 = plt.subplots(figsize=(6, 4.5))
axs1.set_xlabel('x [mm]')
axs1.set_ylabel('y [mm]')
axs1.set_title('raw profile')
canvas1 = FigureCanvasTkAgg(fig1, master=frame_plot_all)
canvas1.get_tk_widget().pack(side='left', padx=5, pady=5)
axs1.axis('equal')
# Figure 2
fig2, axs2 = plt.subplots(figsize=(6, 4.5))
axs2.set_xlabel('x [mm]')
axs2.set_ylabel('y [mm]')
axs2.set_title('cleaned profile')
canvas2 = FigureCanvasTkAgg(fig2, master=frame_plot_all)
canvas2.get_tk_widget().pack(side='left', padx=5, pady=5)
axs2.axis('equal')

### Print Panel ###
frame_print = tk.Frame(root, bg=color)
frame_print.pack(side='top', fill='x')
print_label = tk.Label(frame_print, text='Info:', bg=color)
print_label.pack(side='left', padx=5, pady=0)

frame_info = tk.Frame(root, bg='snow3')
frame_info.pack(side='top', fill='x')
info_label = ttk.Label(frame_info, text='Select file or folder to start...')
info_label.pack(side='left', padx=5, pady=0)

root.mainloop()