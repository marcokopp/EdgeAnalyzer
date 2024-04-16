import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from circle_fit import taubinSVD
import os
import datetime
from sys import exit
from skimage.measure import EllipseModel  # pip install scikit-image
from matplotlib.patches import Ellipse


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
        file_tree.insert('', 'end', values=(os.path.basename(file), 'k.A.', 'k.A.'))
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
        axs2.set_title('clean edge')
        canvas2.draw()
    # update Info
    info = os.path.dirname(file_list[0])
    info_label.configure(text="Selected folder: " + str(info))

def clear_selection():
    global file_list; file_list = None; del file_list
    global result_radius; result_radius = None; del result_radius
    global result_kappa; result_kappa = None; del result_kappa
    global center_list; center_list = None; del center_list
    global center_ell_list; center_ell_list = None; del center_ell_list
    global radii_ell_list; radii_ell_list = None; del radii_ell_list
    global theta_ell_list; theta_ell_list = None; del theta_ell_list
    global x_raw_list; x_raw_list = None; del x_raw_list
    global y_raw_list; y_raw_list = None; del y_raw_list
    global x_min_limit_list; x_min_limit_list = None; del x_min_limit_list
    global x_max_limit_list; x_max_limit_list = None; del x_max_limit_list
    global x_shift_list; x_shift_list = None; del x_shift_list
    global y_shift_list; y_shift_list = None; del y_shift_list
    global x_lin_left_list; x_lin_left_list = None; del x_lin_left_list
    global y_lin_left_list; y_lin_left_list = None; del y_lin_left_list
    global x_lin_right_list; x_lin_right_list = None; del x_lin_right_list
    global y_lin_right_list; y_lin_right_list = None; del y_lin_right_list
    global x_edge_list; x_edge_list = None; del x_edge_list
    global y_edge_list; y_edge_list = None; del y_edge_list
    global x_left_list; x_left_list = None; del x_left_list
    global y_left_list; y_left_list = None; del y_left_list
    global x_right_list; x_right_list = None; del x_right_list
    global y_right_list; y_right_list = None; del y_right_list
    global x_tip_list; x_tip_list = None; del x_tip_list
    global y_tip_list; y_tip_list = None; del y_tip_list
    global x_relief_left_list; x_relief_left_list = None; del x_relief_left_list
    global y_relief_left_list; y_relief_left_list = None; del y_relief_left_list
    global x_relief_right_list; x_relief_right_list = None; del x_relief_right_list
    global y_relief_right_list; y_relief_right_list = None; del y_relief_right_list
    global err_msg_list; err_msg_list = None; del err_msg_list

    file_tree.delete(*file_tree.get_children())
    axs1.clear()
    axs2.clear()
    axs1.set_xlabel('x [mm]')
    axs1.set_ylabel('y [mm]')
    axs1.set_title('raw profile')
    axs2.set_xlabel('x [mm]')
    axs2.set_ylabel('y [mm]')
    axs2.set_title('clean edge')
    canvas1.draw()
    canvas2.draw()
    info_label.configure(text="Selection cleared! Select file or folder to start... ")
    browse_label.configure(text=default_text)

# Calculate shape factor ("K" or "Kappa") to quantify edge asymmetry
def kappa_factor(x_relief_left, y_relief_left, x_lin_right, y_lin_right, x_relief_right, y_relief_right):    
    S_left = np.sqrt(pow(x_relief_left-x_lin_right[0], 2) + pow(y_relief_left-y_lin_right[0], 2))
    S_right = np.sqrt(pow(x_relief_right-x_lin_right[0], 2) + pow(y_relief_right-y_lin_right[0], 2))
    return S_left / S_right

def fit_calculation(export):
    try:
        # Clear Treeview and Plot
        file_tree.delete(*file_tree.get_children())
        cut_value = float(entry_cut_value.get())               # Get the integer value from the Entry field
        no_flank = cut_value - float(entry_linear_area.get())  # get linear model of profile flanks
        # initialize result list
        result_file = []
        global result_radius; result_radius = []
        result_radii_ell = []
        global result_kappa; result_kappa = []
        # initialize result list for plotting (all must be gobal for replotting when broswing through subplots)
        global center_list; center_list = []
        global center_ell_list; center_ell_list = []
        global radii_ell_list; radii_ell_list = []
        global theta_ell_list; theta_ell_list = []
        global x_raw_list; x_raw_list = []
        global y_raw_list; y_raw_list = []
        global x_min_limit_list; x_min_limit_list = []
        global x_max_limit_list; x_max_limit_list = []
        global x_shift_list; x_shift_list = []
        global y_shift_list; y_shift_list = []
        global x_lin_left_list; x_lin_left_list = []
        global y_lin_left_list; y_lin_left_list = []
        global x_lin_right_list; x_lin_right_list = []
        global y_lin_right_list; y_lin_right_list = []
        global x_edge_list; x_edge_list = []
        global y_edge_list; y_edge_list = []
        global x_left_list; x_left_list = []
        global y_left_list; y_left_list = []
        global x_right_list; x_right_list = []
        global y_right_list; y_right_list = []
        global x_tip_list; x_tip_list = []
        global y_tip_list; y_tip_list = []
        global x_relief_left_list; x_relief_left_list = []
        global y_relief_left_list; y_relief_left_list = []
        global x_relief_right_list; x_relief_right_list = []
        global y_relief_right_list; y_relief_right_list = []
        global err_msg_list; err_msg_list = []

        for file in file_list:
            # update Browse-lable
            browse_label.configure(text=os.path.basename(file))

            axs1.clear() # see every plot seperatly
            axs2.clear()
            x_raw, y_raw = prepare_data(file)
            # determin transition points
            x_min_limit, x_max_limit, x_shift, y_shift, x_tip, y_tip, x_lin_left, y_lin_left, x_lin_right, y_lin_right, \
                x_relief_left, y_relief_left, x_relief_right, y_relief_right, x_left, y_left, x_right, y_right, \
                    tp_error = edge_detection(cut_value, no_flank, x_raw, y_raw)
            # fit calculation
            x_edge = x_shift[(x_shift > x_relief_left) & (x_shift < x_relief_right)]
            y_edge = y_shift[(x_shift > x_relief_left) & (x_shift < x_relief_right)]

            # Error catching
            # check if transition points exist
            err_msg = 'no error'

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
                        center_ell, radii_ell, theta_ell = ellipsis_fit(x_edge, y_edge)
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

            Kappa = kappa_factor(x_relief_left, y_relief_left, x_lin_right, y_lin_right, x_relief_right, y_relief_right)
            result_kappa.append(Kappa)

            # print file list in treeview and append results
            if np.isnan(radius):
                file_tree.insert('', 'end', values=(os.path.basename(file),'nan','nan'))
                result_file.append(os.path.basename(file))
                result_radius.append('nan')
                result_radii_ell.append(('nan','nan'))
            else:
                file_tree.insert('', 'end', values=(os.path.basename(file), round(radius*1000), round(Kappa,2)))
                result_file.append(os.path.basename(file))
                result_radius.append(radius)
                result_radii_ell.append(radii_ell)

            # fill all lists for replotting
            center_list.append(center)
            center_ell_list.append(center_ell)
            radii_ell_list.append(radii_ell)
            theta_ell_list.append(theta_ell)
            x_raw_list.append(x_raw)
            y_raw_list.append(y_raw)
            x_min_limit_list.append(x_min_limit)
            x_max_limit_list.append(x_max_limit)
            x_shift_list.append(x_shift)
            y_shift_list.append(y_shift)
            x_lin_left_list.append(x_lin_left)
            y_lin_left_list.append(y_lin_left)
            x_lin_right_list.append(x_lin_right)
            y_lin_right_list.append(y_lin_right)
            x_edge_list.append(x_edge)
            y_edge_list.append(y_edge)
            x_left_list.append(x_left)
            y_left_list.append(y_left)
            x_right_list.append(x_right)
            y_right_list.append(y_right)
            x_tip_list.append(x_tip)
            y_tip_list.append(y_tip)
            x_relief_left_list.append(x_relief_left)
            y_relief_left_list.append(y_relief_left)
            x_relief_right_list.append(x_relief_right)
            y_relief_right_list.append(y_relief_right)
            err_msg_list.append(err_msg)

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
            axs2.set_title('clean edge')
            axs2.plot(x_shift, y_shift)
            axs2.plot(x_lin_left, y_lin_left, 'k--', x_lin_right, y_lin_right, 'k--')
            axs2.plot(x_edge, y_edge, 'r')
            axs2.plot(x_left, y_left, 'c')
            axs2.plot(x_right, y_right, 'c')
            axs2.plot(x_tip, y_tip, 'rx', markersize=10)
            axs2.plot(x_relief_left, y_relief_left, 'bx', markersize=10)
            axs2.plot(x_relief_right, y_relief_right, 'bx', markersize=10)
            if np.isnan(radius):
                axs2.text(0.5, 0, f'no calculation: {err_msg} ', ha='left', va='bottom', color='red')
            else:
                circle_finale = plt.Circle((center[0], center[1]), radius, color='b', fill=False)
                ellipsis_finale = PlotEllipsis(center_ell, radii_ell, theta_ell)
                axs2.plot(center[0], center[1], 'k+')
                axs2.add_patch(circle_finale)
                axs2.add_patch(ellipsis_finale)
                # axs2.text(center[0], center[1], f'radius: {round(radius, 2)}', ha='right', va='top', color='red')
                axs2.text(0, y_tip, 'r\u03b2 = {:.0f} \u03bcm\nK = {:.3f}'.format(radius * 1000, Kappa), ha='left', va='bottom', color='red')
            canvas2.draw()  # Redraw canvas with new plot
            root.update()

        if export == 1:
            # not_nan_indexes = [index for index, element in enumerate(result_radius) if element != 'nan']
            # filtered_file_list = [file_list[i] for i in not_nan_indexes]
            # filtered_result_file = [result_file[i] for i in not_nan_indexes]
            # filtered_result_radius = [result_radius[i] for i in not_nan_indexes]
            # save_path = result_exporter(filtered_file_list, filtered_result_file, filtered_result_radius)
            save_path = result_exporter(file_list, result_file, result_radius, result_radii_ell, result_kappa)
            # update Info
            info_label.configure(text="Results are exported to " + str(save_path))
        else:
            pass
    except NameError:
        info_label.configure(text="No file or folder selected")

def PlotEllipsis(center, radii, theta):
    return Ellipse(center, radii[0]*2, radii[1]*2, angle=theta, alpha=0.5)

def circle_fit(x_edge, y_edge):
    xy_edge = np.stack((x_edge, y_edge), axis=1)
    list_of_pairs = [tuple(row) for row in xy_edge]
    xc, yc, r, sigma = taubinSVD(list_of_pairs)
    center = (xc, yc)
    return r, center

def ellipsis_fit(x_edge, y_edge):
    xy_edge = np.stack((x_edge, y_edge), axis=1)
    # list_of_pairs = [tuple(row) for row in xy_edge]
    # xc, yc, r, sigma = taubinSVD(list_of_pairs)

    ell = EllipseModel()
    ell.estimate(xy_edge)
    xc, yc, a, b, theta = ell.params
    center = (xc, yc)
    radii = (a,b)
    return center, radii, theta

# def three_point_calculation(export):
#     try:
#         file_list
#         # Clear Treeviw and Plot
#         file_tree.delete(*file_tree.get_children())
#         cut_value = float(entry_cut_value.get())               # Get the integer value from the Entry field
#         no_flank = cut_value - float(entry_linear_area.get())  # get linear model of profile flanks
#         # initialize result list
#         result_file = []
#         result_radius = []
#         result_kappa = []
#         for file in file_list:
#             axs1.clear() # see every plot seperatly
#             axs2.clear()
#             x_raw, y_raw = prepare_data(file)
#             # determin transition points
#             x_min_limit, x_max_limit, x_shift, y_shift, x_tip, y_tip, x_lin_left, y_lin_left, x_lin_right, y_lin_right, \
#                 x_relief_left, y_relief_left, x_relief_right, y_relief_right, x_left, y_left, x_right, \
#                     y_right = edge_detection(cut_value, no_flank, x_raw, y_raw)
#             # 3-Point calculaiton
#             center, radius = define_circle((x_relief_left, y_relief_left), (x_relief_right, y_relief_right),(x_tip, y_tip))
#             # print file list in treeview and append results
#             file_tree.insert('', 'end', values=(os.path.basename(file), round(radius*1000)))
#             result_file.append(os.path.basename(file))
#             result_radius.append(radius)

#             K = kappa_factor(x_relief_left, y_relief_left, x_tip, y_tip, x_relief_right, y_relief_right)
#             result_kappa.append(K)

#             # Plotting
#             # Plot scaled data on the first plot window
#             axs1.set_xlabel('x [mm]')
#             axs1.set_ylabel('y [mm]')
#             axs1.set_title('raw profile')
#             axs1.plot(x_raw, y_raw)
#             axs1.vlines(x_min_limit, ymax=max(y_raw), ymin=min(y_raw), linestyles='dashed')
#             axs1.vlines(x_max_limit, ymax=max(y_raw), ymin=min(y_raw), linestyles='dashed')
#             canvas1.draw()  # Redraw canvas with new plot

#             # Plot scaled data on the second plot window
#             axs2.set_xlabel('x [mm]')
#             axs2.set_ylabel('y [mm]')
#             axs2.set_title('cleaned edge')
#             axs2.plot(x_shift, y_shift, x_tip, y_tip, 'kx')
#             axs2.plot(x_lin_left, y_lin_left, 'k--', x_lin_right, y_lin_right, 'k--')
#             axs2.plot(x_relief_left, y_relief_left, 'kx', x_relief_right, y_relief_right, 'kx')
#             axs2.plot(x_left, y_left, 'ko')
#             axs2.plot(x_right, y_right, 'ko')
#             circle_finale = plt.Circle((center[0], center[1]), radius, color='b', fill=False)
#             axs2.plot(center[0], center[1], 'k+')
#             axs2.add_patch(circle_finale)
#             axs2.text(center[0], center[1], 'radius: {:.3f}\nK: {:.3f}'.format(radius, K), ha='right', va='bottom', color='red')
#             canvas2.draw()  # Redraw canvas with new plot
#             root.update()

#             # file export ?
#             if export == 1:
#                 save_path = result_exporter(file_list, result_file, result_radius, result_kappa)
#                 # update Info
#                 info_label.configure(text="Results are exported to" + str(save_path))
#             else:
#                 pass
#     except NameError:
#         info_label.configure(text = "No file or folder selected")

def result_exporter(file_list, result_file, result_radius, result_radii_ell, result_kappa):
    save_path = os.path.dirname(file_list[0])
    upper_folder = os.path.dirname(save_path)
    with open(os.path.join(upper_folder, 'Kantenmessung.txt'),'w') as file:
        file.write("File\tDate-Time\tRadius [µm]\trA\trB\tK-Faktor\n")  # Writing the header

        # Writing data from both lists into the file
        for file_name, radius, radii_ell, kappa in zip(result_file, result_radius, result_radii_ell, result_kappa):
            modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(save_path, file_name)))
            if radius != 'nan':  # is valid edge rounding
                file.write("{}\t{}\t{:.0f}\t{:.0f}\t{:.0f}\t{}\n".format(file_name,modification_time, \
                    radius*1000, radii_ell[0]*1000, radii_ell[1]*1000, str(round(kappa, 3)).replace('.',',')))
            else:  # could not measure edge rounding
                file.write(f"{file_name}\t{modification_time}\t0\t1\n")  # assume sharp, symmetrical edge
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

def edge_detection(cut_value, no_flank, x_raw, y_raw):
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

    # linear model of flanks
    m_left, b_left = np.polyfit(x_left, y_left, 1)
    m_right, b_right = np.polyfit(x_right, y_right, 1)
    x_intersect = (b_right - b_left)/(m_left - m_right)
    x_lin_left = x_shift[np.where(x_shift < x_intersect)]
    x_lin_right = x_shift[np.where(x_shift > x_intersect)]
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

def browse_up(current_text):
    current_index = 0
    for i in range(len(file_list)):
        if current_text == os.path.basename(file_list[i]):
            current_index = i
            break
    next_index = current_index + 1
    if next_index == len(file_list):
        next_index = 0
        browse_label.configure(text=os.path.basename(file_list[next_index]))
    else:
        browse_label.configure(text=os.path.basename(file_list[next_index]))


    # Plotting
    axs1.clear()  # see every plot seperatly
    axs2.clear()
    axs1.set_xlabel('x [mm]')
    axs1.set_ylabel('y [mm]')
    axs1.set_title('raw profile')
    axs1.plot(x_raw_list[next_index], y_raw_list[next_index])
    axs1.vlines(x_min_limit_list[next_index], ymax=max(y_raw_list[next_index]), ymin=min(y_raw_list[next_index]),
                linestyles='dashed')
    axs1.vlines(x_max_limit_list[next_index], ymax=max(y_raw_list[next_index]), ymin=min(y_raw_list[next_index]),
                linestyles='dashed')
    canvas1.draw()  # Redraw canvas with new plot

    # Plot scaled data on the second plot window
    axs2.set_xlabel('x [mm]')
    axs2.set_ylabel('y [mm]')
    axs2.set_title('clean edge')
    axs2.plot(x_shift_list[next_index], y_shift_list[next_index])
    axs2.plot(x_lin_left_list[next_index], y_lin_left_list[next_index], 'k--', x_lin_right_list[next_index],
              y_lin_right_list[next_index], 'k--')
    axs2.plot(x_edge_list[next_index], y_edge_list[next_index], 'r')
    axs2.plot(x_left_list[next_index], y_left_list[next_index], 'c')
    axs2.plot(x_right_list[next_index], y_right_list[next_index], 'c')
    axs2.plot(x_tip_list[next_index], y_tip_list[next_index], 'rx', markersize=10)
    axs2.plot(x_relief_left_list[next_index], y_relief_left_list[next_index], 'bx', markersize=10)
    axs2.plot(x_relief_right_list[next_index], y_relief_right_list[next_index], 'bx', markersize=10)
    if np.isnan(result_radius[next_index]):
        axs2.text(0.5, 0, f'no calculation: {err_msg_list[next_index]} ', ha='left', va='bottom', color='red')
    else:
        circle_finale = plt.Circle((center_list[next_index][0], center_list[next_index][1]),
                                   result_radius[next_index], color='b', fill=False)
        ellipsis_finale = PlotEllipsis(center_ell_list[next_index], radii_ell_list[next_index],
                                       theta_ell_list[next_index])
        axs2.plot(center_list[next_index][0], center_list[next_index][1], 'k+')
        axs2.add_patch(circle_finale)
        axs2.add_patch(ellipsis_finale)
        # axs2.text(center[0], center[1], f'radius: {round(radius, 2)}', ha='right', va='top', color='red')
        axs2.text(0, y_tip_list[next_index],
                  'r\u03b2 = {:.0f} \u03bcm\nK = {:.3f}'.format(result_radius[next_index] * 1000,
                                                                result_kappa[next_index]), ha='left', va='bottom',
                  color='red')
    canvas2.draw()  # Redraw canvas with new plot
    root.update()

def browse_down(current_text):
    current_index = 0
    for i in range(len(file_list)):
        if current_text == os.path.basename(file_list[i]):
            current_index = i
            break
    next_index = current_index - 1
    if next_index < 0:
        next_index = len(file_list)-1
        browse_label.configure(text=os.path.basename(file_list[next_index]))
    else:
        browse_label.configure(text=os.path.basename(file_list[next_index]))

    # Plotting
    axs1.clear()  # see every plot seperatly
    axs2.clear()
    axs1.set_xlabel('x [mm]')
    axs1.set_ylabel('y [mm]')
    axs1.set_title('raw profile')
    axs1.plot(x_raw_list[next_index], y_raw_list[next_index])
    axs1.vlines(x_min_limit_list[next_index], ymax=max(y_raw_list[next_index]), ymin=min(y_raw_list[next_index]), linestyles='dashed')
    axs1.vlines(x_max_limit_list[next_index], ymax=max(y_raw_list[next_index]), ymin=min(y_raw_list[next_index]), linestyles='dashed')
    canvas1.draw()  # Redraw canvas with new plot

    # Plot scaled data on the second plot window
    axs2.set_xlabel('x [mm]')
    axs2.set_ylabel('y [mm]')
    axs2.set_title('clean edge')
    axs2.plot(x_shift_list[next_index], y_shift_list[next_index])
    axs2.plot(x_lin_left_list[next_index], y_lin_left_list[next_index], 'k--', x_lin_right_list[next_index], y_lin_right_list[next_index], 'k--')
    axs2.plot(x_edge_list[next_index], y_edge_list[next_index], 'r')
    axs2.plot(x_left_list[next_index], y_left_list[next_index], 'c')
    axs2.plot(x_right_list[next_index], y_right_list[next_index], 'c')
    axs2.plot(x_tip_list[next_index], y_tip_list[next_index], 'rx', markersize=10)
    axs2.plot(x_relief_left_list[next_index], y_relief_left_list[next_index], 'bx', markersize=10)
    axs2.plot(x_relief_right_list[next_index], y_relief_right_list[next_index], 'bx', markersize=10)
    if np.isnan(result_radius[next_index]):
        axs2.text(0.5, 0, f'no calculation: {err_msg_list[next_index]} ', ha='left', va='bottom', color='red')
    else:
        circle_finale = plt.Circle((center_list[next_index][0], center_list[next_index][1]), result_radius[next_index], color='b', fill=False)
        ellipsis_finale = PlotEllipsis(center_ell_list[next_index], radii_ell_list[next_index], theta_ell_list[next_index])
        axs2.plot(center_list[next_index][0], center_list[next_index][1], 'k+')
        axs2.add_patch(circle_finale)
        axs2.add_patch(ellipsis_finale)
        # axs2.text(center[0], center[1], f'radius: {round(radius, 2)}', ha='right', va='top', color='red')
        axs2.text(0, y_tip_list[next_index], 'r\u03b2 = {:.0f} \u03bcm\nK = {:.3f}'.format(result_radius[next_index] * 1000, result_kappa[next_index]), ha='left',va='bottom', color='red')
    canvas2.draw()  # Redraw canvas with new plot
    root.update()






# GUI Setup
root = tk.Tk()
root.geometry("1410x530")
color = "snow3"
root.configure(bg=color)

### Control Panel ###
frame_control = tk.Frame(root, bg = color)
frame_control.pack(side='left', fill='y')
## Selection Frame
frame_selection = tk.Frame(frame_control, bg = color)
frame_selection.pack(side='top', fill='x')
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
file_tree = ttk.Treeview(frame_file_shower, columns=('Name', 'Radius', 'K-Faktor'), show='headings')
file_tree.heading('Name', text='Name')
file_tree.heading('Radius', text='Radius')
file_tree.heading('K-Faktor', text='K-Faktor')
file_tree.column('Name', width=50)
file_tree.column('Radius', width=70)
file_tree.column('K-Faktor', width=70)
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
calc_frame = tk.Frame(frame_control, bg = color)
calc_frame.pack(side='top', fill='x')
calc_button_single = tk.Button(calc_frame, text="Calculate (fit)", command=lambda: fit_calculation(export_checker.get()))
calc_button_single.pack(side='left', padx=5, pady=5)
# Quit application
quit_frame = tk.Frame(frame_control, bg = color)
quit_frame.pack(side='top', fill='x')
quit_button_single = tk.Button(quit_frame, text="Quit", command=lambda: exit())
quit_button_single.pack(side='left', padx=5, pady=5)

### Plot Panel ###
frame_plot_all = tk.Frame(root, bg=color)
frame_plot_all.pack(side='top', fill='both')
# Figure 1
frame_left = tk.Frame(root, bg=color)
frame_left.pack(side='left', fill='both', pady=5)
fig1, axs1 = plt.subplots(figsize=(6, 4.5))
axs1.set_xlabel('x [mm]')
axs1.set_ylabel('y [mm]')
axs1.set_title('raw profile')
canvas1 = FigureCanvasTkAgg(fig1, master=frame_left)
canvas1.get_tk_widget().pack(side='top', padx=5, pady=0)
axs1.axis('equal')
# Add toolbar for Figure 1
toolbar1 = NavigationToolbar2Tk(canvas1, frame_left)
toolbar1.update()
toolbar1.pack(side='top', padx=5, pady=0)

# Figure 2
frame_right= tk.Frame(root, bg=color)
frame_right.pack(side='left', fill='both', pady=5)
fig2, axs2 = plt.subplots(figsize=(6, 4.5))
axs2.set_xlabel('x [mm]')
axs2.set_ylabel('y [mm]')
axs2.set_title('clean edge')
canvas2 = FigureCanvasTkAgg(fig2, master=frame_right)
canvas2.get_tk_widget().pack(side='top', padx=5, pady=0)
axs2.axis('equal')
# Add toolbar for Figure 2
toolbar2 = NavigationToolbar2Tk(canvas2, frame_right)
toolbar2.update()
toolbar2.pack(side='top', padx=5, pady=0)

### Print Panel ###
frame_print = tk.Frame(frame_plot_all, bg=color)
frame_print.pack(side='top', fill='x')
print_label = tk.Label(frame_print, text='Info:', bg=color)
print_label.pack(side='left', padx=5, pady=8)

#frame_info = tk.Frame(frame_plot_all, bg='snow3')
#frame_info.pack(side='top', fill='x')
info_label = ttk.Label(frame_print, text='Select file or folder to start...')
info_label.pack(side='left', padx=5, pady=8)

### Browse Elements ###
right_arrow_button = tk.Button(frame_print, text="-->", command=lambda: browse_up(browse_label.cget("text")))
right_arrow_button.pack(side='right', padx=5)

default_text = 'nothing to show :('
browse_label = ttk.Label(frame_print, text=default_text)
browse_label.pack(side='right',padx=5, pady=5)


left_arrow_button = tk.Button(frame_print, text="<--", command=lambda: browse_down(browse_label.cget("text")))
left_arrow_button.pack(side='right', padx=5)


root.mainloop()