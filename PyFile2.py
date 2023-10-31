import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time as TIME
from mpl_toolkits.mplot3d import Axes3D

start_time = TIME()


def CyclesToFailure(DOD):
    # Compute the battery lifetime consumption with reference to the battery cycles exploited
    return 15790 * np.exp(-11.96*DOD) + 2633 * np.exp(-1.699*DOD)


x_llp = np.arange(101)
loadCurve_titles = [100]
makePlot = 1
columns = len(loadCurve_titles) * 6
MA_opt_norm_bhut_jun15_20_10 = np.zeros((len(x_llp), columns))

min_PV = 100               # Min PV power simulated [kW]
max_PV = 300               # Max PV power simulated [kW]
step_PV = 5                # PV power simulation step [kW]
min_batt = 50              # Min Battery capacity simulated [kWh]
max_batt = 800             # Max Battery capacity simulated [kWh]
step_batt = 10             # Battery capacity simulation step [kWh]

n_PV = int(((max_PV - min_PV) / step_PV) + 1)
n_batt = int(((max_batt - min_batt) / step_batt) + 1)

load_curves_counter = -1

for year in loadCurve_titles:
    load_curves_counter += 1
    path_to_dataBase = ''
    irr = np.array(list(sio.loadmat(path_to_dataBase + 'solar_data_Phuntsholing_baseline.mat').values())[-1][0])
    filename = path_to_dataBase + 'LoadCurve_normalized_single_3percent_' + str(year) + '.mat'
    Load = np.array(list(sio.loadmat(filename).values())[-1][0])
    T_amb = np.array(list(sio.loadmat(path_to_dataBase + 'surface_temp_phuent_2004_hour.mat').values())[-1][0])

    EPV = np.zeros((n_PV, n_batt))
    ELPV = np.zeros((len(irr), n_PV, n_batt))
    LL = np.zeros((len(irr), n_PV, n_batt))
    batt_balance = np.zeros(len(irr))
    num_batt = np.zeros((n_PV, n_batt))
    SoC = np.zeros(len(Load))
    IC = np.zeros((n_PV, n_batt))
    YC = np.zeros((n_PV, n_batt))

    eff_BoS = 0.85
    T_ref = 20
    T_nom = 47
    coeff_T_pow = 0.004
    irr_nom = 0.8

    SoC_min = 0.4
    SoC_start = 1
    eff_char = 0.85
    eff_disch = 0.9
    max_y_repl = 5
    batt_ratio = 0.5
    eff_inv = 0.9
    costPV = 1000
    costINV = 500
    costOeM_spec = 50
    coeff_cost_BoSeI = 0.2
    costBatt_coef_a = 140
    costBatt_coef_b = 0
    LT = 20
    r_int = 0.06

    for PV_i in range(n_PV):
        PVpower_i = min_PV + PV_i * step_PV
        T_cell = T_amb + irr * (T_nom - T_ref) / irr_nom
        eff_cell = 1 - coeff_T_pow * (T_cell - T_ref)
        P_pv = irr * PVpower_i * eff_cell * eff_BoS
        batt_balance = Load / eff_inv - P_pv

        for batt_i in range(n_batt):
            batt_cap_i = min_batt + batt_i * step_batt
            EPV[PV_i, batt_i] = np.sum(P_pv)
            SoC[0] = SoC_start
            Pow_max = batt_ratio * batt_cap_i
            Den_rainflow = 0

            for t in range(len(Load)):
                if t > 7:
                    if all(batt_balance[t - i] > 0 for i in range(1, 9)) and batt_balance[t] < 0:
                        DoD = 1 - SoC[t]
                        cycles_failure = CyclesToFailure(DoD)
                        Den_rainflow += 1 / cycles_failure

                if batt_balance[t] < 0:
                    flow_from_batt = batt_balance[t] * eff_char
                    if abs(batt_balance[t]) > Pow_max and SoC[t] < 1:
                        flow_from_batt = Pow_max * eff_char
                        ELPV[t, PV_i, batt_i] += abs(batt_balance[t]) - Pow_max
                    try:
                        SoC[t + 1] = SoC[t] + abs(flow_from_batt) / batt_cap_i
                    except IndexError:
                        SoC = np.append(SoC, SoC[t] + abs(flow_from_batt) / batt_cap_i)
                    if SoC[t + 1] > 1:
                        ELPV[t, PV_i, batt_i] += (SoC[t + 1] - 1) * batt_cap_i / eff_char
                        SoC[t + 1] = 1
                else:
                    flow_from_batt = batt_balance[t] / eff_disch
                    if batt_balance[t] > Pow_max and SoC[t] > SoC_min:
                        flow_from_batt = Pow_max / eff_disch
                        LL[t, PV_i, batt_i] += (batt_balance[t] - Pow_max) * eff_inv

                    try:
                        SoC[t + 1] = SoC[t] - flow_from_batt / batt_cap_i
                    except IndexError:
                        SoC = np.append(SoC, SoC[t] - flow_from_batt / batt_cap_i)

                    if SoC[t + 1] < SoC_min:
                        LL[t, PV_i, batt_i] += (SoC_min - SoC[t + 1]) * batt_cap_i * eff_disch * eff_inv
                        SoC[t + 1] = SoC_min

            if batt_i == 1 and PV_i == 1:
                batt_balance_pos = np.maximum(batt_balance, 0)
                LL_this = LL[:, PV_i, batt_i]
                print(abs(np.sum(LL_this) / np.sum(Load)))
                print(len(LL_this[LL_this < 0]) / len(LL_this))

                if makePlot == 1:
                    plt.figure(1)
                    plt.plot(Load, color=np.array([72, 122, 255]) / 255)
                    plt.plot(P_pv, color=np.array([255, 192, 33]) / 255)
                    plt.plot(batt_balance_pos, color=np.array([178, 147, 68]) / 255)
                    plt.xlabel('Time over the year [hour]')
                    plt.ylabel('Energy [kWh]')
                    plt.title('Energy produced and estimated load profile over the year (2nd steps PV and Batt)')
                    plt.legend(['Load profile', 'Energy from PV', 'Energy flow from battery'])

                    free = np.minimum(Load, P_pv)
                    time = np.arange(len(irr))
                    free_area = np.trapz(free, time)
                    area_to_batt = np.trapz(P_pv, time) - free_area
                    area_load_needed_from_batt = np.trapz(Load, time) - free_area

                    unmet_load = area_load_needed_from_batt - area_to_batt
                    unmet_load_perc = unmet_load / np.trapz(Load, time) * 100

                    nr_days = len(irr) / 24
                    Load_av = np.zeros(24)
                    P_pv_av = np.zeros(24)
                    batt_balance_pos_av = np.zeros(24)

                    for hour in range(24):
                        hours_i = range(hour, int((nr_days - 1) * 24 + hour), 24)
                        for k in hours_i:
                            Load_av[hour] += Load[k]
                            P_pv_av[hour] += P_pv[k]
                            batt_balance_pos_av[hour] += batt_balance_pos[k]
                        Load_av[hour] /= nr_days
                        P_pv_av[hour] /= nr_days
                        batt_balance_pos_av[hour] /= nr_days

                    plt.figure(2)
                    plt.plot(Load_av, color=np.array([72, 122, 255]) / 255)
                    plt.plot(P_pv_av, color=np.array([255, 192, 33]) / 255)
                    plt.plot(batt_balance_pos_av, color=np.array([178, 147, 68]) / 255)
                    plt.xlabel('Time over the day [hour]')
                    plt.ylabel('Energy [kWh]')
                    plt.title('Energy produced and estimated load profile of an average day (2nd steps PV and Batt)')
                    plt.legend(['Load profile', 'Energy from PV', 'Energy flow in battery'])

                    plt.figure(3)
                    plt.plot(ELPV[:, PV_i, batt_i] / batt_cap_i + 1, color=np.array([142, 178, 68]) / 255)
                    plt.plot(LL[:, PV_i, batt_i] / batt_cap_i + SoC_min, color=np.array([255, 91, 60]) / 255)
                    plt.plot(SoC, color=np.array([64, 127, 255]) / 255)
                    plt.xlabel('Time over the year [hour]')
                    plt.ylabel('Power referred to State of Charge of the battery')
                    plt.legend(['Overproduction, not utilized', 'Loss of power', 'State of charge'])

            costBatt_tot = costBatt_coef_a * batt_cap_i + costBatt_coef_b
            peak = np.max(Load)
            costINV_tot = (peak / eff_inv) * costINV
            costPV_tot = costPV * PVpower_i
            costBoSeI = coeff_cost_BoSeI * (costBatt_tot + costINV_tot + costPV_tot)

            IC[PV_i, batt_i] = costPV_tot + costBatt_tot + costINV_tot + costBoSeI
            costOeM = costOeM_spec * PVpower_i
            years_to_go_batt = 1 / Den_rainflow

            if years_to_go_batt > max_y_repl:
                years_to_go_batt = max_y_repl

            num_batt[PV_i, batt_i] = np.ceil(LT / years_to_go_batt)

            for k in range(1, LT + 1):
                if k > years_to_go_batt:
                    YC[PV_i, batt_i] += costBatt_tot / ((1 + r_int) ** years_to_go_batt)
                    years_to_go_batt += years_to_go_batt
                YC[PV_i, batt_i] += costOeM / ((1 + r_int) ** k)

            YC[PV_i, batt_i] -= costBatt_tot * ((years_to_go_batt - LT) / years_to_go_batt) / (1 + r_int) ** LT
            YC[PV_i, batt_i] += costINV_tot / ((1 + r_int) ** (LT / 2))

    NPC = IC + YC
    CRF = (r_int * ((1 + r_int) ** LT)) / (((1 + r_int) ** LT) - 1)
    total_loss_load = np.sum(LL, axis=0)
    LLP = total_loss_load / np.sum(Load)
    LCoE = (NPC * CRF) / (np.sum(Load) - total_loss_load)

    opt_sol_counter = 0

    for a_x in range(len(x_llp)):
        # clear PV_opt;
        # In Python, you don't need to clear variables.
        # If you want to delete a variable, you can use `del variable_name`
        PV_opt = None

        LLP_target = x_llp[a_x] / 100
        LLP_var = 0.005
        posPV, posBatt = np.where((LLP_target - LLP_var < LLP) & (LLP < LLP_target + LLP_var))

        try:
            NPC_opt = np.min(np.diag(NPC[posPV, posBatt]))
        except ValueError:
            NPC_opt = np.array([])

        for i in range(len(posPV)):
            if NPC[posPV[i], posBatt[i]] == NPC_opt:
                PV_opt = posPV[i]
                Batt_opt = posBatt[i]

        if PV_opt is None:
            continue

        kW_opt = PV_opt * step_PV + min_PV
        kWh_opt = (Batt_opt) * step_batt + min_batt
        LLP_opt = LLP[PV_opt, Batt_opt]
        LCoE_opt = LCoE[PV_opt, Batt_opt]
        IC_opt = IC[PV_opt, Batt_opt]

        if NPC_opt is None:
            NPC_opt = np.nan

        opt_sol_counter += 1
        opt_sol = [LLP_opt, NPC_opt, kW_opt, kWh_opt, LCoE_opt, IC_opt]

        MA_opt_norm_bhut_jun15_20_10[opt_sol_counter, (6 * load_curves_counter):(6 * (load_curves_counter+1))] = opt_sol

    flag = True
    if flag:
        X, Y = np.meshgrid(np.arange(min_batt, max_batt + step_batt, step_batt),
                           np.arange(min_PV, max_PV + step_PV, step_PV))
        fig = plt.figure(4)
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, NPC)
        ax.set_title('Net Present Cost')
        ax.set_xlabel('Battery Bank size [kWh]')
        ax.set_ylabel('PV array size [kW]')

        # Set the font size, name and weight of the axes and title
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'

        fig = plt.figure(5)
        ax = fig.add_subplot(projection='3d')

        ax.plot_surface(X, Y, LLP)
        ax.set_title('Loss of Load Probability')
        ax.set_xlabel('Battery Bank size [kWh]')
        ax.set_ylabel('PV array size [kW]')

        # Set the font size, name and weight of the axes and title
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'

        fig = plt.figure(6)
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, LCoE)

        ax.set_title('Levelized Cost of Energy')
        ax.set_xlabel('Battery Bank size [kWh]')
        ax.set_ylabel('PV array size [kW]')

        # Set the font size, name and weight of the axes and title
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'

        fig = plt.figure(7)
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, num_batt)

        ax.set_title('Num. of battery employed due to lifetime limit')
        ax.set_xlabel('Battery Bank size [kWh]')
        ax.set_ylabel('PV array size [kW]')

        # Set the font size, name and weight of the axes and title
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'

budget = 800000
counter_fill = 0
counter_empty = 0
x_filled = []
y_filled = []
colour_filled = []
x_empty = []
y_empty = []
colour_empty = []

for i in range(n_PV):
    for j in range(n_batt):
        this_PV = min_PV + i * step_PV
        this_batt = min_batt + j * step_batt
        if NPC[i, j] <= budget:
            counter_fill += 1
            x_filled.append(this_batt)
            y_filled.append(this_PV)
            colour_filled.append(round(LLP[i, j] * 100, 1))
        else:
            counter_empty += 1
            x_empty.append(this_batt)
            y_empty.append(this_PV)
            colour_empty.append(round(LLP[i, j] * 100, 1))

if counter_fill + counter_empty != n_PV * n_batt:
    # toc
    # In Python, you can use the time module to measure elapsed time
    # For example: import time; start_time = time.time(); print(time.time() - start_time)
    raise ValueError('ERROR: The number of datapoints in the two subsets "filled" and "empty"'
                     ' does not add up to the original dataset.')

plt.figure(8)

# Plot the filled and empty scatter points based on the counter values

if counter_fill > 0:
    plt.scatter(x_filled, y_filled, c=colour_filled, marker='o')
if counter_empty > 0:
    # plt.scatter(x_empty, y_empty, c=colour_empty, marker='o', facecolors='none')
    # plt.scatter(x_empty, y_empty, edgecolors=colour_empty, marker='o', facecolors='none')
    plt.scatter(x_empty, y_empty, c=colour_empty, alpha=0.4, marker='o')

# Add a colorbar and label it
bar = plt.colorbar()
bar.set_label('Loss of Load Probability [%]')
bar.set_alpha(1)

# Label the x-axis and y-axis
plt.xlabel('Battery bank size [kWh]')
plt.ylabel('PV array size [kW]')

# Add a title with the budget value
plt.title(f'Systems with filled dots are within the budget of E{budget}')

# Set the font size, name and weight of the axes and title
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'

np.save('MA_opt_norm_bhut_jun15_20_10.npy', MA_opt_norm_bhut_jun15_20_10)

end_time = TIME()
total_time = end_time - start_time

print(f"Execution time in seconds: {total_time}")

plt.show()


