"""
This scripts downloads and evaluates W&B run data to produce plots and tables used in the original paper.
"""
import numpy as np
import wandb
import matplotlib.pyplot as plt


def get_data(project):
    from tqdm import tqdm
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(project)

    info_list = []

    # history_list = []
    for run in tqdm(runs, desc='Downloading data...'):
        config = {k:v for k,v in run.config.items() if not k.startswith('_')}
        info_dict = {'metrics':run.history(), 'config':config}
        info_list.append((run.name,info_dict))
    return info_list

all_df  = get_data("confusezius/RevisitDML")

names_to_check = list(np.unique(['_s'.join(x[0].split('_s')[:-1]) for x in all_df]))
metrics        = ['Test: discriminative_e_recall: e_recall@1', 'Test: discriminative_e_recall: e_recall@2', \
                  'Test: discriminative_e_recall: e_recall@4', 'Test: discriminative_nmi: nmi', \
                  'Test: discriminative_f1: f1', 'Test: discriminative_mAP: mAP']
metric_names   = ['R@1', 'R@2', 'R@4', 'NMI', 'F1', 'mAP']

idxs = {x:[i for i,y in enumerate(all_df) if x=='_s'.join(y[0].split('_s')[:-1])] for x in names_to_check}
vals = {}
for group, runs in idxs.items():
    if 'CUB' in group:
        min_len = 40
    elif 'CAR' in group:
        min_len = 40
    elif 'SOP' in group:
        min_len = 40

    vals[group] = {metric_name:[] for metric_name in metric_names}
    vals[group]['Max_Epoch'] = []
    vals[group]['Intra_over_Inter'] = []
    vals[group]['Intra'] = []
    vals[group]['Inter'] = []
    vals[group]['Rho1'] = []
    vals[group]['Rho2'] = []
    vals[group]['Rho3'] = []
    vals[group]['Rho4'] = []
    for i,run in enumerate(runs):
        name, data = all_df[run]
        for metric,metric_name in zip(metrics, metric_names):
            if len(data['metrics']):
                sub_data = list(data['metrics'][metric])
                if len(sub_data)>min_len:
                    vals[group][metric_name].append(np.nanmax(sub_data))
                    if metric_name=='R@1':
                        r_argmax = np.nanargmax(sub_data)
                        vals[group]['Max_Epoch'].append(r_argmax)
                        vals[group]['Intra_over_Inter'].append(data['metrics']['Train: discriminative_dists: dists@intra_over_inter'][r_argmax])
                        vals[group]['Intra'].append(data['metrics']['Train: discriminative_dists: dists@intra'][r_argmax])
                        vals[group]['Inter'].append(data['metrics']['Train: discriminative_dists: dists@inter'][r_argmax])
                        vals[group]['Rho1'].append(data['metrics']['Train: discriminative_rho_spectrum: rho_spectrum@-1'][r_argmax])
                        vals[group]['Rho2'].append(data['metrics']['Train: discriminative_rho_spectrum: rho_spectrum@1'][r_argmax])
                        vals[group]['Rho3'].append(data['metrics']['Train: discriminative_rho_spectrum: rho_spectrum@2'][r_argmax])
                        vals[group]['Rho4'].append(data['metrics']['Train: discriminative_rho_spectrum: rho_spectrum@10'][r_argmax])
    vals[group] = {metric_name:(np.mean(metric_vals),np.std(metric_vals)) for metric_name, metric_vals in vals[group].items()}


###
cub_vals = {key:item for key,item in vals.items() if 'CUB' in key}
car_vals = {key:item for key,item in vals.items() if 'CAR' in key}
sop_vals = {key:item for key,item in vals.items() if 'SOP' in key}




##########
def name_filter(n):
    n = '_'.join(n.split('_')[1:])
    return n

def name_adjust(n, prep='', app='', for_plot=True):
    if 'Margin_b06_Distance' in n:
        t = 'Margin (D), \\beta=0.6' if for_plot else 'Margin (D, \\beta=0.6)'
    elif 'Margin_b12_Distance' in n:
        t = 'Margin (D), \\beta=1.2' if for_plot else 'Margin (D, \\beta=1.2)'
    elif 'ArcFace' in n:
        t = 'ArcFace'
    elif 'Histogram' in n:
        t = 'Histogram'
    elif 'SoftTriple' in n:
        t = 'SoftTriple'
    elif 'Contrastive' in n:
        t = 'Contrastive (D)'
    elif 'Triplet_Distance' in n:
        t = 'Triplet (D)'
    elif 'Quadruplet_Distance' in n:
        t = 'Quadruplet (D)'
    elif 'SNR_Distance' in n:
        t = 'SNR (D)'
    elif 'Triplet_Random' in n:
        t = 'Triplet (R)'
    elif 'Triplet_Semihard' in n:
        t = 'Triplet (S)'
    elif 'Triplet_Softhard' in n:
        t = 'Triplet (H)'
    elif 'Softmax' in n:
        t = 'Softmax'
    elif 'MS' in n:
        t = 'Multisimilarity'
    else:
        t = '_'.join(n.split('_')[1:])

    if for_plot:
        t = r'${0}$'.format(t)

    return prep+t+app


########
def single_table(vals):
    print_str = ''
    for name,metrics in vals.items():
        prep = 'R-' if 'reg_' in name else ''
        name = name_adjust(name, for_plot=False, prep=prep)
        add = '{0} & ${1:2.2f}\\pm{2:2.2f}$ & ${3:2.2f}\\pm{4:2.2f}$ & ${5:2.2f}\\pm{6:2.2f}$ & ${7:2.2f}\\pm{8:2.2f}$ & ${9:2.2f}\\pm{10:2.2f}$ & ${11:2.2f}\\pm{12:2.2f}$'.format(name,
                                                                                                                                     metrics['R@1'][0]*100, metrics['R@1'][1]*100,
                                                                                                                                     metrics['R@2'][0]*100, metrics['R@2'][1]*100,
                                                                                                                                     metrics['F1'][0]*100, metrics['F1'][1]*100,
                                                                                                                                     metrics['mAP'][0]*100, metrics['mAP'][1]*100,
                                                                                                                                     metrics['NMI'][0]*100, metrics['NMI'][1]*100,
                                                                                                                                     metrics['Max_Epoch'][0], metrics['Max_Epoch'][1])
        print_str += add
        print_str += '\\'
        print_str += '\\'
        print_str += '\n'
    return print_str

print(single_table(cub_vals))
print(single_table(car_vals))
print(single_table(sop_vals))



########
def shared_table():
    cub_names, car_names, sop_names = list(cub_vals.keys()), list(car_vals.keys()), list(sop_vals.keys())
    cub_names  = [name_adjust(n, for_plot=False, prep='R-' if 'reg_' in n else '') for n in cub_names]
    cub_vals_2 = {name_adjust(n, for_plot=False, prep='R-' if 'reg_' in n else ''):item for n,item in cub_vals.items()}
    car_names = [name_adjust(n, for_plot=False, prep='R-' if 'reg_' in n else '') for n in car_names]
    car_vals_2 = {name_adjust(n, for_plot=False, prep='R-' if 'reg_' in n else ''):item for n,item in car_vals.items()}
    sop_names = [name_adjust(n, for_plot=False, prep='R-' if 'reg_' in n else '') for n in sop_names]
    sop_vals_2 = {name_adjust(n, for_plot=False, prep='R-' if 'reg_' in n else ''):item for n,item in sop_vals.items()}
    cub_vvals, car_vvals, sop_vvals = list(cub_vals.values()), list(car_vals.values()), list(sop_vals.values())
    unique_names = np.unique(np.concatenate([cub_names, car_names, sop_names], axis=0).reshape(-1))
    unique_names = sorted([x for x in unique_names if 'R-' not in x]) + sorted([x for x in unique_names if 'R-' in x])

    print_str = ''

    for name in unique_names:
        cub_rm, cub_rs = ('{0:2.2f}'.format(cub_vals_2[name]['R@1'][0]*100), '{0:2.2f}'.format(cub_vals_2[name]['R@1'][1]*100)) if name in cub_vals_2 else ('-', '-')
        cub_nm, cub_ns = ('{0:2.2f}'.format(cub_vals_2[name]['NMI'][0]*100), '{0:2.2f}'.format(cub_vals_2[name]['NMI'][1]*100)) if name in cub_vals_2 else ('-', '-')
        car_rm, car_rs = ('{0:2.2f}'.format(car_vals_2[name]['R@1'][0]*100), '{0:2.2f}'.format(car_vals_2[name]['R@1'][1]*100)) if name in car_vals_2 else ('-', '-')
        car_nm, car_ns = ('{0:2.2f}'.format(car_vals_2[name]['NMI'][0]*100), '{0:2.2f}'.format(car_vals_2[name]['NMI'][1]*100)) if name in car_vals_2 else ('-', '-')
        sop_rm, sop_rs = ('{0:2.2f}'.format(sop_vals_2[name]['R@1'][0]*100), '{0:2.2f}'.format(sop_vals_2[name]['R@1'][1]*100)) if name in sop_vals_2 else ('-', '-')
        sop_nm, sop_ns = ('{0:2.2f}'.format(sop_vals_2[name]['NMI'][0]*100), '{0:2.2f}'.format(sop_vals_2[name]['NMI'][1]*100)) if name in sop_vals_2 else ('-', '-')

        add = '{0} & ${1}\\pm{2}$ & ${3}\\pm{4}$ & ${5}\\pm{6}$ & ${7}\\pm{8}$ & ${9}\\pm{10}$ & ${11}\\pm{12}$'.format(name,
                                                                                                                        cub_rm, cub_rs,
                                                                                                                        cub_nm, cub_ns,
                                                                                                                        car_rm, car_rs,
                                                                                                                        car_nm, car_ns,
                                                                                                                        sop_rm, sop_rs,
                                                                                                                        sop_nm, sop_ns)

        print_str += add
        print_str += '\\'
        print_str += '\\'
        print_str += '\n'
    return print_str

print(shared_table())




"""==================================================="""
def give_basic_metr(vals, key='CUB'):
    if key=='CUB':
        Basic  = sorted(list(filter(lambda x: '{}_'.format(key) in x, list(vals.keys()))))
    elif key=='CARS':
        Basic  = sorted(list(filter(lambda x: 'CARS_' in x, list(vals.keys()))))
    elif key=='SOP':
        Basic  = sorted(list(filter(lambda x: 'SOP_' in x, list(vals.keys()))))

    basic_recall      = np.array([vals[k]['R@1'][0] for k in Basic])
    basic_recall_err  = np.array([vals[k]['R@1'][1] for k in Basic])
    #
    basic_recall2 = np.array([vals[k]['R@2'][0] for k in Basic])
    basic_recall4 = np.array([vals[k]['R@4'][0] for k in Basic])
    basic_nmi     = np.array([vals[k]['NMI'][0] for k in Basic])
    basic_f1      = np.array([vals[k]['F1'][0] for k in Basic])
    basic_map     = np.array([vals[k]['mAP'][0] for k in Basic])

    mets = [basic_recall, basic_recall2, basic_recall4, basic_nmi, basic_f1, basic_map]

    return Basic, mets, basic_recall, basic_recall_err


def give_reg_metr(vals, key='CUB'):
    if key=='CUB':
        RhoReg = sorted(list(filter(lambda x: '{}reg_'.format(key) in x, list(vals.keys()))))
    elif key=='CARS':
        RhoReg = sorted(list(filter(lambda x: 'CARreg_' in x, list(vals.keys()))))
    elif key=='SOP':
        RhoReg = sorted(list(filter(lambda x: 'SOPreg_' in x, list(vals.keys()))))

    rho_recall      = np.array([vals[k]['R@1'][0] for k in RhoReg])
    rho_recall_err  = np.array([vals[k]['R@1'][1] for k in RhoReg])
    #
    rho_recall2 = np.array([vals[k]['R@2'][0] for k in RhoReg])
    rho_recall4 = np.array([vals[k]['R@4'][0] for k in RhoReg])
    rho_nmi     = np.array([vals[k]['NMI'][0] for k in RhoReg])
    rho_f1      = np.array([vals[k]['F1'][0] for k in RhoReg])
    rho_map     = np.array([vals[k]['mAP'][0] for k in RhoReg])

    mets = [rho_recall, rho_recall2, rho_recall4, rho_nmi, rho_f1, rho_map]

    return RhoReg, mets, rho_recall, rho_recall_err

cub_basic_names, cub_mets, cub_basic_recall, cub_basic_recall_err = give_basic_metr(cub_vals, key='CUB')
car_basic_names, car_mets, car_basic_recall, car_basic_recall_err = give_basic_metr(car_vals, key='CARS')
sop_basic_names, sop_mets, sop_basic_recall, sop_basic_recall_err = give_basic_metr(sop_vals, key='SOP')
cub_reg_names, cub_reg_mets, cub_reg_recall, cub_reg_recall_err = give_reg_metr(cub_vals, key='CUB')
car_reg_names, car_reg_mets, car_reg_recall, car_reg_recall_err = give_reg_metr(car_vals, key='CARS')
sop_reg_names, sop_reg_mets, sop_reg_recall, sop_reg_recall_err = give_reg_metr(sop_vals, key='SOP')








"""============================================================="""
# def produce_plot(basic_recall, basic_recall_err, BasicLosses, vals, ylim=[0.58, 0.635]):
#
#     intra  = np.array([vals[k]['Intra'][0] for k in BasicLosses])
#     inter  = np.array([vals[k]['Inter'][0] for k in BasicLosses])
#     ratio  = np.array([vals[k]['Intra_over_Inter'][0] for k in BasicLosses])
#     rho1  = np.array([vals[k]['Rho1'][0] for k in BasicLosses])
#     rho2  = np.array([vals[k]['Rho2'][0] for k in BasicLosses])
#     rho3  = np.array([vals[k]['Rho3'][0] for k in BasicLosses])
#     rho4  = np.array([vals[k]['Rho4'][0] for k in BasicLosses])
#
#     def comp(met):
#         sort = np.argsort(met)
#         corr = np.corrcoef(met[sort],basic_recall[sort])[0,1]
#         m,b  = np.polyfit(met[sort], basic_recall[sort], 1)
#         lim  = [np.min(met)*0.9, np.max(met)*1.1]
#         x    = np.linspace(lim[0], lim[1], 50)
#         linfit = m*x + b
#         return sort, corr, linfit, x, lim
#
#     intra_sort, intra_corr, intra_linfit, intra_x, intra_lim = comp(intra)
#     inter_sort, inter_corr, inter_linfit, inter_x, inter_lim = comp(inter)
#     ratio_sort, ratio_corr, ratio_linfit, ratio_x, ratio_lim = comp(ratio)
#     rho1_sort, rho1_corr, rho1_linfit, rho1_x, rho1_lim = comp(rho1)
#     rho2_sort, rho2_corr, rho2_linfit, rho2_x, rho2_lim = comp(rho2)
#     rho3_sort, rho3_corr, rho3_linfit, rho3_x, rho3_lim = comp(rho3)
#     rho4_sort, rho4_corr, rho4_linfit, rho4_x, rho4_lim = comp(rho4)
#
#
#
#     f,ax = plt.subplots(1,4)
#     # f,ax = plt.subplots(1,7)
#     colors = np.array([np.random.rand(3,) for _ in range(len(basic_recall))])
#     for i in range(len(colors)):
#         ax[0].errorbar(intra[intra_sort][i], basic_recall[intra_sort][i], yerr=basic_recall_err[intra_sort][i], fmt='o', color=colors[intra_sort][i], ecolor='gray', elinewidth=3, capsize=0, label='Basic Criteria', markersize=8)
#         ax[1].errorbar(inter[inter_sort][i], basic_recall[inter_sort][i], yerr=basic_recall_err[inter_sort][i], fmt='o', color=colors[inter_sort][i], ecolor='gray', elinewidth=3, capsize=0, label='Basic Criteria', markersize=8)
#         ax[2].errorbar(ratio[ratio_sort][i], basic_recall[ratio_sort][i], yerr=basic_recall_err[ratio_sort][i], fmt='o', color=colors[ratio_sort][i], ecolor='gray', elinewidth=3, capsize=0, label='Basic Criteria', markersize=8)
#         # ax[3].errorbar(rho1[rho1_sort][i], basic_recall[rho1_sort][i], yerr=basic_recall_err[rho1_sort][i], fmt='o', color=colors[rho1_sort][i], ecolor='gray', elinewidth=3, capsize=0, label='Basic Criteria', markersize=8)
#         # ax[4].errorbar(rho2[rho2_sort][i], basic_recall[rho2_sort][i], yerr=basic_recall_err[rho2_sort][i], fmt='o', color=colors[rho2_sort][i], ecolor='gray', elinewidth=3, capsize=0, label='Basic Criteria', markersize=8)
#         ax[3].errorbar(rho3[rho3_sort][i], basic_recall[rho3_sort][i], yerr=basic_recall_err[rho3_sort][i], fmt='o', color=colors[rho3_sort][i], ecolor='gray', elinewidth=3, capsize=0, label='Basic Criteria', markersize=8)
#         # ax[6].errorbar(rho4[rho4_sort][i], basic_recall[rho4_sort][i], yerr=basic_recall_err[rho4_sort][i], fmt='o', color=colors[rho4_sort][i], ecolor='gray', elinewidth=3, capsize=0, label='Basic Criteria', markersize=8)
#     ax[1].set_yticks([])
#     ax[2].set_yticks([])
#     ax[3].set_yticks([])
#     # ax[4].set_yticks([])
#     # ax[5].set_yticks([])
#     # ax[6].set_yticks([])
#     ax[0].plot(intra_x, intra_linfit, 'k--', alpha=0.5, linewidth=3)
#     ax[1].plot(inter_x, inter_linfit, 'k--', alpha=0.5, linewidth=3)
#     ax[2].plot(ratio_x, ratio_linfit, 'k--', alpha=0.5, linewidth=3)
#     # ax[3].plot(rho1_x, rho1_linfit, 'k--', alpha=0.5, linewidth=3)
#     ax[3].plot(rho2_x, rho2_linfit, 'k--', alpha=0.5, linewidth=3)
#     # ax[5].plot(rho3_x, rho3_linfit, 'k--', alpha=0.5, linewidth=3)
#     # ax[6].plot(rho4_x, rho4_linfit, 'k--', alpha=0.5, linewidth=3)
#     ax[0].text('Correlation: {0:2.2f}'.format(intra_corr), fontsize=18)
#     ax[1].text('Correlation: {0:2.2f}'.format(inter_corr), fontsize=18)
#     ax[2].text('Correlation: {0:2.2f}'.format(ratio_corr), fontsize=18)
#     # ax[3].text('Correlation: {0:2.2f}'.format(rho1_corr), fontsize=18)
#     ax[3].text('Correlation: {0:2.2f}'.format(rho2_corr), fontsize=18)
#     # ax[5].set_title('Correlation: {0:2.2f}'.format(rho3_corr), fontsize=18)
#     # ax[6].set_title('Correlation: {0:2.2f}'.format(rho4_corr), fontsize=18)
#     ax[0].set_title(r'$\pi_{intra}$', fontsize=18)
#     ax[1].set_title(r'$\pi_{inter}$', fontsize=18)
#     ax[2].set_title(r'$\pi_{ratio}$', fontsize=18)
#     ax[3].set_title(r'$\rho(\Phi)$', fontsize=18)
#     ax[0].set_ylabel('Recall Performance', fontsize=18)
#     for a in ax.reshape(-1):
#         a.tick_params(axis='both', which='major', labelsize=16)
#         a.tick_params(axis='both', which='minor', labelsize=16)
#         a.set_ylim(ylim)
#     f.set_size_inches(22,8)
#     f.tight_layout()


# produce_plot(cub_basic_recall, cub_basic_recall_err, cub_basic_names, cub_vals, ylim=[0.581,0.635])
# produce_plot(car_basic_recall, car_basic_recall_err, car_basic_names, car_vals, ylim=[0.70,0.82])
# produce_plot(sop_basic_recall, sop_basic_recall_err, sop_basic_names, sop_vals, ylim=[0.67,0.79])


def full_rel_plot():
    recallss= [cub_basic_recall, car_basic_recall, sop_basic_recall]
    rerrss  = [cub_basic_recall_err, car_basic_recall_err, sop_basic_recall_err]
    namess  = [cub_basic_names, car_basic_names, sop_basic_names]
    valss   = [cub_vals, car_vals, sop_vals]
    ylims   = [[0.581, 0.638],[0.70,0.82],[0.67,0.79]]

    f,axes = plt.subplots(3,4)
    for k,(ax, recalls, rerrs, names, vals, ylim) in enumerate(zip(axes, recallss, rerrss, namess, valss, ylims)):
        col = 'red' if k==3 else 'gray'

        intra  = np.array([vals[k]['Intra'][0] for k in names])
        inter  = np.array([vals[k]['Inter'][0] for k in names])
        ratio  = np.array([vals[k]['Intra_over_Inter'][0] for k in names])
        rho    = np.array([vals[k]['Rho3'][0] for k in names])

        def comp(met):
            sort = np.argsort(met)
            corr = np.corrcoef(met[sort],recalls[sort])[0,1]
            m,b  = np.polyfit(met[sort], recalls[sort], 1)
            lim  = [np.min(met)*0.9, np.max(met)*1.1]
            x    = np.linspace(lim[0], lim[1], 50)
            linfit = m*x + b
            return sort, corr, linfit, x, lim

        intra_sort, intra_corr, intra_linfit, intra_x, intra_lim = comp(intra)
        inter_sort, inter_corr, inter_linfit, inter_x, inter_lim = comp(inter)
        ratio_sort, ratio_corr, ratio_linfit, ratio_x, ratio_lim = comp(ratio)
        rho_sort, rho_corr, rho_linfit, rho_x, rho_lim = comp(rho)

        # f,ax = plt.subplots(1,7)
        colors = np.array([np.random.rand(3,) for _ in range(len(recalls))])
        for i in range(len(colors)):
            ax[0].errorbar(intra[intra_sort][i], recalls[intra_sort][i], yerr=rerrs[intra_sort][i], fmt='o', color=colors[intra_sort][i], ecolor='gray', elinewidth=3, capsize=0, label='Basic Criteria', markersize=8)
            ax[1].errorbar(inter[inter_sort][i], recalls[inter_sort][i], yerr=rerrs[inter_sort][i], fmt='o', color=colors[inter_sort][i], ecolor='gray', elinewidth=3, capsize=0, label='Basic Criteria', markersize=8)
            ax[2].errorbar(ratio[ratio_sort][i], recalls[ratio_sort][i], yerr=rerrs[ratio_sort][i], fmt='o', color=colors[ratio_sort][i], ecolor='gray', elinewidth=3, capsize=0, label='Basic Criteria', markersize=8)
            ax[3].errorbar(rho[rho_sort][i], recalls[rho_sort][i], yerr=rerrs[rho_sort][i], fmt='o', color=colors[rho_sort][i], ecolor='gray', elinewidth=3, capsize=0, label='Basic Criteria', markersize=8)
        ax[1].set_yticks([])
        ax[2].set_yticks([])
        ax[3].set_yticks([])
        ax[0].plot(intra_x, intra_linfit, 'k--', alpha=0.5, linewidth=3)
        ax[1].plot(inter_x, inter_linfit, 'k--', alpha=0.5, linewidth=3)
        ax[2].plot(ratio_x, ratio_linfit, 'k--', alpha=0.5, linewidth=3)
        ax[3].plot(rho_x, rho_linfit, 'r--', alpha=0.5, linewidth=3)

        ax[0].text(intra_lim[1]-0.7*(intra_lim[1]-intra_lim[0]),ylim[0]+0.05*(ylim[1]-ylim[0]),'Corr: {0:1.2f}'.format(intra_corr), bbox=dict(facecolor='gray', alpha=0.5), fontsize=26)
        ax[1].text(inter_lim[1]-0.7*(inter_lim[1]-inter_lim[0]),ylim[0]+0.05*(ylim[1]-ylim[0]),'Corr: {0:1.2f}'.format(inter_corr), bbox=dict(facecolor='gray', alpha=0.5), fontsize=26)
        ax[2].text(ratio_lim[1]-0.7*(ratio_lim[1]-ratio_lim[0]),ylim[0]+0.05*(ylim[1]-ylim[0]),'Corr: {0:1.2f}'.format(ratio_corr), bbox=dict(facecolor='gray', alpha=0.5), fontsize=26)
        ax[3].text(rho_lim[1]-0.7*(rho_lim[1]-rho_lim[0]),ylim[0]+0.05*(ylim[1]-ylim[0]),'Corr: {0:1.2f}'.format(rho_corr),   bbox=dict(facecolor='red', alpha=0.5), fontsize=26)

        if k==0:
            ax[0].set_title(r'$\pi_{intra}$', fontsize=26)
            ax[1].set_title(r'$\pi_{inter}$', fontsize=26)
            ax[2].set_title(r'$\pi_{ratio}$', fontsize=26)
            ax[3].set_title(r'$\rho(\Phi)$', fontsize=26, color='red')
        if k==0:
            ax[0].set_ylabel('CUB200-2011 R@1', fontsize=23)
        elif k==1:
            ax[0].set_ylabel('CARS196 R@1', fontsize=23)
        elif k==2:
            ax[0].set_ylabel('SOP R@1', fontsize=23)
        for a in ax.reshape(-1):
            a.tick_params(axis='both', which='major', labelsize=20)
            a.tick_params(axis='both', which='minor', labelsize=20)
            a.set_ylim(ylim)
    f.set_size_inches(21,15)
    f.tight_layout()

    f.savefig('comp_metric_relation.pdf')
    f.savefig('comp_metric_relation.png')

full_rel_plot()











"""================================================"""
import itertools as it
cub_corr_mat = np.corrcoef(cub_mets)
f,ax         = plt.subplots(1,3)
ax[0].imshow(cub_corr_mat, vmin=0, vmax=1, cmap='plasma')
corr_x = [0,1,2,3,4,5]
ax[0].set_xticklabels(metric_names)
ax[0].set_yticklabels(metric_names)
ax[0].set_xticks(corr_x)
ax[0].set_yticks(corr_x)
ax[0].set_xlim([-0.5,5.5])
ax[0].set_ylim([-0.5,5.5])
cs = list(it.product(corr_x, corr_x))
for c in cs:
    ax[0].text(c[0]-0.2, c[1]-0.11, '{0:1.2f}'.format(cub_corr_mat[c[0], c[1]]), fontsize=18)
ax[0].tick_params(axis='both', which='major', labelsize=18)
ax[0].tick_params(axis='both', which='minor', labelsize=18)
car_corr_mat = np.corrcoef(car_mets)
ax[1].imshow(car_corr_mat, vmin=0, vmax=1, cmap='plasma')
corr_x = [0,1,2,3,4,5]
ax[1].set_xticklabels(metric_names)
ax[1].set_yticklabels(metric_names)
ax[1].set_xticks(corr_x)
ax[1].set_yticks(corr_x)
ax[1].set_xlim([-0.5,5.5])
ax[1].set_ylim([-0.5,5.5])
cs = list(it.product(corr_x, corr_x))
for c in cs:
    ax[1].text(c[0]-0.2, c[1]-0.11, '{0:1.2f}'.format(car_corr_mat[c[0], c[1]]), fontsize=18)
ax[1].tick_params(axis='both', which='major', labelsize=18)
ax[1].tick_params(axis='both', which='minor', labelsize=18)
sop_corr_mat = np.corrcoef(sop_mets)
ax[2].imshow(sop_corr_mat, vmin=0, vmax=1, cmap='plasma')
corr_x = [0,1,2,3,4,5]
ax[2].set_xticklabels(metric_names)
ax[2].set_yticklabels(metric_names)
ax[2].set_xticks(corr_x)
ax[2].set_yticks(corr_x)
ax[2].set_xlim([-0.5,5.5])
ax[2].set_ylim([-0.5,5.5])
cs = list(it.product(corr_x, corr_x))
for c in cs:
    ax[2].text(c[0]-0.2, c[1]-0.11, '{0:1.2f}'.format(sop_corr_mat[c[0], c[1]]), fontsize=18)
ax[2].tick_params(axis='both', which='major', labelsize=18)
ax[2].tick_params(axis='both', which='minor', labelsize=18)
ax[0].set_title('CUB200-2011', fontsize=22)
ax[1].set_title('CARS196', fontsize=22)
ax[2].set_title('Stanford Online Products', fontsize=22)
f.set_size_inches(22,8)
f.tight_layout()
f.savefig('metric_correlation_matrix.pdf')
f.savefig('metric_correlation_matrix.png')










"""=================================================="""
####
recallss, valss = [cub_basic_recall, car_basic_recall, sop_basic_recall], [cub_vals, car_vals, sop_vals]
errss           = [cub_basic_recall_err, car_basic_recall_err, sop_basic_recall_err]
namess          = [cub_basic_names, car_basic_names, sop_basic_names]
reg_recallss, reg_valss = [cub_reg_recall, car_reg_recall, sop_reg_recall], [cub_vals, car_vals, sop_vals]
reg_errss           = [cub_reg_recall_err, car_reg_recall_err, sop_reg_recall_err]
reg_namess          = [cub_reg_names, car_reg_names, sop_reg_names]

####
def plot(vals, recalls, errs, names, reg_vals=None, reg_recalls=None, reg_errs=None, reg_names=None, xlab=None, ylab=None, xlim=[0,1], ylim=[0,1], savename=None):
    from adjustText import adjust_text
    f, ax = plt.subplots(1)
    texts = []
    rho       = np.array([vals[k]['Rho3'][0] for k in names])
    adj_names = names

    nnames    = []
    for n in adj_names:
        nnames.append(name_adjust(n, prep='', app=''))
    print(nnames)
    ax.errorbar(rho, recalls, yerr=errs,color='deepskyblue',fmt='o',ecolor='deepskyblue',elinewidth=5,capsize=0,markersize=16,mec='k')
    recalls = np.array(recalls)
    for rho_v, rec_v, n in zip(rho, recalls, nnames):
        r = ax.text(rho_v, rec_v, n, fontsize=17, va='top', ha='left')
        # r = ax.text(rho_v, rec_v, n, fontsize=15, bbox=dict(facecolor='gray', alpha=0.5), va='left', ha='left')
        texts.append(r)

    if reg_names is not None:
        rho       = np.array([vals[k]['Rho3'][0] for k in reg_names])
        adj_names = ['_'.join(x.split('_')[1:]) for x in reg_names]

        nnames    = []
        for n in adj_names:
            nnames.append(name_adjust(n, prep='R-', app=''))
        ax.errorbar(rho, reg_recalls, yerr=reg_errs,color='orange',fmt='o',ecolor='gray',elinewidth=5,capsize=0,markersize=16,mec='k')
        for rho_v, rec_v, n in zip(rho, reg_recalls, nnames):
            r = ax.text(rho_v, rec_v, n, fontsize=17, va='top', ha='left', color='chocolate')
            texts.append(r)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    if xlab is not None:
        ax.set_xlabel(xlab, fontsize=20)
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize=20)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid()
    f.set_size_inches(25,5)
    f.tight_layout()
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='k', lw=1))
    f.savefig('{}.png'.format(savename))
    f.savefig('{}.pdf'.format(savename))

plot(valss[0], recallss[0], errss[0], namess[0], reg_valss[0], reg_recallss[0], reg_errss[0], reg_namess[0], xlab=r'$\rho(\Phi)$', ylab=r'$CUB200-2011, R@1$', xlim=[0,0.59], ylim=[0.58, 0.66], savename='Detailed_Rel_Recall_Rho_CUB')
plot(valss[1], recallss[1], errss[1], namess[1], reg_valss[1], reg_recallss[1], reg_errss[1], reg_namess[1], xlab=r'$\rho(\Phi)$', ylab=r'$CARS196, R@1$',     xlim=[0,0.59], ylim=[0.7, 0.84], savename='Detailed_Rel_Recall_Rho_CAR')
plot(valss[2], recallss[2], errss[2], namess[2], reg_valss[2], reg_recallss[2], reg_errss[2], reg_namess[2], xlab=r'$\rho(\Phi)$', ylab=r'$SOP, R@1$', xlim=[0,0.59], ylim=[0.67, 0.81], savename='Detailed_Rel_Recall_Rho_SOP')





"""=================================================="""
#### First Page Figure
plt.style.use('seaborn')
total_recall = np.array(cub_basic_recall.tolist() + cub_reg_recall.tolist())
total_err    = np.array(cub_basic_recall_err.tolist() + cub_reg_recall_err.tolist())
total_names  = np.array(cub_basic_names+cub_reg_names)
sort_idx = np.argsort(total_recall)
f, ax = plt.subplots(1)
basic_label, reg_label = False, False
for i,idx in enumerate(sort_idx):
    if 'reg_' not in total_names[idx]:
        if basic_label:
            ax.barh(i,total_recall[idx], xerr=total_err[idx], color='orange', alpha=0.6)
        else:
            ax.barh(i,total_recall[idx], xerr=total_err[idx], color='orange', alpha=0.6, label='Basic DML Criteria')
            basic_label = True
        ax.text(0.5703,i-0.2,name_adjust(total_names[idx]), fontsize=17)
    else:
        if reg_label:
            ax.barh(i,total_recall[idx], xerr=total_err[idx], color='forestgreen', alpha=0.8)
        else:
            ax.barh(i,total_recall[idx], xerr=total_err[idx], color='forestgreen', alpha=0.8, label='Regularized Variant')
            reg_label = True
        ax.text(0.5703,i-0.2,name_adjust(total_names[idx], prep='R-'), fontsize=17)
ax.legend(fontsize=20)
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xticks([0.58, 0.6, 0.62, 0.64])
ax.tick_params(axis='both', which='major', labelsize=22)
ax.tick_params(axis='both', which='minor', labelsize=22)
ax.set_title('CUB200-2011, R@1', fontsize=20)
ax.set_ylim([-0.5,22.5])
ax.set_xlim([0.57, 0.655])
f.set_size_inches(15,8)
f.tight_layout()
f.savefig('FirstPage.png')
f.savefig('FirstPage.pdf')
