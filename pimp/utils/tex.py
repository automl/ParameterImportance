import os
import shutil
import json
import glob
import itertools
from collections import OrderedDict as od
import numpy as np
from pandas import DataFrame, set_option as pd_option, concat as pd_concat
import pandas as pd


def jon_dirs(directory):
    if not isinstance(directory, list):
        directory = [directory]
    for d in directory:
        p_dir = os.getcwd()
        os.chdir(d)
        dir_ = sorted(os.listdir('.'))
        count = 0
        for entry in dir_:
            if '_all' in entry:
                count = 0
                continue
            if os.path.isdir(entry):
                tmp = entry.split('_')
                if '2017' in entry:
                    tmp = tmp[:tmp.index('2017') - 1]
                    tmp.extend(['rerun', 'all'])
                else:
                    tmp = tmp[:-1]
                    tmp.extend(['all'])
                base = '_'.join(tmp)
                print(base)

                if not os.path.exists(base):
                    os.mkdir(base)
                    count += 1
                if count > 0:
                    for file in os.listdir(entry):
                        shutil.move(os.path.join(entry, file), os.path.join(base, file))
                    os.removedirs(entry)
        os.chdir(p_dir)


def load_data_frames(directory, rerun=False, show=True):
    pd_option('max_colwidth', 120)
    dirs = glob.glob(os.path.join(directory, '*/'))

    data_dict = {}
    name_dict = {}
    keys = sorted(['fanova', 'fanova_cut', 'ablation', 'forward-selection', 'incneighbor'])
    for dir_ in dirs:
        if rerun and 'rerun' not in dir_:
            continue
        elif not rerun and 'rerun' in dir_:
            continue
        if show:
            print('Loading data from %s' % dir_)
        dn = dir_.split(os.path.sep)[-2]
        data_dict[dn] = {'fanova': None, 'fanova_cut': None, 'ablation': None, 'incneighbor': None}
        name_dict[dn] = {'fanova': None, 'fanova_cut': None, 'ablation': None, 'incneighbor': None}
        json_files = glob.glob(os.path.join(dir_, '*.json'))
        try:
            rm = json_files.index(os.path.join(dir_, 'pimp_args.json'))
            del json_files[rm]
        except ValueError:
            pass
        json_files = sorted(json_files)
        key_idx = 0
        jso_idx = 0
        while key_idx < len(keys) and jso_idx < len(json_files):
            if keys[key_idx] in ['forward-selection']:
                key_idx += 1
                continue
            cut = False
            if keys[key_idx] in json_files[jso_idx]:
                with open(json_files[jso_idx], 'r') as fh:
                    tmp = json.load(fh)
                if keys[key_idx] == 'fanova_cut':
                    keys[key_idx] = 'fanova'
                    cut = True
                importance = od([(k, tmp[keys[key_idx]]['imp'][k]) for k in tmp[keys[key_idx]]['order']])
                name = [k for k in tmp[keys[key_idx]]['order']]
                if keys[key_idx] == 'ablation':
                    if '-target-' in importance:
                        del importance['-target-']
                        del name[-1]
                    if '-source-' in importance:
                        del importance['-source-']
                        del name[0]
                if cut:
                    keys[key_idx] = 'fanova_cut'
                data_dict[dn][keys[key_idx]] = importance
                name_dict[dn][keys[key_idx]] = name
                key_idx += 1
                jso_idx += 1
            elif key_idx > jso_idx:
                jso_idx += 1
            else:
                key_idx += 1
    df = DataFrame.from_dict(data_dict, orient='index')
    dff = DataFrame.from_dict(name_dict, orient='index')
    cols = list(df.columns)
    cols[cols.index('incneighbor')] = 'local improvement analysis'
    df.columns = cols
    dff.columns = cols
    return df, dff


def _on_instance_sets(pd, m, index, store_in=None, algo_name=None, show=True, diagonal=False):
    first = sorted(index)
    if not diagonal:
        first = sorted(index[:-1])
    for idx, i in enumerate(first):
        a = pd[m][i]
        if a is None:
            continue
        a_keys = np.array(list(a.keys()))
        a_vals = np.array(list(a.values()))
        for at in range(len(a_keys)):
            if '[' in a_keys[at]:
                a_keys = a_keys[:at]
                a_vals = a_vals[:at]
                break
        a_indices_1 = np.array(np.nonzero(a_vals >= 0.01)).squeeze()
        a_indices_5 = np.array(np.nonzero(a_vals >= 0.05)).squeeze()
        a_keys_1 = a_keys[a_indices_1].reshape(1, -1).flatten()
        a_keys_5 = a_keys[a_indices_5].reshape(1, -1).flatten()
        second = sorted(index)
        if not diagonal:
            second = sorted(index[idx + 1:])
        for j in second:
            b = pd[m][j]
            if b is None:
                continue
            if algo_name:
                i = i.replace('_' + algo_name, '')
                i = i.replace(algo_name + '_', '')
                i = i.replace(algo_name, '')
                j = j.replace('_' + algo_name, '')
                j = j.replace(algo_name + '_', '')
                j = j.replace(algo_name, '')
            i = i.replace('_all', '')
            if '\\_' not in i:
                i = i.replace('_', '\\_')
            j = j.replace('_all', '')
            if '\\_' not in j:
                j = j.replace('_', '\\_')
            if show:
                print(i, j)
            b_keys = np.array(list(b.keys()))
            b_vals = np.array(list(b.values()))
            for at in range(len(b_keys)):
                if '[' in b_keys[at]:
                    b_keys = b_keys[:at]
                    b_vals = b_vals[:at]
                    break
            b_indices_1 = np.array(np.nonzero(b_vals >= 0.01)).squeeze()
            b_indices_5 = np.array(np.nonzero(b_vals >= 0.05)).squeeze()
            b_keys_1 = b_keys[b_indices_1].reshape(1, -1).flatten()
            b_keys_5 = b_keys[b_indices_5].reshape(1, -1).flatten()
            intersection_1 = list(set.intersection(set(a_keys_1), set(b_keys_1)))
            union_1 = list(set.union(set(a_keys_1), set(b_keys_1)))
            tmp = []
            for key in intersection_1:
                if ';' in key:
                    tmp.extend(key.split('; '))
                else:
                    tmp.append(key)
            intersection_1 = np.array(tmp).flatten()
            tmp = []
            for key in union_1:
                if ';' in key:
                    tmp.extend(key.split('; '))
                else:
                    tmp.append(key)
            union_1 = np.array(tmp).flatten()
            intersection_5 = list(set.intersection(set(a_keys_5), set(b_keys_5)))
            union_5 = list(set.union(set(a_keys_5), set(b_keys_5)))
            tmp = []
            for key in intersection_5:
                if ';' in key:
                    tmp.extend(key.split('; '))
                else:
                    tmp.append(key)
            intersection_5 = np.array(tmp).flatten()
            tmp = []
            for key in union_5:
                if ';' in key:
                    tmp.extend(key.split('; '))
                else:
                    tmp.append(key)
            union_5 = np.array(tmp).flatten()
            if show:
                print('          >=5%/>=1%')
                print('intersect: %3d/%3d' % (len(intersection_5), len(intersection_1)))
                print('    union: %3d/%3d' % (len(union_5), len(union_1)))
                print()
            if store_in is not None:
                if m not in store_in:
                    store_in[m] = {}
                store_in[m][' / '.join([i, j])] = [len(intersection_5), len(union_5)]
    return store_in


def _pairwise(pd, methods, index, store_in=None, algo_name=None, show=True):
    for i in sorted(index):
        skip = False
        intersection_set_1 = []
        intersection_set_5 = []
        intersection_set_f = []
        intersection_set_a = []
        for m in methods:
            entry = pd[m][i]
            if entry is None:
                skip = True
                break
            keys = np.array(list(entry.keys()))
            vals = np.array(list(entry.values()))
            for at in range(len(keys)):
                if '[' in keys[at]:
                    keys = keys[:at]
                    vals = vals[:at]
                    break
            indices_1 = np.array(np.nonzero(vals >= 0.01)).squeeze()
            indices_5 = np.array(np.nonzero(vals >= 0.05)).squeeze()
            keys_1 = keys[indices_1].reshape(1, -1).flatten()
            keys_1 = [x for y in list(map(lambda x: x.split('; '), keys_1)) for x in y]
            keys_5 = keys[indices_5].reshape(1, -1).flatten()
            keys_5 = [x for y in list(map(lambda x: x.split('; '), keys_5)) for x in y]
            intersection_set_1.append(set(keys_1))
            intersection_set_5.append(set(keys_5))
            if m == 'ablation':
                intersection_set_a.append(set(keys_1))
                intersection_set_f.append(set(keys_5))
            else:
                intersection_set_a.append(set(keys_5))
                intersection_set_f.append(set(keys_1))
        if not skip:
            if algo_name:
                i = i.replace('_' + algo_name, '')
                i = i.replace(algo_name + '_', '')
                i = i.replace(algo_name, '')
            i = i.replace('_all', '')
            union_set_1 = set.union(*intersection_set_1)
            union_set_5 = set.union(*intersection_set_5)
            union_set_a = set.union(*intersection_set_a)
            union_set_f = set.union(*intersection_set_f)
            intersection_set_1 = set.intersection(*intersection_set_1)
            intersection_set_5 = set.intersection(*intersection_set_5)
            intersection_set_a = set.intersection(*intersection_set_a)
            intersection_set_f = set.intersection(*intersection_set_f)
            if show:
                print(i)
                print('          >=5%/>=1%')
                print('intersect: %3d/%3d' % (len(intersection_set_5), len(intersection_set_1)))
                print('    union: %3d/%3d' % (len(union_set_5), len(union_set_1)))
                print()
            if store_in is not None:
                name = (i, methods[0], methods[1])
                if name not in store_in:
                    store_in[name] = {}
                store_in[name][i.replace('_', '\\_')] = [len(intersection_set_5), len(union_set_5)]
    return store_in


def generate_table_structure(pd, methods=None, store_in=None, algo_name=None, show=True, diagonal=False):
    if methods is None:
        methods = ['fanova', 'ablation', 'fanova_cut', 'local improvement analysis']
        combos = [['fanova', 'ablation'], ['fanova_cut', 'ablation'],
                  ['fanova', 'local improvement analysis'], ['fanova_cut', 'local improvement analysis'],
                  ['ablation', 'local improvement analysis']]
        if show:
            print('#'*180)
        store_in = {}
        for method in methods:
            if show:
                print(method)
            generate_table_structure(pd, [method], store_in, algo_name, show, diagonal)
            if show:
                print('#'*180)
        for combo in combos:
            if show:
                print(combo)
            generate_table_structure(pd, list(combo), store_in, algo_name, show, diagonal)
            if show:
                print('#'*180)
    elif len(methods) == 1:
        m = methods[0]
        index = pd[m].index
        store_in = _on_instance_sets(pd, m, index, store_in, algo_name, show, diagonal)
    else:
        index = pd[methods[0]].index
        store_in = _pairwise(pd, methods, index, store_in, algo_name, show)
    return store_in


def merge_data(result):
    tmp = sorted(list(result.keys()), key=lambda x: (x[1], x[2], x[0]))
    combos = [['fanova', 'ablation'], ['fanova_cut', 'ablation'],
              ['fanova', 'local improvement analysis'], ['fanova_cut', 'local improvement analysis'],
              ['ablation', 'local improvement analysis']]
    skip = False
    try:
        del tmp[tmp.index('fanova')]
    except ValueError:
        skip = True
    try:
        del tmp[tmp.index('ablation')]
    except ValueError:
        skip = True
    try:
        del tmp[tmp.index('fanova_cut')]
    except ValueError:
        skip = True
    try:
        del tmp[tmp.index('local improvement analysis')]
    except ValueError:
        skip = True
    df_dict = {}
    if not skip:
        multi = []
        all = None
        prev = tmp[0]
        for combo in combos:
            tap = np.array(tmp)
            indices = list(set.intersection(set(list(np.where(tap[:, 1] == combo[0]))[0]),
                                            set(list(np.where(tap[:, 2] == combo[1]))[0])))
            tap = list(tap[indices])
            for benchmark in tap:
                cols = ['cap_', 'cup_']
                if 'fanova' == benchmark[1]:
                    cols[0], cols[1] = cols[0] + 'f', cols[1] + 'f'
                elif 'ab' in benchmark[1]:
                    cols[0], cols[1] = cols[0] + 'a', cols[1] + 'a'
                else:
                    cols[0], cols[1] = cols[0] + 'c', cols[1] + 'c'
                if 'ab' in benchmark[2]:
                    cols[0], cols[1] = cols[0] + 'a', cols[1] + 'a'
                else:
                    cols[0], cols[1] = cols[0] + 'l', cols[1] + 'l'
                benchmark = tuple(benchmark)
                if all is None:
                    all = DataFrame.from_dict(result[benchmark], orient='index')
                    all.columns = cols
                    idx = list(map(lambda y: y[1], sorted(enumerate(list(all.index)), key=lambda x: x[1])))
                    all = all.loc[idx]
                else:
                    d = DataFrame.from_dict(result[benchmark], orient='index')
                    d.columns = cols
                    all = pd_concat([all, d])
            multi.append(all.copy())
            all = None
        df_dict['joint'] = pd_concat(multi, axis=1)
    for key in ['ablation', 'fanova', 'fanova_cut', 'local improvement analysis']:
        try:
            df = DataFrame.from_dict(result[key], orient='index')
            df.columns = ['cap', 'cup']
            idx = list(df.index)
            idx_0 = sorted(enumerate(list(map(lambda x: x.split(' / '), idx))), key=lambda x: (x[1][0], x[1][1]))
            idx_0 = list(map(lambda x: idx[x[0]], idx_0))
            df = df.loc[idx_0]
            df_dict[key] = df
        except KeyError:
            pass
    return df_dict


def create_latex_output(result):
    dfs = merge_data(result)
    skip = False
    if 'joint' not in dfs:
        skip = True
    if not skip:
        df = dfs['joint']
        tmp = list(df.keys())
        for idx, t in enumerate(tmp):
            t = t.split('_')
            t[0] = '$\\' + t[0]
            t[1] = [t[1][0], t[1][1]]
            t[1][0] = '{' + t[1][0] + '_{c5\\%}'
            t[1][1] = t[1][1] + '_{c5\\%}}$'
            t[1] = ''.join(t[1])
            t = '_'.join(t)
            tmp[idx] = t
        df.columns = tmp
        print('\\begin{table}[htbp]')
        print('\\centering')
        print(df.to_latex(escape=False, column_format='l' + 'c'*len(df.columns)))
        print('\\caption{%s}' % 'Comparison of fANOVA and ablation results on the same instance sets.')
        print('\\end{table}')
        print()
    for key in ['ablation', 'fanova', 'fanova_cut', 'local improvement analysis']:
        try:
            df = dfs[key]
            col = df['cap']
            coll = df['cup']
            tmp_idx = list(map(lambda x: x.split(' / '), list(col.index)))
            idx = list(map(lambda y: y[0], tmp_idx))
            idx.append(tmp_idx[-1][1])
            idx = sorted(list(set(idx)))
            data = []
            for i in idx:
                data.append([])
                for j in idx:
                    tmp = ' / '.join([i, j])
                    pmt = ' / '.join([j, i])
                    if tmp in list(col.index):
                        data[-1].append(col.loc[tmp])
                    elif pmt in list(col.index):
                        data[-1].append(coll.loc[pmt])
                    else:
                        data[-1].append('-')
            tmp_df = DataFrame.from_records(data, columns=idx)
            tmp_df.index = idx
            df = tmp_df
            if 'fanova' in key:
                key = key.replace('fanova', 'fANOVA')
            key = key.replace('_', '\\_')
            print('\\begin{table}[htbp]')
            print('\\centering')
            print(df.to_latex(escape=False, column_format='cl' + 'c' * len(idx), multirow=True))
            print('\\caption{Comparison of %s results across datasets. The lower triangle shows '
                  'the size of the union of all important parameters of both sets. The upper triangle shows '
                  'the size of the intersection of all important parameters.}' % key)
            print('\\end{table}')
            print()
        except KeyError:
            pass


def get_latex_outputs(algo_names, diagonal=False):
    rs = {}
    for algo in algo_names:
        v, _ = load_data_frames(algo, show=False)
        section = algo
        if 'clasp' in algo:
            if '_asp' in algo:
                algo = 'clasp'
            else:
                algo = 'clasp-3.0.4-p8'
        if 'clasp' in section:
            if '_asp' in section:
                section = 'clasp ASP'
            else:
                if 'random' in section:
                    section = 'clasp RAND'
                else:
                    section = 'clasp HAND'
        if section == 'cplex':
            section = 'CPLEX'
        elif section == 'satenstein':
            section = 'SATenstein'
        print('\\section{%s}' % section.replace('_', '\\_'))
        create_latex_output(generate_table_structure(v, algo_name=algo, show=False, diagonal=diagonal))
        print('\\clearpage')
        print()


def collect_all_dfs(algo_names, diagonal=False):
    rs = {}
    for algo in algo_names:
        v, _ = load_data_frames(algo)
        res = merge_data(generate_table_structure(v, algo_name=algo, diagonal=diagonal))
        rs[algo] = res
    return rs


def create_stats_df(algo_names, show=True, diagonal=False):
    all_ = collect_all_dfs(algo_names, diagonal=diagonal)
    res = []
    ros = []
    names = []
    for idx, algo in enumerate(algo_names):
        try:
            stat = [np.mean(all_[algo]['ablation']['cap'] / all_[algo]['ablation']['cup']),
                    np.std(all_[algo]['ablation']['cap'] / all_[algo]['ablation']['cup'])]
        except KeyError:
            stat.append(None)
            stat = [None, None]
        try:
            stat.append(np.mean(all_[algo]['fanova']['cap'] / all_[algo]['fanova']['cup']))
            stat.append(np.std(all_[algo]['fanova']['cap'] / all_[algo]['fanova']['cup']))
        except KeyError:
            stat.append(None)
            stat.append(None)
        try:
            stat.append(np.mean(all_[algo]['fanova_cut']['cap'] / all_[algo]['fanova_cut']['cup']))
            stat.append(np.std(all_[algo]['fanova_cut']['cap'] / all_[algo]['fanova_cut']['cup']))
        except KeyError:
            stat.append(None)
            stat.append(None)
        try:
            stat.append(np.mean(all_[algo][
                                    'local improvement analysis'][
                                    'cap'] / all_[algo]['local improvement analysis']['cup']))
            stat.append(np.std(all_[algo][
                                   'local improvement analysis'][
                                   'cap'] / all_[algo]['local improvement analysis']['cup']))
        except KeyError:
            stat.append(None)
            stat.append(None)

        try:
            stot = [np.mean(all_[algo]['joint']['cap_fa'] / all_[algo]['joint']['cup_fa']),
                    np.std(all_[algo]['joint']['cap_fa'] / all_[algo]['joint']['cup_fa'])]
        except KeyError:
            stot = [None, None]
        try:
            stot.append(np.mean(all_[algo]['joint']['cap_fl'] / all_[algo]['joint']['cup_fl']))
            stot.append(np.std(all_[algo]['joint']['cap_fl'] / all_[algo]['joint']['cup_fl']))
        except KeyError:
            stot.append(None)
            stot.append(None)
        try:
            stot.append(np.mean(all_[algo]['joint']['cap_ca'] / all_[algo]['joint']['cup_ca']))
            stot.append(np.std(all_[algo]['joint']['cap_ca'] / all_[algo]['joint']['cup_ca']))
        except KeyError:
            stot.append(None)
            stot.append(None)
        try:
            stot.append(np.mean(all_[algo]['joint']['cap_cl'] / all_[algo]['joint']['cup_cl']))
            stot.append(np.std(all_[algo]['joint']['cap_cl'] / all_[algo]['joint']['cup_cl']))
        except KeyError:
            stot.append(None)
            stot.append(None)
        try:
            stot.append(np.mean(all_[algo]['joint']['cap_al'] / all_[algo]['joint']['cup_al']))
            stot.append(np.std(all_[algo]['joint']['cap_al'] / all_[algo]['joint']['cup_al']))
        except KeyError:
            stot.append(None)
            stot.append(None)
        res.append(stat)
        ros.append(stot)
        if algo == 'cplex':
            names.append('CPLEX')
        elif algo == 'satenstein':
            names.append('SATenstein')
        elif 'clasp' in algo:
            if '_asp' in algo:
                algo = 'clasp ASP'
            else:
                if 'random' in algo:
                    algo = 'clasp RAND'
                else:
                    algo = 'clasp HAND'
            names.append(algo)
        else:
            names.append(algo)

    vs = DataFrame.from_records(ros, columns=[('fANOVA', 'vs. ablation', '$\mu$'),
                                              ('fANOVA', 'vs. ablation', '$\sigma$'),
                                              ('fANOVA', 'vs. LIN', '$\mu$'),
                                              ('fANOVA', 'vs. LIN', '$\sigma$'),
                                              ('fANOVA$\\_c$', 'vs. ablation', '$\mu$'),
                                              ('fANOVA$\\_c$', 'vs. ablation', '$\sigma$'),
                                              ('fANOVA$\\_c$', 'vs. LIN', '$\mu$'),
                                              ('fANOVA$\\_c$', 'vs. LIN', '$\sigma$'),
                                              ('ablation', 'vs. LIN', '$\mu$'),
                                              ('ablation', 'vs. LIN', '$\sigma$')
                                              ])
    df = DataFrame.from_records(res, columns=[
                                              ('Set vs. Set', 'ablation', '$\mu$'),
                                              ('Set vs. Set', 'ablation', '$\sigma$'),
                                              ('Set vs. Set', 'fANOVA', '$\mu$'),
                                              ('Set vs. Set', 'fANOVA', '$\sigma$'),
                                              ('Set vs. Set', 'fANOVA$\\_c$', '$\mu$'),
                                              ('Set vs. Set', 'fANOVA$\\_c$', '$\sigma$'),
                                              ('Set vs. Set', 'LIN', '$\mu$'),
                                              ('Set vs. Set', 'LIN', '$\sigma$')
                                              ])
    names = list(map(lambda x: x.replace('_', '\\_'), names))
    df *= 100
    vs *= 100
    df.index = names
    vs.index = names
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['', '', ''])
    vs.columns = pd.MultiIndex.from_tuples(vs.columns, names=['', '', ''])
    pd.options.display.float_format = '{:.2f}'.format
    if show:
        print()
        print('\\begin{table}[htbp]')
        print('\\centering')
        print(df.to_latex(escape=False, column_format='l|cc|cc|cc|cc|cc|cc', multicolumn_format='|c|'))
        print('\\caption{Comparison of fANOVA/ablation/LIN results across different instance sets.}')
        print('\\end{table}')
        print()
        print('\\begin{table}[htbp]')
        print('\\centering')
        print(vs.to_latex(escape=False, column_format='l|cc|cc|cc|cc|cc', multicolumn_format='|c|'))
        print('\\caption{Comparison of x and y results on the same instance sets.}')
        print('\\end{table}')
    return df, vs


def generate_all_possible_outputs(algo_names, diagonal=False):
    for algo in algo_names:
        jon_dirs(algo)
    df, vs = create_stats_df(algo_names, False, diagonal=diagonal)
    print('~*'*60)
    print('~*'*60)
    print('~*'*60)
    print()
    print()
    get_latex_outputs(algo_names, diagonal)
    print()
    print('\\section{%s}' % "Aggregated results")
    print('\\begin{table}[htbp]')
    print('\\centering')
    print(df.to_latex(escape=False, column_format='l|cc|cc|cc|cc|cc|cc', multicolumn_format='|c|'))
    print('\\caption{Comparison of fANOVA/ablation/LIN results across different instance sets.}')
    print('\\end{table}')
    print()
    print('\\begin{table}[htbp]')
    print('\\centering')
    print(vs.to_latex(escape=False, column_format='l|cc|cc|cc|cc|cc', multicolumn_format='|c|'))
    print('\\caption{Column 1 \\& 2: Comparison of x and y results on the same instance sets.}')
    print('\\end{table}')
    print('\\end{table}')
