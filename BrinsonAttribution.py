# -*- coding: utf-8 -*-
# @Time     : 2022/1/20 16:44
# @Author   ：Wang Guosong
# @File     : BrinsonAttribution.py
# @Software : PyCharm

import pandas as pd
import numpy as np
from wm.database import WindIndex


class BrinsonAttribution:

    @staticmethod
    def cal_sr(bm_sector_weight, pf_sector_ret, bm_sector_ret):
        """

        Parameters
        ----------
        bm_sector_weight
        pf_sector_ret
        bm_sector_ret

        Returns
        -------
        Examples
        -------
        >>> bm_sector_weight = pd.Series([0.1, 0.2, 0.3, 0.15, 0.05],
        index=['a', 'b', 'c', 'd', 'e'])
        >>> pf_sector_ret = pd.Series([0.11, 0.19, 0.32, 0.14, 0.09],
        index=['a', 'b', 'c', 'd', 'e'])
        >>> bm_sector_ret = pd.Series([0.1, 0.2, 0.3, 0.15, 0.05],
        index=['a', 'b', 'c', 'd', 'e'])
        >>> bm_sector_ret = pd.Series([0.1, 0.2, 0.3, 0.15, 0.05],
        index=['aa', 'b', 'c', 'd', 'e'])
        >>> BrinsonAttribution.cal_sr(bm_sector_weight, pf_sector_ret, bm_sector_ret)
        """
        if not((bm_sector_weight.index == pf_sector_ret.index).all() and (pf_sector_ret.index == bm_sector_ret.index).all()):
            raise ValueError("Benchmark and portfolio must have the same index with value of sectors.")
        sr = bm_sector_weight * (pf_sector_ret - bm_sector_ret)
        return sr

    @staticmethod
    def cal_sr_ir(pf_sector_weight, pf_sector_ret, bm_sector_ret):
        """

        Parameters
        ----------
        pf_sector_weight
        pf_sector_ret
        bm_sector_ret

        Returns
        -------
        Examples
        -------
        >>> bm_weight = pd.Series([0.1, 0.2, 0.3, 0.15, 0.05],
        index=['a', 'b', 'c', 'd', 'e'])
        >>> pf_sector_ret = pd.Series([0.11, 0.19, 0.32, 0.14, 0.09],
        index=['a', 'b', 'c', 'd', 'e'])
        >>> bm_sector_ret = pd.Series([0.1, 0.2, 0.3, 0.15, 0.05],
        index=['a', 'b', 'c', 'd', 'e'])
        >>> bm_sector_ret = pd.Series([0.1, 0.2, 0.3, 0.15, 0.05],
        index=['aa', 'b', 'c', 'd', 'e'])
        >>> BrinsonAttribution.cal_sr(bm_weight, pf_sector_ret, bm_sector_ret)
        """
        pf_sector_ret = pf_sector_ret.reindex(bm_sector_ret.index).fillna(0)
        if not ((pf_sector_weight.index == pf_sector_ret.index).all() and (pf_sector_ret.index == bm_sector_ret.index).all()):
            raise ValueError("Benchmark and portfolio must have the same index with value of sectors.")
        sr_ir = pf_sector_weight * (pf_sector_ret - bm_sector_ret)
        return sr_ir

    @staticmethod
    def cal_ar(pf_sector_weight, bm_sector_weight, bm_sector_ret, bm_index_ret=0):
        if not ((pf_sector_weight.index == bm_sector_ret.index).all and (bm_sector_weight.index == bm_sector_ret.index).all()):
            raise ValueError("Benchmark and portfolio must have the same index with value of sectors.")
        ar = (pf_sector_weight - bm_sector_weight) * (bm_sector_ret - bm_index_ret)
        return ar

    @staticmethod
    def cal_ir(pf_sector_weight, bm_sector_weight, pf_sector_ret, bm_sector_ret):
        if not ((pf_sector_weight.index == bm_sector_ret.index).all and (bm_sector_weight.index == pf_sector_ret.index).all() and \
                (pf_sector_ret.index == bm_sector_ret.index).all()):
            raise ValueError("Benchmark and portfolio must have the same index with value of sectors.")
        ir = (pf_sector_weight - bm_sector_weight) * (pf_sector_ret - bm_sector_ret)
        return ir

    @staticmethod
    def cal_pf_sector_ret_weight(stock_ret, stock_weight_in_pf, stock_sector):
        """

        Parameters
        ----------
        stock_ret
        stock_weight_in_pf
        stock_sector

        Returns
        -------

        Examples
        -------
        >>> import pandas as pd
        >>> weight_dir = "D:/E/Python/Brinson-Attribution/fund_holding/161810.OF持股.csv"
        >>> stock_weight_in_pf = pd.read_csv(weight_dir, encoding='utf-8')
        >>> stock_weight_in_pf  = stock_weight_in_pf[['报告期', '品种代码',  '占股票市值比(%)']]
        >>> stock_weight_in_pf  = stock_weight_in_pf[stock_weight_in_pf['报告期'] == '2017-12-31']
        >>> stock_weight_in_pf = stock_weight_in_pf.drop('报告期', axis=1).set_index(['品种代码']).drop('合计')
        >>> sector_dir = "D:/E/Python/Brinson-Attribution/quote_data/industry_zx.csv"
        >>> stock_sector = pd.read_csv(sector_dir, encoding='gbk')[['Unnamed: 0', '2017/12/29']]
        >>> stock_sector.set_index('Unnamed: 0', inplace=True)
        >>> stock_ret = pd.read_csv("D:/E/Python/Brinson-Attribution/quote_data/pct_chg_6M.csv")[['Unnamed: 0', '2017/12/31']]
        >>> stock_ret.set_index('Unnamed: 0', inplace=True)
        """
        stock_weight_in_pf = stock_weight_in_pf.dropna()
        stock_sector = stock_sector.dropna()
        stock_ret = stock_ret.dropna()
        multi_index = stock_weight_in_pf.index.intersection(stock_sector.index).intersection(stock_ret.index)
        stock_weight_in_pf = stock_weight_in_pf.reindex(multi_index)
        stock_sector = stock_sector.reindex(multi_index)
        stock_ret = stock_ret.reindex(multi_index)
        weight_sector = pd.concat([stock_weight_in_pf, stock_sector], axis=1)
        weight_sector.columns = ['weight', 'sector']
        pf_sector_weight = (weight_sector.groupby('sector').sum() / weight_sector['weight'].sum())['weight']
        pf_sector_return = weight_sector.groupby('sector').apply(
            lambda x: ((x.loc[:, 'weight'] * stock_ret.iloc[:, 0]).dropna() / x['weight'].sum()).sum())
        return pf_sector_return, pf_sector_weight

    @staticmethod
    def cal_bm_sector_ret_weight(stock_ret, stock_weight_in_bm, stock_sector):
        """

        Parameters
        ----------
        stock_ret
        stock_weight_in_bm
        stock_sector

        Returns
        -------

        Examples
        -------
        >>> import pandas as pd
        >>> weight_dir = "D:/E/Python/Brinson-Attribution/index_weight/000300.csv"
        >>> stock_weight_in_bm = pd.read_csv(weight_dir, encoding='utf-8')
        >>> stock_weight_in_bm  = stock_weight_in_bm[['Unnamed: 0', '2017-12-29']]
        >>> stock_weight_in_bm.set_index('Unnamed: 0', inplace=True)
        >>> sector_dir = "D:/E/Python/Brinson-Attribution/quote_data/industry_zx.csv"
        >>> stock_sector = pd.read_csv(sector_dir, encoding='gbk')[['Unnamed: 0', '2017/12/29']]
        >>> stock_sector.set_index('Unnamed: 0', inplace=True)
        >>> stock_ret = pd.read_csv("D:/E/Python/Brinson-Attribution/quote_data/pct_chg_6M.csv")[['Unnamed: 0', '2017/12/31']]
        >>> stock_ret.set_index('Unnamed: 0', inplace=True)
        """
        stock_weight_in_bm = stock_weight_in_bm.dropna()
        stock_sector = stock_sector.dropna()
        stock_ret = stock_ret.dropna()
        multi_index = stock_weight_in_bm.index.intersection(stock_sector.index).intersection(stock_ret.index)
        stock_weight_in_bm = stock_weight_in_bm.reindex(multi_index)
        stock_sector = stock_sector.reindex(multi_index)
        stock_ret = stock_ret.reindex(multi_index)
        weight_sector = pd.concat([stock_weight_in_bm, stock_sector], axis=1)
        weight_sector.columns = ['weight', 'sector']
        bm_sector_weight = (weight_sector.groupby('sector').sum() / weight_sector['weight'].sum())['weight']
        bm_sector_return = weight_sector.groupby('sector').apply(
            lambda x: ((x.loc[:, 'weight'] * stock_ret.iloc[:, 0]).dropna() / x['weight'].sum()).sum())
        return bm_sector_return, bm_sector_weight

    @staticmethod
    def cal_bm_whole_ret(bm_sector_ret, bm_sector_weight):
        """

        Parameters
        ----------
        bm_sector_ret
        bm_sector_weight

        Returns
        -------

        Examples
        -------
        >>> import pandas as pd
        >>> weight_dir = "D:/E/Python/Brinson-Attribution/index_weight/000300.csv"
        >>> stock_weight_in_bm = pd.read_csv(weight_dir, encoding='utf-8')
        >>> stock_weight_in_bm  = stock_weight_in_bm[['Unnamed: 0', '2017-12-29']]
        >>> stock_weight_in_bm.set_index('Unnamed: 0', inplace=True)
        >>> sector_dir = "D:/E/Python/Brinson-Attribution/quote_data/industry_zx.csv"
        >>> stock_sector = pd.read_csv(sector_dir, encoding='gbk')[['Unnamed: 0', '2017/12/29']]
        >>> stock_sector.set_index('Unnamed: 0', inplace=True)
        >>> stock_ret = pd.read_csv("D:/E/Python/Brinson-Attribution/quote_data/pct_chg_6M.csv")[['Unnamed: 0', '2017/12/31']]
        >>> stock_ret.set_index('Unnamed: 0', inplace=True)
        >>> bm_sector_ret, bm_sector_weight = BrinsonAttribution.cal_bm_sector_ret_weight(stock_ret, stock_weight_in_bm, stock_sector)
        """
        bm_whole_ret = (bm_sector_ret * bm_sector_weight).sum()
        return bm_whole_ret

    @staticmethod
    def _attribute_by_stock_position(stock_ret, stock_weight_in_pf,  stock_weight_in_bm, stock_sector, model='BF',
                                     ir_separated=True, out_df=False):
        """

        Parameters
        ----------
        stock_ret
        stock_weight_in_pf
        stock_weight_in_bm
        stock_sector
        model
        ir_separated

        Returns
        -------

        Examples
        --------
        >>> sr, ar = BrinsonAttribution._attribute_by_stock_position(stock_ret, stock_weight_in_pf, stock_weight_in_bm,
        stock_sector, ir_separated=False)

        """
        pf_sector_ret, pf_sector_weight = BrinsonAttribution.cal_pf_sector_ret_weight(stock_ret, stock_weight_in_pf,
                                                                                      stock_sector)
        bm_sector_ret, bm_sector_weight = BrinsonAttribution.cal_bm_sector_ret_weight(stock_ret, stock_weight_in_bm,
                                                                                      stock_sector)
        pf_sector_weight = pf_sector_weight.reindex(bm_sector_weight.index).fillna(0)
        pf_sector_ret = pf_sector_ret.reindex(bm_sector_ret.index).fillna(0)
        pf_ret = pf_sector_ret * pf_sector_weight
        bm_ret = bm_sector_ret * bm_sector_weight
        bm_whole_ret = BrinsonAttribution.cal_bm_whole_ret(bm_sector_ret, bm_sector_weight)
        excess_ret = pf_ret - bm_ret
        if ir_separated:
            sr = BrinsonAttribution.cal_sr(bm_sector_weight, pf_sector_ret, bm_sector_ret)
            ir = BrinsonAttribution.cal_ir(pf_sector_weight, bm_sector_weight, pf_sector_ret, bm_sector_ret)
            if model == "BF":
                ar = BrinsonAttribution.cal_ar(pf_sector_weight, bm_sector_weight, bm_sector_ret,
                                               bm_index_ret=bm_whole_ret)
            else:
                ar = BrinsonAttribution.cal_ar(pf_sector_weight, bm_sector_weight, bm_sector_ret)
            if not out_df:
                return sr, ir, ar, pf_ret, bm_ret
            else:
                attr_df = pd.DataFrame([pf_sector_weight, pf_sector_ret, bm_sector_weight, bm_sector_ret, pf_ret, bm_ret,
                                        sr, ir, ar, excess_ret]).T
                attr_df.columns = ['pf_sector_weight', 'pf_sector_ret', 'bm_sector_weight', 'bm_sector_ret', 'pfr',
                                   'bmr', 'sr', 'ir', 'ar', 'excess_ret']
                attr_df.loc['合计', :] = attr_df.sum()
                attr_df.loc['合计', ['pf_sector_ret', 'bm_sector_ret']] = np.nan
                return attr_df
        else:
            sr = BrinsonAttribution.cal_sr_ir(pf_sector_weight, pf_sector_ret, bm_sector_ret)
            if model == "BF":
                ar = BrinsonAttribution.cal_ar(pf_sector_weight, bm_sector_weight, bm_sector_ret,
                                               bm_index_ret=bm_whole_ret)
            else:
                ar = BrinsonAttribution.cal_ar(pf_sector_weight, bm_sector_weight, bm_sector_ret)
            if not out_df:
                return sr, ar, pf_ret, bm_ret
            else:
                attr_df = pd.DataFrame([pf_sector_weight, pf_sector_ret, bm_sector_weight, bm_sector_ret, pf_ret, bm_ret,
                                        sr, ar, excess_ret]).T
                attr_df.columns = ['pf_sector_weight', 'pf_sector_ret', 'bm_sector_weight', 'bm_sector_ret', 'pfr',
                                   'bmr', 'sr', 'ar', 'excess_ret']
                attr_df.loc['合计', :] = attr_df.sum()
                attr_df.loc['合计', ['pf_sector_ret', 'bm_sector_ret']] = np.nan
                return attr_df


    @staticmethod
    def attribute_by_stock_position(stock_ret, stock_weight_in_pf, stock_weight_in_bm, stock_sector, model='BF',
                                    ir_separated=True):
        """

        Parameters
        ----------
        stock_ret
        stock_weight_in_pf
        stock_weight_in_bm
        stock_sector
        model
        ir_separated

        Returns
        -------

        Examples
        --------
        >>> BrinsonAttribution.attribute_by_stock_position(stock_ret, stock_weight_in_pf, stock_weight_in_bm, stock_sector,
            ir_separated=False)

        """
        if ir_separated:
            sr, ir, ar, pfr, bmr = BrinsonAttribution._attribute_by_stock_position(stock_ret, stock_weight_in_pf,
                                                                                   stock_weight_in_bm, stock_sector,
                                                                                   model=model, ir_separated=ir_separated)

            return {'sr': sr.sum(), 'ir': ir.sum(), 'ar': ar.sum(), 'pfr': pfr.sum(), 'bmr': bmr.sum()}
        else:
            sr, ar, pfr, bmr = BrinsonAttribution._attribute_by_stock_position(stock_ret, stock_weight_in_pf,
                                                                               stock_weight_in_bm, stock_sector,
                                                                               model=model, ir_separated=ir_separated)
            return {'sr': sr.sum(), 'ar': ar.sum(), 'pfr': pfr.sum(), 'bmr': bmr.sum()}

    @staticmethod
    def gather_attribute_by_stock_position(stock_ret, stock_weight_in_pf, stock_weight_in_bm, stock_sector, model='BF',
                                           ir_separated=True):
        """

        Parameters
        ----------
        stock_ret
        stock_weight_in_pf
        stock_weight_in_bm
        stock_sector
        model
        ir_separated

        Returns
        -------

        Examples
        --------
        >>> BrinsonAttribution.attribute_by_stock_position(stock_ret, stock_weight_in_pf, stock_weight_in_bm, stock_sector,
            ir_separated=False)

        """
        return BrinsonAttribution._attribute_by_stock_position(stock_ret, stock_weight_in_pf, stock_weight_in_bm, stock_sector,
                                                                model, ir_separated, out_df=True)

    # @staticmethod
    # def attribute_in_asset_and_stock(stock_ret, stock_weight_in_pf, stock_weight_in_bm, stock_sector, model='BF',
    #                                  ir_separated=False, stock_bm_code='000300.SH', bond_bm_code='000012.SH', freq='6M'):
    #     attr_stock_position = BrinsonAttribution.attribute_by_stock_position(stock_ret, stock_weight_in_pf, stock_weight_in_bm,
    #                                                                    stock_sector, model=model, ir_separated=ir_separated)
    #
    #     bond_bm_ret = get_index_ret(bond_bm, freq)
    #     bond_bm_ret = bond_bm_ret.loc[brinson_stock.index]
    #
    #     stock_bm_ret = get_index_ret(stock_bm, freq)
    #     stock_bm_ret = stock_bm_ret.loc[brinson_stock.index]
    #
    #     #    bm_ret = brinson_stock['re_b'] * asset_weight['bm_stock'] + bond_bm_ret * asset_weight['bm_bond']
    #     bm_ret = stock_bm_ret * asset_weight['bm_stock'] + bond_bm_ret * asset_weight['bm_bond']
    #
    #     fund_ret = get_index_ret(fund_code, freq)
    #     fund_ret = fund_ret.loc[brinson_stock.index]
    #
    #     timing_ret = (asset_weight['pt_stock'] - asset_weight['bm_stock']) * (brinson_stock['re_b'] - bm_ret) + \
    #                  (asset_weight['pt_bond'] - asset_weight['bm_bond']) * (bond_bm_ret - bm_ret)
    #     ind_ret = brinson_stock['配置效应(AR)'] * asset_weight['pt_stock']
    #     select_ret = brinson_stock['选股效应(SR)'] * asset_weight['pt_stock']
    #
    #     res_con_timing = pd.concat([fund_ret, bm_ret, timing_ret, ind_ret, select_ret], axis=1)
    #     res_con_timing.columns = ['基金收益', '基准实际收益', '大类资产择时收益(TR)', '配置效应(AR)', '选股效应(SR)']
    #     res_con_timing['估计误差'] = res_con_timing['基金收益'] - res_con_timing[
    #         ['基准实际收益', '大类资产择时收益(TR)', '配置效应(AR)', '选股效应(SR)']].sum(axis=1)
    #     res_con_timing['是否调整'] = '调整后'
    #
    #     res_wo_timing = pd.concat([fund_ret, bm_ret, brinson_stock[['配置效应(AR)', '选股效应(SR)']]], axis=1)
    #     res_wo_timing.columns = ['基金收益', '基准实际收益', '配置效应(AR)', '选股效应(SR)']
    #     res_wo_timing['大类资产择时收益(TR)'] = np.nan
    #     res_wo_timing['估计误差'] = res_wo_timing['基金收益'] - res_wo_timing[['基准实际收益', '配置效应(AR)', '选股效应(SR)']].sum(axis=1)
    #     res_wo_timing['是否调整'] = '调整前'
    #
    #     res = pd.concat([res_con_timing, res_wo_timing], axis=0)
    #     res.index.name = 'date'
    #     res = res.reset_index()
    #     res = res.set_index(['date', '是否调整'])
    #     res = res.sort_index()
    #     res = res[['基金收益', '基准实际收益', '大类资产择时收益(TR)',
    #                '选股效应(SR)', '配置效应(AR)', '估计误差']]
    #     return res







