# -*- coding: utf-8 -*-
# @Time     : 2022/1/20 16:44
# @Author   ：Wang Guosong
# @File     : BrinsonAttribution.py
# @Software : PyCharm

import pandas as pd

class BrinsonAttribution:

    @staticmethod
    def cal_sr(bm_weight, pf_sector_ret, bm_sector_ret):
        """

        Parameters
        ----------
        bm_weight
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
        if not((bm_weight.index == pf_sector_ret.index).all() and (pf_sector_ret.index == bm_sector_ret.index).all()):
            raise ValueError("Benchmark and portfolio must have the same index with value of sectors.")
        sr = bm_weight * (pf_sector_ret - bm_sector_ret)
        return sr

    @staticmethod
    def cal_sr_ir(pf_weight, pf_sector_ret, bm_sector_ret):
        """

        Parameters
        ----------
        pf_weight
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
        if not ((pf_weight.index == pf_sector_ret.index).all() and (pf_sector_ret.index == bm_sector_ret.index).all()):
            raise ValueError("Benchmark and portfolio must have the same index with value of sectors.")
        sr_ir = pf_weight * (pf_sector_ret - bm_sector_ret)
        return sr_ir

    @staticmethod
    def cal_ar(pf_weight, bm_weight, bm_sector_ret, bm_index_ret=0):
        if not ((pf_weight.index == bm_sector_ret.index).all and (bm_weight.index == bm_sector_ret.index).all()):
            raise ValueError("Benchmark and portfolio must have the same index with value of sectors.")
        ar = (pf_weight - bm_weight) * (bm_sector_ret - bm_index_ret)
        return ar

    @staticmethod
    def cal_ir(pf_weight, bm_weight, pf_sector_ret, bm_sector_ret):
        if not ((pf_weight.index == bm_sector_ret.index).all and (bm_weight.index == pf_sector_ret.index).all() and\
                (pf_sector_ret.index == bm_sector_ret.index).all()):
            raise ValueError("Benchmark and portfolio must have the same index with value of sectors.")
        ir = (pf_weight - bm_weight) * (pf_sector_ret - bm_sector_ret)
        return ir

    @staticmethod
    def cal_sector_ret(stock_ret, stock_weight_in_bm, stock_sector):
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
        pd.concat =         bm_sector_return = (stock_weight_in_bm.iloc[:, 0] * stock_ret.iloc[:, 0]) / stock_weight_in_bm.iloc[:, 0].sum()
        pass


