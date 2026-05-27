# pre-2024 multi-year stable label gate

- status: completed
- verdict: NO_GO
- candidates_evaluated: 9
- gate_pass_count: 0
- data_end: 2023-12-31

## Top candidates
- ridge_mstate_mstate_bull20_bear_dd10_vol_dd10_trainfit_rank_wflat_a10 | family=market_state | combined_ir=1.1456 | turnover=0.1465 | gate_pass=False | reasons=worst_year_ir_not_positive_years=2020; combined_ir=1.14555<1.8
- ridge_mstate_mstate_bull20_bear_dd10_vol_dd10_trainfit_rank_wtrend_a10 | family=market_state | combined_ir=1.0654 | turnover=0.1470 | gate_pass=False | reasons=portfolio_ir_gt_1_years=2<3; worst_year_ir_not_positive_years=2020; combined_ir=1.0654<1.8
- ridge_mstate_mstate_bull10_bear_dd10_vol_def_trainfit_rank_wtrend_a10 | family=market_state | combined_ir=1.0624 | turnover=0.1479 | gate_pass=False | reasons=portfolio_ir_gt_1_years=2<3; worst_year_ir_not_positive_years=2020; combined_ir=1.06245<1.8
- ridge_liqsurv_liqsurv_20d_w1_trainfit_rank_a10 | family=liquidity | combined_ir=0.9882 | turnover=0.1457 | gate_pass=False | reasons=worst_year_ir_not_positive_years=2020; combined_ir=0.988165<1.8
- ridge_liqsurv_liqsurv_5d_w1_trainfit_rank_a10 | family=liquidity | combined_ir=0.9838 | turnover=0.1470 | gate_pass=False | reasons=portfolio_ir_gt_1_years=2<3; worst_year_ir_not_positive_years=2020; combined_ir=0.983845<1.8
- ridge_liqsurv_liqsurv_10d_w1_trainfit_rank_a10 | family=liquidity | combined_ir=0.9023 | turnover=0.1465 | gate_pass=False | reasons=portfolio_ir_gt_1_years=2<3; worst_year_ir_not_positive_years=2020; combined_ir=0.902273<1.8
- ridge_ddcond_ddcond_10d_lam1_trainfit_rank_a10 | family=drawdown | combined_ir=0.5078 | turnover=0.1477 | gate_pass=False | reasons=portfolio_ir_gt_1_years=2<3; worst_year_ir_not_positive_years=2020; combined_ir=0.5078<1.8
- ridge_ddcond_ddcond_5d_lam1_trainfit_rank_a10 | family=drawdown | combined_ir=0.4157 | turnover=0.1488 | gate_pass=False | reasons=portfolio_ir_gt_1_years=2<3; worst_year_ir_not_positive_years=2020; combined_ir=0.41573<1.8
- ridge_ddcond_ddcond_20d_lam1_trainfit_rank_a10 | family=drawdown | combined_ir=0.3660 | turnover=0.1458 | gate_pass=False | reasons=portfolio_ir_gt_1_years=2<3; worst_year_ir_not_positive_years=2020; combined_ir=0.366001<1.8
