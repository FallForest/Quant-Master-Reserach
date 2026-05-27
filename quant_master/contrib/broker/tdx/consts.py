# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

"""TQL Entry name registry and protocol constants.

Entry names confirmed from funcs_jy/*.sp files (银河证券 TDX addins).

Confirmed entry names (from GPMM.sp):
  - flatjy.ptbuy   — Stock buy  (买入)
  - flatjy.ptsell  — Stock sell (卖出)
  - FlatJy.gpzhzt  — Stock account status (股票账户状态)

NOTE: The HTTP interface (/TQLEX) returns -7201 "模块不存在" for all
trading entry names.  Trading operations are only available via the
binary protocol through xiadan.exe.  Use EasytraderBroker for live trading.

Login: ACL.checkuser (from web JS source — returns -7353 on HTTP)
"""

# Confirmed TQL Entry names (from funcs_jy/*.sp files)
# These are the real names compiled into the C++ DLLs.
TDX_ENTRIES = {
    # Authentication
    "login": "ACL.checkuser",
    # Stock trading (confirmed from GPMM.sp TCFeatureID)
    "buy": "flatjy.ptbuy",
    "sell": "flatjy.ptsell",
    "cancel_order": "flatjy.ptcancel",
    # Queries
    "query_orders": "flatjy.ptcxwt",
    "query_deals": "flatjy.ptcxcj",
    "query_positions": "flatjy.ptcxcc",
    "query_account": "FlatJy.gpzhzt",
    # HK Stock Connect
    "hk_buy": "flatjy.ggtbuy",
    "hk_sell": "flatjy.ggtsell",
    # Credit/Margin trading
    "credit_buy": "flatjy.xybuy",
    "credit_sell": "flatjy.xysell",
    "margin_buy": "flatjy.rzbuy",
    "margin_sell": "flatjy.rqsell",
}

# Legacy entry names (from earlier analysis — may not work on HTTP interface)
LEGACY_ENTRIES = {
    "login": "ACL.checkuser",
    "buy": "Stock.Buy",
    "sell": "Stock.Sell",
    "cancel_order": "Stock.ktqx",
    "query_orders": "Stock.QueryWt",
    "query_deals": "Stock.QueryCj",
    "query_positions": "Stock.QueryCc",
    "query_account": "Stock.QueryZj",
}

# Alternative entry name sets to try during discovery
CANDIDATE_PREFIXES = [
    "flatjy",
    "FlatJy",
    "Stock",
    "YWZQ",
    "YHZQ",
    "ACL",
    "TRADE",
    "JY",
    "GP",
]

CANDIDATE_OPERATIONS = {
    "login": ["checkuser", "login", "Login", "Auth", "Verify"],
    "buy": ["ptbuy", "Buy", "BuyStock", "BuyOrder", "wt_buy"],
    "sell": ["ptsell", "Sell", "SellStock", "SellOrder", "wt_sell"],
    "cancel_order": ["ptcancel", "ktqx", "Withdraw", "Cancel"],
    "query_orders": ["ptcxwt", "QueryWt", "QueryOrders"],
    "query_deals": ["ptcxcj", "QueryCj", "QueryDeals"],
    "query_positions": ["ptcxcc", "QueryCc", "QueryPositions"],
    "query_account": ["gpzhzt", "QueryZj", "QueryAccount"],
}

# TDX market codes
MARKET_SH = 1  # Shanghai (600/601/603/605/688)
MARKET_SZ = 0  # Shenzhen (000/001/002/003/300/301)

# Stock code prefix -> market code mapping
MARKET_MAP = {
    "600": MARKET_SH,
    "601": MARKET_SH,
    "603": MARKET_SH,
    "605": MARKET_SH,
    "688": MARKET_SH,
    "689": MARKET_SH,
    "000": MARKET_SZ,
    "001": MARKET_SZ,
    "002": MARKET_SZ,
    "003": MARKET_SZ,
    "300": MARKET_SZ,
    "301": MARKET_SZ,
}

# Market code -> prefix encoding for TQL parameter zqdm field
# DLL format: zqdm=%s%s (market_prefix + stock_code)
# e.g., "1sh600036" for Shanghai, "0sz000001" for Shenzhen
MARKET_PREFIX = {
    MARKET_SH: "1sh",
    MARKET_SZ: "0sz",
}

# Cookie names set by TDX server
COOKIE_TOKEN = "Token"
COOKIE_TDXID = "TDXID"
COOKIE_LOGID = "LOGID"
COOKIE_ATYPE = "ATYPE"
COOKIE_BTYPE = "BTYPE"
COOKIE_NICK = "NICK"

# TOUCH request parameters
TOUCH_PARAMS = {
    "Device": "Browser",
    "Ip": "0.0.0.0",
    "Mac": "00-00-00-00-00-00-00-00",
    "Build": "WEB",
    "Type": "41",
    "Ver": "1.0.0",
    "EP": "0",
}

# HTTP endpoint paths
PATH_TOUCH = "/TOUCH"
PATH_TQL = "/TQL"
PATH_TQLEX = "/TQLEX"
PATH_ALIVE = "/ALIVE"
PATH_QUIT = "/QUIT"

# Session timeout (milliseconds in JS, seconds here)
SESSION_TIMEOUT_SEC = 1790

# Known trading server addresses from connect.cfg
YINHE_TRADING_SERVERS = [
    ("61.135.173.138", 7708),
    ("211.100.23.198", 7708),
    ("61.151.252.170", 7708),
    ("58.49.110.76", 7708),
    ("59.42.252.136", 7708),
    ("218.75.75.18", 7708),
    ("218.108.13.247", 7708),
    ("221.136.93.19", 7708),
    ("221.12.53.53", 7708),
    ("218.75.85.139", 7708),
]

# Order status Chinese text -> OrderStatus mapping
ORDER_STATUS_MAP = {
    "已报": "pending",
    "部成": "partial_filled",
    "已成": "filled",
    "已撤": "cancelled",
    "废单": "failed",
    "待撤": "pending",
    "部撤": "cancelled",
}
