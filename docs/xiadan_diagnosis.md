# xiadan GUI 自动化诊断报告

## 结论：64位 Python 无法通过 Win32 消息可靠控制 32 位 MFC xiadan

### 核心问题

`SetDlgItemTextW` 能正确设置控件显示文本（`GetDlgItemTextW`/`GetWindowTextW` 返回正确值），
但 `SendMessageW(WM_GETTEXT)` 从 64 位进程调用到 32 位 MFC Edit 控件时返回空（length=0）。

xiadan 内部的 MFC DDX（Dialog Data Exchange）通过 `GetWindowText` → `WM_GETTEXT` 读取控件值。
由于 `WM_GETTEXT` 跨进程失败，DDX 读到的是空值。

**结果**：显示正确，app 内部变量为空。点击买入时：
- "请输入代码"（内部 stock code 为空）
- "请输入正确的委托价格"（内部 price 为空）

### 测试过的所有方法

| 方法 | 设置显示 | 触发 DDX | 备注 |
|------|---------|---------|------|
| SetDlgItemTextW | ✅ | ❌ | 只改显示，不触发 MFC 内部同步 |
| WM_SETTEXT | ✅ | ❌ | 同上 |
| WM_CHAR via PostMessageW | ❌ | ❌ | 消息被 MFC 忽略 |
| keybd_event + SetFocus | ❌ | ❌ | 击键未到达 Edit 控件 |
| SendInput | ❌ | ❌ | 同上 |
| SendInput + scancode | ❌ | ❌ | 同上 |
| UIA ValuePattern.SetValue | ❌ | ❌ | COM 接口也无效 |
| UIA LegacyIAccessible.SetValue | ❌ | ❌ | 同上 |
| EM_SETSEL + EM_REPLACESEL | ❌ | ❌ | 同上 |
| EN_CHANGE/EN_UPDATE 通知 | ❌ | ❌ | 不触发 DDX |
| WM_SETFOCUS/WM_KILLFOCUS | ❌ | ❌ | 不触发 DDX |

### 为什么不行

1. **xiadan 是 32 位 MFC 应用**，Python 是 64 位
2. **32/64 边界问题**：`SendMessageW(WM_GETTEXT)` 的指针 thunk 可能有问题
3. **MFC 内部缓存**：CEdit 的内部文本缓存只通过键盘输入更新
4. **注入检测**：xiadan 可能检测并忽略合成输入（keybd_event/SendInput）

### 唯一可行方案

1. **32 位 Python + Tc.dll** — 直接调用银河交易 DLL，完全绕过 GUI
2. **miniQMT (xtquant)** — 券商官方量化接口
3. **混合方案** — 脚本填好大部分字段，用户手动输入股票代码触发 lookup

### 文件

- `scripts/tdx_direct.py` — Win32 GUI 自动化引擎（受限于上述问题）
- `scripts/diag_windows.py` — 窗口层级诊断
- `scripts/test_buy_zhidu.py` — 买入测试脚本
- `quant_master/contrib/broker/xiadan_broker.py` — Broker 适配器
