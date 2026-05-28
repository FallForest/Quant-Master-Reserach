# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: model-lab.spec.ts >> Model Lab Page >> has category filter tabs
- Location: e2e\model-lab.spec.ts:10:3

# Error details

```
Test timeout of 30000ms exceeded.
```

```
Error: page.goto: Test timeout of 30000ms exceeded.
Call log:
  - navigating to "http://localhost:5180/model-lab", waiting until "load"

```

# Page snapshot

```yaml
- generic [ref=e3]:
  - complementary [ref=e4]:
    - generic [ref=e5]:
      - img [ref=e6]
      - generic [ref=e8]: QuantMaster
    - navigation [ref=e9]:
      - generic [ref=e10]:
        - generic [ref=e11]: 概览
        - button "总览" [ref=e13] [cursor=pointer]:
          - img [ref=e14]
          - generic [ref=e16]: 总览
      - generic [ref=e17]:
        - generic [ref=e18]: 数据
        - generic [ref=e19]:
          - button "数据管道" [ref=e20] [cursor=pointer]:
            - img [ref=e21]
            - generic [ref=e23]: 数据管道
          - button "数据浏览" [ref=e24] [cursor=pointer]:
            - img [ref=e25]
            - generic [ref=e27]: 数据浏览
          - button "因子分析" [ref=e28] [cursor=pointer]:
            - img [ref=e29]
            - generic [ref=e31]: 因子分析
      - generic [ref=e32]:
        - generic [ref=e33]: 模型
        - generic [ref=e34]:
          - button "模型工坊" [ref=e35] [cursor=pointer]:
            - img [ref=e36]
            - generic [ref=e38]: 模型工坊
          - button "模型绩效" [ref=e39] [cursor=pointer]:
            - img [ref=e40]
            - generic [ref=e42]: 模型绩效
          - button "模型选股" [ref=e43] [cursor=pointer]:
            - img [ref=e44]
            - generic [ref=e46]: 模型选股
          - button "实验管理" [ref=e47] [cursor=pointer]:
            - img [ref=e48]
            - generic [ref=e50]: 实验管理
      - generic [ref=e51]:
        - generic [ref=e52]: 策略
        - generic [ref=e53]:
          - button "策略工坊" [ref=e54] [cursor=pointer]:
            - img [ref=e55]
            - generic [ref=e57]: 策略工坊
          - button "策略回测" [ref=e58] [cursor=pointer]:
            - img [ref=e59]
            - generic [ref=e61]: 策略回测
      - generic [ref=e62]:
        - generic [ref=e63]: 分析
        - generic [ref=e64]:
          - button "持仓分析" [ref=e65] [cursor=pointer]:
            - img [ref=e66]
            - generic [ref=e68]: 持仓分析
          - button "组合优化" [ref=e69] [cursor=pointer]:
            - img [ref=e70]
            - generic [ref=e72]: 组合优化
          - button "收益归因" [ref=e73] [cursor=pointer]:
            - img [ref=e74]
            - generic [ref=e76]: 收益归因
    - button "收起侧边栏" [ref=e77] [cursor=pointer]:
      - img [ref=e78]
      - generic [ref=e80]: 收起
  - generic [ref=e81]:
    - banner [ref=e82]:
      - heading "模型工坊" [level=1] [ref=e84]
      - generic [ref=e85]:
        - generic [ref=e86]: v2.1.0
        - button "切换到深色模式" [ref=e87] [cursor=pointer]:
          - img [ref=e88]
        - generic [ref=e90]: Q
    - generic [ref=e92]:
      - generic [ref=e93]:
        - generic [ref=e94]:
          - img [ref=e95]
          - heading "模型工坊" [level=2] [ref=e97]
          - generic [ref=e98]: 13 个模型
        - generic [ref=e99]:
          - button "全部" [ref=e100] [cursor=pointer]
          - button "树模型" [ref=e101] [cursor=pointer]
          - button "RNN" [ref=e102] [cursor=pointer]
          - button "注意力" [ref=e103] [cursor=pointer]
          - button "CNN" [ref=e104] [cursor=pointer]
          - button "DNN" [ref=e105] [cursor=pointer]
          - button "集成" [ref=e106] [cursor=pointer]
      - generic [ref=e107]:
        - generic [ref=e109] [cursor=pointer]:
          - generic [ref=e110]:
            - img [ref=e111]
            - generic [ref=e113]: LightGBM
            - generic [ref=e114]: 树模型
            - generic [ref=e115]:
              - generic [ref=e116]: 速度
              - generic [ref=e117]: 快
          - paragraph [ref=e118]: 微软开源的梯度提升框架，训练速度快，内存占用低，适合大规模表格数据
          - generic [ref=e119]:
            - generic [ref=e120]: 复杂度
            - generic [ref=e123]: 1/5
        - generic [ref=e125] [cursor=pointer]:
          - generic [ref=e126]:
            - img [ref=e127]
            - generic [ref=e129]: XGBoost
            - generic [ref=e130]: 树模型
            - generic [ref=e131]:
              - generic [ref=e132]: 速度
              - generic [ref=e133]: 较快
          - paragraph [ref=e134]: 经典梯度提升库，正则化能力强，支持自定义目标函数
          - generic [ref=e135]:
            - generic [ref=e136]: 复杂度
            - generic [ref=e139]: 1/5
        - generic [ref=e141] [cursor=pointer]:
          - generic [ref=e142]:
            - img [ref=e143]
            - generic [ref=e145]: CatBoost
            - generic [ref=e146]: 树模型
            - generic [ref=e147]:
              - generic [ref=e148]: 速度
              - generic [ref=e149]: 较快
          - paragraph [ref=e150]: Yandex开源的梯度提升库，自动处理类别特征，支持GPU加速
          - generic [ref=e151]:
            - generic [ref=e152]: 复杂度
            - generic [ref=e155]: 1/5
        - generic [ref=e157] [cursor=pointer]:
          - generic [ref=e158]:
            - img [ref=e159]
            - generic [ref=e161]: LSTM
            - generic [ref=e162]: RNN
            - generic [ref=e163]:
              - generic [ref=e164]: 速度
              - generic [ref=e165]: 较慢
          - paragraph [ref=e166]: 长短期记忆网络，捕捉时间序列中的长期依赖关系
          - generic [ref=e167]:
            - generic [ref=e168]: 复杂度
            - generic [ref=e171]: 3/5
        - generic [ref=e173] [cursor=pointer]:
          - generic [ref=e174]:
            - img [ref=e175]
            - generic [ref=e177]: GRU
            - generic [ref=e178]: RNN
            - generic [ref=e179]:
              - generic [ref=e180]: 速度
              - generic [ref=e181]: 中等
          - paragraph [ref=e182]: 门控循环单元，比LSTM参数更少，训练更快
          - generic [ref=e183]:
            - generic [ref=e184]: 复杂度
            - generic [ref=e187]: 3/5
        - generic [ref=e189] [cursor=pointer]:
          - generic [ref=e190]:
            - img [ref=e191]
            - generic [ref=e193]: ALSTM
            - generic [ref=e194]: RNN
            - generic [ref=e195]:
              - generic [ref=e196]: 速度
              - generic [ref=e197]: 较慢
          - paragraph [ref=e198]: 注意力增强的LSTM，通过注意力机制自动学习不同时间步的重要性
          - generic [ref=e199]:
            - generic [ref=e200]: 复杂度
            - generic [ref=e203]: 4/5
        - generic [ref=e205] [cursor=pointer]:
          - generic [ref=e206]:
            - img [ref=e207]
            - generic [ref=e209]: Transformer
            - generic [ref=e210]: 注意力
            - generic [ref=e211]:
              - generic [ref=e212]: 速度
              - generic [ref=e213]: 慢
          - paragraph [ref=e214]: 基于自注意力机制的模型，捕获股票间的交叉关系和时间依赖
          - generic [ref=e215]:
            - generic [ref=e216]: 复杂度
            - generic [ref=e219]: 5/5
        - generic [ref=e221] [cursor=pointer]:
          - generic [ref=e222]:
            - img [ref=e223]
            - generic [ref=e225]: GATs
            - generic [ref=e226]: 注意力
            - generic [ref=e227]:
              - generic [ref=e228]: 速度
              - generic [ref=e229]: 慢
          - paragraph [ref=e230]: 图注意力网络，将股票关系建模为图结构，学习股票间的动态关联
          - generic [ref=e231]:
            - generic [ref=e232]: 复杂度
            - generic [ref=e235]: 5/5
        - generic [ref=e237] [cursor=pointer]:
          - generic [ref=e238]:
            - img [ref=e239]
            - generic [ref=e241]: TCN
            - generic [ref=e242]: CNN
            - generic [ref=e243]:
              - generic [ref=e244]: 速度
              - generic [ref=e245]: 中等
          - paragraph [ref=e246]: 时序卷积网络，用因果膨胀卷积捕获多尺度时间模式
          - generic [ref=e247]:
            - generic [ref=e248]: 复杂度
            - generic [ref=e251]: 3/5
        - generic [ref=e253] [cursor=pointer]:
          - generic [ref=e254]:
            - img [ref=e255]
            - generic [ref=e257]: DNN
            - generic [ref=e258]: DNN
            - generic [ref=e259]:
              - generic [ref=e260]: 速度
              - generic [ref=e261]: 较快
          - paragraph [ref=e262]: 深度神经网络，多层全连接，适合横截面因子建模
          - generic [ref=e263]:
            - generic [ref=e264]: 复杂度
            - generic [ref=e267]: 2/5
        - generic [ref=e269] [cursor=pointer]:
          - generic [ref=e270]:
            - img [ref=e271]
            - generic [ref=e273]: Double Ensemble
            - generic [ref=e274]: 集成
            - generic [ref=e275]:
              - generic [ref=e276]: 速度
              - generic [ref=e277]: 较慢
          - paragraph [ref=e278]: 双层集成模型：底层多个同质基模型 + 上层Stacking融合，内置残差学习和换手控制
          - generic [ref=e279]:
            - generic [ref=e280]: 复杂度
            - generic [ref=e283]: 4/5
        - generic [ref=e285] [cursor=pointer]:
          - generic [ref=e286]:
            - img [ref=e287]
            - generic [ref=e289]: TFT
            - generic [ref=e290]: 注意力
            - generic [ref=e291]:
              - generic [ref=e292]: 速度
              - generic [ref=e293]: 慢
          - paragraph [ref=e294]: Temporal Fusion Transformer，支持静态/已知/未知三种输入，可解释性强
          - generic [ref=e295]:
            - generic [ref=e296]: 复杂度
            - generic [ref=e299]: 5/5
        - generic [ref=e301] [cursor=pointer]:
          - generic [ref=e302]:
            - img [ref=e303]
            - generic [ref=e305]: TabNet
            - generic [ref=e306]: CNN
            - generic [ref=e307]:
              - generic [ref=e308]: 速度
              - generic [ref=e309]: 较慢
          - paragraph [ref=e310]: 面向表格数据的注意力网络，自动特征选择，可解释性强
          - generic [ref=e311]:
            - generic [ref=e312]: 复杂度
            - generic [ref=e315]: 4/5
```

# Test source

```ts
  1  | import { test, expect } from '@playwright/test'
  2  | 
  3  | test.describe('Model Lab Page', () => {
  4  |   test('loads model catalog', async ({ page }) => {
  5  |     await page.goto('/model-lab')
  6  |     await page.waitForLoadState('domcontentloaded')
  7  |     await expect(page.locator('body')).toContainText('模型工坊')
  8  |   })
  9  | 
  10 |   test('has category filter tabs', async ({ page }) => {
> 11 |     await page.goto('/model-lab')
     |                ^ Error: page.goto: Test timeout of 30000ms exceeded.
  12 |     await page.waitForLoadState('domcontentloaded')
  13 |     // Should have category filter buttons
  14 |     await expect(page.locator('body')).toContainText('全部')
  15 |   })
  16 | })
  17 | 
```