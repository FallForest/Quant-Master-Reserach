"""Inlined prompt data previously loaded from RD-Agent YAML files."""

PROPOSAL_PROMPTS: dict = {
    "hypothesis_gen": {
        "system_prompt": (
            "The user is working on generating new hypotheses for the {{ targets }} in a data-driven research and development process.\n"
            "The {{ targets }} are used in the following scenario:\n"
            "{{ scenario }}\n"
            "\n"
            "{% if user_instruction %}\n"
            "**User's overall instruction:**\n"
            "{{ user_instruction }}\n"
            "{% endif %}\n"
            "\n"
            "The user has already proposed several hypotheses and conducted evaluations on them. This information will be provided to you. Your task is to analyze previous experiments, reflect on the decision made in each experiment, and consider why experiments with a decision of true were successful while those with a decision of false failed. Then, think about how to improve further — either by refining the existing approach or by exploring an entirely new direction.\n"
            "If one exists and you agree with it, feel free to use it. If you disagree, please generate an improved version.\n"
            "{% if hypothesis_specification %}\n"
            "To assist you in formulating new hypotheses, the user has provided some additional information:\n"
            "{{ hypothesis_specification }}\n"
            "**Important:** If the hypothesis_specification outlines the next steps you need to follow, ensure you adhere to those instructions.\n"
            "{% endif %}\n"
            "Please generate the output using the following format and specifications:\n"
            "{{ hypothesis_output_format }}"
        ),
        "user_prompt": (
            "{% if hypothesis_and_feedback|length == 0 %}\n"
            "It is the first round of hypothesis generation. The user has no hypothesis on this scenario yet.\n"
            "{% else %}\n"
            "The former hypothesis and the corresponding feedbacks are as follows:\n"
            "{{ hypothesis_and_feedback }}\n"
            "{% endif %}\n"
            "{% if last_hypothesis_and_feedback %}\n"
            "Here is the last trial's hypothesis and the corresponding feedback (The main feedback contains a new hypothesis for your reference only. You need to evaluate the complete trace chain to decide whether to adopt it or propose a more appropriate hypothesis):\n"
            "{{ last_hypothesis_and_feedback }}\n"
            "{% endif %}\n"
            "{% if sota_hypothesis_and_feedback != \"\" %}\n"
            "Here is the SOTA trail's hypothesis and the corresponding feedback:\n"
            "{{ sota_hypothesis_and_feedback }}\n"
            "{% endif %}\n"
            "{% if RAG %}\n"
            "To assist you in generating new {{ targets }}, we have provided the following information: {{ RAG }}.\n"
            "{% endif %}"
        ),
    },
    "hypothesis2experiment": {
        "system_prompt": (
            "The user is trying to generate new {{ targets }} based on the hypothesis generated in the previous step.\n"
            "The {{ targets }} are used in certain scenario, the scenario is as follows:\n"
            "{{ scenario }}\n"
            "The user will use the {{ targets }} generated to do some experiments. The user will provide this information to you:\n"
            "1. The target hypothesis you are targeting to generate {{ targets }} for.\n"
            "2. The hypothesis generated in the previous steps and their corresponding feedbacks.\n"
            "3. Former proposed {{ targets }} on similar hypothesis.\n"
            "4. Some additional information to help you generate new {{ targets }}.\n"
            "Please generate the output following the format below:\n"
            "{{ experiment_output_format }}"
        ),
        "user_prompt": (
            "The user has made several hypothesis on this scenario and did several evaluation on them.\n"
            "The target hypothesis you are targeting to generate {{ targets }} for is as follows:\n"
            "{{ target_hypothesis }}\n"
            "{% if hypothesis_and_feedback %}\n"
            "The former hypothesis and the corresponding feedbacks are as follows:\n"
            "{{ hypothesis_and_feedback }}\n"
            "{% endif %}\n"
            "{% if last_hypothesis_and_feedback %}\n"
            "The latest hypothesis and the corresponding feedback are as follows:\n"
            "{{ last_hypothesis_and_feedback }}\n"
            "{% endif %}\n"
            "{% if sota_hypothesis_and_feedback %}\n"
            "The SOTA hypothesis and the corresponding feedback are as follows:\n"
            "{{ sota_hypothesis_and_feedback }}\n"
            "{% endif %}\n"
            "\n"
            "Please generate the new {{ targets }} based on the information above."
        ),
    },
}

QUANT_MASTER_PROMPTS: dict = {
    "factor_hypothesis_output_format": (
        "The output should follow JSON format. The schema is as follows:\n"
        "{\n"
        '"hypothesis": "The new hypothesis generated based on the information provided. Limit in two or three sentences.",\n'
        '"reason": "The reason why you generate this hypothesis. It should be comprehensive and logical. It should cover the other keys below and extend them. Limit in two or three sentences.",\n'
        "}"
    ),
    "factor_hypothesis_specification": (
        "1. **1-5 Factors per Generation:**\n"
        "  - Ensure each generation produces 1-5 factors.\n"
        "  - Balance simplicity and complexity to build a robust factor library.\n"
        "  - Make full use of the financial data provided to you instead of focusing solely on a specific field.\n"
        "2. **Simple and Effective Factors First:**\n"
        "  - Start with factors that are simple, easy to achieve and likely effective.\n"
        "  - Concisely explain why these factors are expected to work.\n"
        "  - Avoid complex or combined factors initially.\n"
        "3. **Gradual Complexity Increase:**\n"
        "  - Introduce more complex factors (e.g. machine learning based factors, factors use mult-dimentional factor raw data, etc.) as more experimental results are gathered.\n"
        "  - Combine factors only after simpler ones are tested and validated.\n"
        "4. **New Directions and Optimizations:**\n"
        "  - If multiple consecutive iterations fail to produce factors surpassing SOTA, consider switching to a new direction and can starting with simple factors again.\n"
        "  - If optimizing a specific type of factor, proceed from simple to complex.\n"
        "5. Note\n"
        "  - Highlight that factors surpassing SOTA are included in the library to avoid re-implementation.\n"
        "  - No matter how many factors you plan to generate, only reply with one set of hypothesis and reason. The hypothesis can include the proposal of multiple factors at the same time."
    ),
    "factor_experiment_output_format": (
        "The output should follow JSON format. The schema is as follows:\n"
        "{\n"
        '    "factor name 1": {\n'
        '        "description": "description of factor 1, start with its type, e.g. [Momentum Factor]",\n'
        '        "formulation": "latex formulation of factor 1",\n'
        '        "variables": {\n'
        '            "variable or function name 1": "description of variable or function 1",\n'
        '            "variable or function name 2": "description of variable or function 2"\n'
        "        }\n"
        "    },\n"
        '    "factor name 2": {\n'
        '        "description": "description of factor 2, start with its type, e.g. [Machine Learning based Factor]",\n'
        '        "formulation": "latex formulation of factor 2",\n'
        '        "variables": {\n'
        '            "variable or function name 1": "description of variable or function 1",\n'
        '            "variable or function name 2": "description of variable or function 2"\n'
        "        }\n"
        "    }\n"
        "    # Don't add ellipsis (...) or any filler text that might cause JSON parsing errors here!\n"
        "}"
    ),
    "hypothesis_output_format": (
        "The output should follow JSON format. The schema is as follows:\n"
        "{\n"
        '"hypothesis": "An exact, testable, and innovative statement derived from previous experimental trace analysis. Avoid overly general ideas and ensure precision. The hypothesis should clearly specify the exact approach and expected improvement in performance in two or three sentences.",\n'
        '"reason": "Provide a clear, logical explanation for why this hypothesis was proposed, grounded in evidence (e.g., trace history, domain principles). Reason should be short with no more than two sentences.",\n'
        "}"
    ),
    "model_hypothesis_specification": (
        "1. First, observe and analyze the overall experimental progression in `hypothesis_and_feedback`. Analyze where the previous model designs were inadequate — whether it was due to parameter settings, architectural flaws, or a lack of novelty (proposing entirely new concepts is highly encouraged as long as they demonstrate effectiveness).\n"
        "2. Second, `last_hypothesis_and_feedback` and `sota_hypothesis_and_feedback` are key references you should pay close attention to. You can choose to optimize based on either of them or generate new ideas to form hypotheses and experiments.\n"
        "3. If there is no prior experiment or result available at the beginning, you can start by implementing a simple and small architecture.\n"
        "4. If a series of attempts fail to achieve SOTA, consider exploring entirely new directions; at this point, it is acceptable to return to simple architectures.\n"
        "5. Focus exclusively on the architecture of PyTorch models. Each hypothesis should specifically address architectural decisions, such as layer configurations, activation functions, regularization methods, and overall model structure. DO NOT do any feature-specific processing. Instead, you can propose innovative transformations on the input time-series data to enhance model training effectiveness.\n"
        "6. Avoid including aspects unrelated to architecture, such as input features or optimization strategies.\n"
        "7. Sometimes, when training performance is poor, adjusting hyperparameters can also be an effective strategy for improvement.\n"
        "8. Use standard libraries for baseline models, but also explore custom architecture designs to investigate novel structures. After sufficient trials with traditional models, aim for innovation comparable to top-tier AI conferences (NeurIPS, ICLR, ICML, SIGKDD, etc.) in time series modeling."
    ),
    "model_experiment_output_format": (
        "So far please only design one model to test the hypothesis!\n"
        "The output should follow JSON format. The schema is as follows (value in training_hyperparameters is a basic setting for reference, you CAN CHANGE depends on the previous training log):\n"
        "{\n"
        '  "model_name (The name of the model)": {\n'
        '      "description": "A detailed description of the model",\n'
        '      "formulation": "A LaTeX formula representing the model\'s formulation",\n'
        '      "architecture": "A detailed description of the model\'s architecture, e.g., neural network layers or tree structures",\n'
        '      "variables": {\n'
        '          "\\\\hat{y}_u": "The predicted output for node u",\n'
        '          "variable_name_2": "Description of variable 2",\n'
        '          "variable_name_3": "Description of variable 3"\n'
        "      },\n"
        '      "hyperparameters": {\n'
        '          "hyperparameter_name_1": "value of hyperparameter 1",\n'
        '          "hyperparameter_name_2": "value of hyperparameter 2",\n'
        '          "hyperparameter_name_3": "value of hyperparameter 3"\n'
        "      },\n"
        '      "training_hyperparameters": {\n'
        '          "n_epochs": "100",\n'
        '          "lr": "1e-3",\n'
        '          "early_stop": 10,\n'
        '          "batch_size": 256,\n'
        '          "weight_decay": 1e-4\n'
        "      },\n"
        '      "model_type": "Tabular or TimeSeries"\n'
        "  }\n"
        "}"
    ),
    "action_gen": {
        "system": (
            "Quantitative investment is a data-driven approach to asset management that relies on mathematical models, statistical techniques, and computational methods to analyze financial markets and make investment decisions. Two essential components of this approach are factors and models.\n"
            "\n"
            "You are one of the most authoritative quantitative researchers at a top Wall Street hedge fund. I need your expertise to develop new factors and models that can enhance our investment returns. Based on the given context, I will ask for your assistance in designing and implementing either factors or a model.\n"
            "\n"
            "You will receive a series of experiments, including their factors and models, and their results.\n"
            "Your task is to analyze the previous experiments and decide whether the next experiment should focus on factors or models.\n"
            "\n"
            "Example JSON Structure for your return:\n"
            "{\n"
            '  "action": "factor" or "model",  # You must choose one of the two\n'
            "}"
        ),
        "user": (
            "{% if hypothesis_and_feedback|length == 0 %}\n"
            "It is the first round of hypothesis generation. The user has no hypothesis on this scenario yet.\n"
            "{% else %}\n"
            "The former hypothesis and the corresponding feedbacks are as follows:\n"
            "{{ hypothesis_and_feedback }}\n"
            "{% endif %}\n"
            "\n"
            "\n"
            "{% if last_hypothesis_and_feedback != \"\" %}\n"
            "Here is the last trial's hypothesis and the corresponding feedback. The main feedback includes a new hypothesis for your reference only. You should evaluate the entire reasoning chain to decide whether to adopt it, propose a more suitable hypothesis, or transfer and optimize it for another scenario (e.g., factor/model), since transfers are generally encouraged:\n"
            "{{ last_hypothesis_and_feedback }}\n"
            "{% endif %}"
        ),
    },
}

QUANT_MASTER_EXPERIMENT_PROMPTS: dict = {
    "quant_master_factor_background": (
        "The factor is a characteristic or variable used in quant investment that can help explain the returns and risks of a portfolio or a single asset. Factors are used by investors to identify and exploit sources of excess returns, and they are central to many quantitative investment strategies.\n"
        "Each number in the factor represents a physics value to an instrument on a day.\n"
        "User will train a model to predict the next several days return based on the factor values of the previous days.\n"
        "The factor is defined in the following parts:\n"
        "1. Name: The name of the factor.\n"
        "2. Description: The description of the factor.\n"
        "3. Formulation: The formulation of the factor.\n"
        "4. Variables: The variables or functions used in the formulation of the factor.\n"
        "The factor might not provide all the parts of the information above since some might not be applicable.\n"
        "Please specifically give all the hyperparameter in the factors like the window size, look back period, and so on. One factor should statically defines one output with a static source data. For example, last 10 days momentum and last 20 days momentum should be two different factors.\n"
        "\n"
        "{% if runtime_environment is not none %}\n"
        "====== Runtime Environment ======\n"
        "You have following environment to run the code:\n"
        "{{ runtime_environment }}\n"
        "{% endif %}"
    ),
    "quant_master_factor_interface": (
        "Your python code should follow the interface to better interact with the user's system.\n"
        'Your python code should contain the following part: the import part, the function part, and the main part. You should write a main function name: "calculate_{function_name}" and call this function in "if __name__ == __main__" part. Don\'t write any try-except block in your python code. The user will catch the exception message and provide the feedback to you.\n'
        'User will write your python code into a python file and execute the file directly with "python {your_file_name}.py". You should calculate the factor values and save the result into a HDF5(H5) file named "result.h5" in the same directory as your python file. The result file is a HDF5(H5) file containing a pandas dataframe. The index of the dataframe is the "datetime" and "instrument", and the single column name is the factor name,and the value is the factor value. The result file should be saved in the same directory as your python file.'
    ),
    "quant_master_factor_output_format": (
        "Your output should be a pandas dataframe similar to the following example information:\n"
        "<class 'pandas.core.frame.DataFrame'>\n"
        "MultiIndex: 40914 entries, (Timestamp('2020-01-02 00:00:00'), 'SH600000') to (Timestamp('2021-12-31 00:00:00'), 'SZ300059')\n"
        "Data columns (total 1 columns):\n"
        " #   Column            Non-Null Count  Dtype\n"
        "---  ------            --------------  -----\n"
        " 0   your factor name  40914 non-null  float64\n"
        "dtypes: float64(1)\n"
        "memory usage: <ignore>\n"
        "Notice: The non-null count is OK to be different to the total number of entries since some instruments may not have the factor value on some days.\n"
        "One possible format of `result.h5` may be like following:\n"
        "datetime    instrument\n"
        "2020-01-02  SZ000001     -0.001796\n"
        "            SZ000166      0.005780\n"
        "            SZ000686      0.004228\n"
        "            SZ000712      0.001298\n"
        "            SZ000728      0.005330\n"
        "                            ...\n"
        "2021-12-31  SZ000750      0.000000\n"
        "            SZ000776      0.002459"
    ),
    "quant_master_factor_simulator": (
        "The factors will be sent into Quant-Master to train a model to predict the next several days return based on the factor values of the previous days.\n"
        "Quant-Master is an AI-oriented quantitative investment platform that aims to realize the potential, empower research, and create value using AI technologies in quantitative investment, from exploring ideas to implementing productions. Quant-Master supports diverse machine learning modeling paradigms. including supervised learning, market dynamics modeling, and RL.\n"
        "User will use Quant-Master to automatically do the following things:\n"
        "1. generate a new factor table based on the factor values.\n"
        "2. train a model like LightGBM, CatBoost, LSTM or simple PyTorch model to predict the next several days return based on the factor values.\n"
        "3. build a portfolio based on the predicted return based on a strategy.\n"
        "4. evaluate the portfolio's performance including the return, sharpe ratio, max drawdown, and so on."
    ),
    "quant_master_model_background": (
        "The model is a machine learning or deep learning structure used in quantitative investment to predict the returns and risks of a portfolio or a single asset. Models are employed by investors to generate forecasts based on historical data and identified factors, which are central to many quantitative investment strategies.\n"
        "Each model takes the factors as input and predicts the future returns. Usually, the bigger the model is, the better the performance would be.\n"
        "The model is defined in the following parts:\n"
        "1. Name: The name of the model.\n"
        "2. Description: The description of the model.\n"
        "3. Architecture: The detailed architecture of the model, such as neural network layers or tree structures.\n"
        "4. Hyperparameters: The hyperparameters used in the model.\n"
        "5. Training_hyperparameters: The hyperparameters used during the training process.\n"
        '6. ModelType: The type of the model, "Tabular" for tabular model and "TimeSeries" for time series model.\n'
        "The model should provide clear and detailed documentation of its architecture and hyperparameters. One model should statically define one output with a fixed architecture and hyperparameters.\n"
        "\n"
        "{% if runtime_environment is not none %}\n"
        "====== Runtime Environment ======\n"
        "You have following environment to run the code:\n"
        "{{ runtime_environment }}\n"
        "{% endif %}"
    ),
    "quant_master_model_interface": (
        "Your python code should follow the interface to better interact with the user's system.\n"
        "You code should contain several parts:\n"
        "1. The import part: import the necessary libraries.\n"
        "2. A class which is a sub-class of pytorch.nn.Module. This class should should have a init function and a forward function which inputs a tensor and outputs a tensor.\n"
        '3. Set a variable called "model_cls" to the class you defined.\n'
        '\n'
        'The user will save your code into a python file called "model.py". Then the user imports model_cls in file "model.py" after setting the cwd into the directory:\n'
        "```python\n"
        "from model import model_cls\n"
        "```\n"
        "So your python code should follow the pattern:\n"
        "```python\n"
        "class XXXModel(torch.nn.Module):\n"
        "    ...\n"
        "model_cls = XXXModel\n"
        "```\n"
        '\n'
        'The model can be configured as either "Tabular" for tabular models or "TimeSeries" for time series models. For a tabular model, the input shape is (batch_size, num_features), while for a time series model, the input shape is (batch_size, num_timesteps, num_features). In both cases, the output shape of the model should be (batch_size, 1).\n'
        "`num_features` will be directly set for the model based on the input data shape.\n"
        "User will initialize the tabular model with the following code:\n"
        "```python\n"
        "model = model_cls(num_features=num_features)\n"
        "```\n"
        "User will initialize the time series model with the following code:\n"
        "```python\n"
        "model = model_cls(num_features=num_features, num_timesteps=num_timesteps)\n"
        "```\n"
        "No other parameters will be passed to the model so give other parameters a default value or just make them static.\n"
        "\n"
        "Don't write any try-except block in your python code. The user will catch the exception message and provide the feedback to you. Also, don't write main function in your python code. The user will call the forward method in the model_cls to get the output tensor.\n"
        "\n"
        "Please notice that your model should only use current features as input. The user will provide the input tensor to the model's forward function."
    ),
    "quant_master_model_output_format": (
        "Your output should be a tensor with shape (batch_size, 1).\n"
        'The output tensor should be saved in a file named "output.pth" in the same directory as your python file.\n'
        'The user will evaluate the shape of the output tensor so the tensor read from "output.pth" should be 8 numbers.'
    ),
    "quant_master_model_simulator": (
        "The models will be sent into Quant-Master to train and evaluate their performance in predicting future returns. Hypothesis is improved upon checking the feedback on the results.\n"
        "Quant-Master is an AI-oriented quantitative investment platform that aims to realize the potential, empower research, and create value using AI technologies in quantitative investment, from exploring ideas to implementing productions. Quant-Master supports diverse machine learning modeling paradigms, including supervised learning, market dynamics modeling, and reinforcement learning (RL).\n"
        "User will use Quant-Master to automatically perform the following tasks:\n"
        "1. Generate a baseline factor table.\n"
        "2. Train the model defined in your class Net to predict the next several days' returns based on the factor values.\n"
        "3. Build a portfolio based on the predicted returns using a specific strategy.\n"
        "4. Evaluate the portfolio's performance, including metrics such as return, IC, max drawdown, and others.\n"
        "5. Iterate on growing the hypothesis to enable model improvements based on performance evaluations and feedback."
    ),
}

MODEL_ONE_SHOT_PROMPTS: dict = {
    "code_implement_sys": "You are an assistant whose job is to answer user's question.",
    "code_implement_user": (
        "With the following given information, write a python code using pytorch and torch_geometric to implement the model.\n"
        "This model is in the graph learning field, only have one layer.\n"
        "The input will be node_feature [num_nodes, dim_feature] and edge_index [2, num_edges]  (It would be the input of the forward model)\n"
        "There is not edge attribute or edge weight as input. The model should detect the node_feature and edge_index shape, if there is Linear transformation layer in the model, the input and output shape should be consistent. The in_channels is the dimension of the node features.\n"
        "Implement the model forward function based on the following information:model formula information.\n"
        "1. model name:{{name}}\n"
        "2. model description:{{description}}\n"
        "3. model formulation:{{formulation}}\n"
        "4. model variables:{{variables}}.\n"
        "You must complete the forward function as far as you can do.\n"
        "Execution Your implemented code will be executed in the follow way:\n"
        "The the implemented code will be placed in a file like [uuid]/model.py\n"
        "We'll import the model in the implementation in file `model.py` after setting the cwd into the directory\n"
        "- from model import model_cls (So you must have a variable named `model_cls` in the file)\n"
        "  - So your implemented code could follow the following pattern\n"
        "    ```Python\n"
        "    class XXXLayer(torch.nn.Module):\n"
        "        ...\n"
        "    model_cls = XXXLayer\n"
        "    ```\n"
        "- initialize the model by initializing it `model_cls(input_dim=INPUT_DIM)`\n"
        "- And then verify the model by comparing the output tensors by feeding specific input tensor."
    ),
}
