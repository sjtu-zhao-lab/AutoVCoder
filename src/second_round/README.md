# Second Round

## Build Dataset

![build_dataset](/pics/3.png)

Using the prompt shown in the image above, multiple code samples are generated from ChatGPT-3.5 (or possibly ChatGPT-4o now) and saved to data/second_round/origin.jsonl. To generate this dataset, navigate to the build_dataset directory and run:
```
python code_gen_by_gpt.py
```
This will produce a large amount of GPT-synthesized data, which is saved in data/second_round/origin.jsonl.

## Code Filter

![build_dataset](/pics/4.png)

The data format is meticulously processed. To generate test cases for combinational logic circuits, navigate to the code_filter directory and run:
```
python dump_python_testcases.py
```
This will save the test cases in the data/second_round/dataset/testcases/ folder. Then, execute:
```
python eq_verification.py
```
to obtain the second round of the dataset.

## Train Model

Refer to first_round/README.md for training instructions.


