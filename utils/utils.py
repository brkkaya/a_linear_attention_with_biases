import pandas as pd


def prepare_databricks(data: pd.DataFrame):
    """data_sample consists of three keys, prompt, input, output, concat them"""
    text_col = []
    for instance in data:
        prompt = instance["instruction"]
        input = instance["context"]
        output = instance["response"]
        # If the input is empty, add an instruction
        if len(input.strip()) == 0:
            text = "### Instruction: \n" + prompt + "###Response: \n"
        # If the input is not empty, add an instruction and input
        else:
            text = "### Instruction: \n" + prompt + "\n### Input: \n" + input + "###Response:" 

        # Add the text to the text_col list
        text_col.append(text)

        # Add the text to the data
        data.loc[:, "text"] = text_col
    # Return the data
    return data
