import os
import pandas as pd
def produce_output(extraction_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    files = os.listdir(extraction_dir)
    output = []
    
    for file in files:
        df = pd.read_csv(f"{extraction_dir}/{file}")
        output.append(df) 
    
    output = pd.concat(output)
    output.to_csv(f"{output_dir}/output.csv", index=None)