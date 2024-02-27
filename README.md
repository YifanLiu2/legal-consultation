# legal-consultation

## Data Preprocessing

### Usage

Execute `data_preprocess.py` from the terminal with the following command:

```sh
python data_preprocess.py -i <input_path> -o <output_path> [-n <nrows>]
```

#### Parameters
- `-i <input_path>`: Path to input data directory.
- `-o <output_path>`: Path for saving processed files.
- `-n <nrows>`: (Optional) Number of rows for debugging.

#### Development
Process the full dataset by omitting `-n`:
```sh
python data_preprocess.py -i ./data/input -o ./data/output
```

#### Debugging
For debugging with a subset, include `-n`:
```sh
python data_preprocess.py -i ./data/input -o ./data/output -n 100
```

### Output
Processed data is saved to the specified output directory.

