import pandas as pd


def parse_chain_string(input_string: str):
    """
    Parses a chain string into its component blocks and sub-blocks.

    The string is first split by '_'. Then, each block is parsed according
    to the following rules:
    - num_c: The value before the first ':'
    - num_db: The value after the first ':' and before the first ';'
    - extra: The value after the first ';'

    The resulting list of blocks is sorted numerically:
    1. By num_c (descending, bigger first)
    2. By num_db (descending, bigger first)

    Args:
        input_string: The chain string to parse (e.g., "9:0;2O_32:8").

    Returns:
        A tuple containing two elements:
        1. (list[dict]): A sorted list of dictionaries, where each
           dictionary represents a block and contains 'num_c', 'num_db',
           and 'extra'.
        2. (int): The count of '_' characters in the input string (num_tail).
    """
    
    def _safe_int_convert_desc(value: str | None) -> float | int:
        """
        Helper to convert string to int for sorting (descending).
        Non-numeric values are treated as -infinity so they sort to the end.
        """
        try:
            # Attempt to convert to integer
            return int(value)
        except (ValueError, TypeError):
            # If it fails (None, empty string, or non-numeric string),
            # return -infinity so it sorts to the end of a descending sort.
            return float('-inf')

    all_blocks_data = []
    
    # 1. Split the input string into 4 blocks (or more/less)
    blocks = input_string.split('_')

    # 2. Process each block
    for block in blocks:
        # Initialize sub-block values to None
        num_c = None
        num_db = None
        extra = None

        # 3. Separate the 'extra' part first by splitting at the first ';'
        # We use split(';', 1) to only split at the first occurrence.
        semicolon_parts = block.split(';', 1)
        
        # The part before the semicolon (or the whole string if no ';')
        before_semicolon = semicolon_parts[0]

        if len(semicolon_parts) == 2:
            # If a ';' was found, the second part is 'extra'
            extra = semicolon_parts[1]

        # 4. Separate 'num_c' and 'num_db' from the 'before_semicolon' part
        # We use split(':', 1) to only split at the first occurrence.
        colon_parts = before_semicolon.split(':', 1)

        # The part before the colon is always 'num_c'
        # This will be the whole 'before_semicolon' string if no ':' exists
        num_c = colon_parts[0]

        if len(colon_parts) == 2:
            # If a ':' was found, the second part is 'num_db'
            num_db = colon_parts[1]

        # 5. Store the results for this block
        block_data = {
            'num_c': num_c,
            'num_db': num_db,
            'extra': extra
        }
        all_blocks_data.append(block_data)

    # 6. Sort the results
    # We sort by num_c (descending) and then num_db (descending).
    # We use a helper that returns -inf for non-numeric values and then
    # sort the entire list in reverse (descending) order.
    all_blocks_data.sort(key=lambda block: (
        _safe_int_convert_desc(block['num_c']), # Primary key for descending
        _safe_int_convert_desc(block['num_db'])  # Secondary key for descending
    ), reverse=True)

    # 7. Generate num_tail
    num_tail = input_string.count('_') + 1

    return all_blocks_data, num_tail


def process_chain_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame with a 'name' column.

    Splits the 'name' column by the first space. The part before the space
    becomes the 'class' column.
    
    The part after the space is treated as the chain string.
    Special handling is applied to the chain string:
    - If it starts with 'O-', '-O' is appended to the class, and 'O-' is removed from the chain.
    - If it starts with 'P-', '-P' is appended to the class, and 'P-' is removed from the chain.
    - If it starts with 'd', '-d' is appended to the class, and 'd' is removed from the chain.
    - If it starts with 't', '-t' is appended to the class, and 't' is removed from the chain.

    Applies the parse_chain_string function to the extracted chain string and
    expands the first 4 sorted blocks and num_tail into new columns.
    
    Also adds 'num_c' and 'num_db' columns which represent the sum of 
    carbons and double bonds across ALL blocks found in the chain string.

    Args:
        df: A pandas DataFrame which must contain a 'name' column.

    Returns:
        The modified pandas DataFrame with new columns added.
        Returns the original DataFrame if 'name' column is missing.
    """
    if 'name' not in df.columns:
        print("Error: DataFrame must have a 'name' column.")
        return df

    # 1. Split 'name' into 'class' and 'chain' parts
    # split by the first space only (n=1)
    split_data = df['name'].astype(str).str.split(' ', n=1, expand=True)
    
    # The part before the first space is the 'class'
    df['class'] = split_data[0]
    
    # The part after the first space is the chain string.
    # If no space exists, index 1 might not exist or be None/NaN.
    # We verify the column exists and strip whitespace to ensure clean prefix detection.
    if 1 in split_data.columns:
        chain_series = split_data[1].fillna('').astype(str).str.strip()
    else:
        chain_series = pd.Series([''] * len(df), index=df.index)

    # --- HANDLE PREFIXES ---
    # Note: We must update the class column first before removing the prefix from the chain string.
    
    # 1. Handle O- (Ether)
    mask_o = chain_series.str.startswith('O-')
    df.loc[mask_o, 'class'] = df.loc[mask_o, 'class'] + '-O'
    chain_series = chain_series.str.replace(r'^O-', '', regex=True)

    # 2. Handle P- (Plasmalogen)
    mask_p = chain_series.str.startswith('P-')
    df.loc[mask_p, 'class'] = df.loc[mask_p, 'class'] + '-P'
    chain_series = chain_series.str.replace(r'^P-', '', regex=True)

    # 3. Handle d (Dihydroxy base)
    # Check for 'd' at the start of the chain string (after O-/P- removal)
    mask_d = chain_series.str.startswith('d')
    # Move the 'd' indicator to the class
    df.loc[mask_d, 'class'] = df.loc[mask_d, 'class'] + '-d'
    # Remove 'd' from chain string so parsing logic works on the numbers (e.g., '17:1')
    chain_series = chain_series.str.replace(r'^d', '', regex=True)

    # 4. Handle t (Trihydroxy base)
    # Check for 't' at the start of the chain string
    mask_t = chain_series.str.startswith('t')
    # Move the 't' indicator to the class
    df.loc[mask_t, 'class'] = df.loc[mask_t, 'class'] + '-t'
    # Remove 't' from chain string
    chain_series = chain_series.str.replace(r'^t', '', regex=True)

    # 2. Apply the parsing function to the extracted chain string
    # This creates a Series of tuples: (sorted_blocks_list, num_tail)
    parsed_results = chain_series.apply(parse_chain_string)

    # 3. Extract num_tail
    # The second element (index 1) of the tuple is num_tail
    df['num_tail'] = parsed_results.apply(lambda x: x[1])

    # 4. Extract the sorted blocks list
    # The first element (index 0) of the tuple is the list of blocks
    blocks_series = parsed_results.apply(lambda x: x[0])

    # 5. Create new columns for each of the top 4 blocks
    for i in range(4):
        block_num = i + 1 # To get 1, 2, 3, 4
        
        # Helper lambda to get a block's value, or None if block doesn't exist
        get_block_val = lambda blocks, key: blocks[i][key] if len(blocks) > i else None

        # Get the raw series (can contain strings, None, or empty strings)
        num_c_series = blocks_series.apply(get_block_val, key='num_c')
        num_db_series = blocks_series.apply(get_block_val, key='num_db')
        
        # Convert to numeric. 'coerce' turns non-numeric values (like None, '') into NaN.
        # Then, fillna(0) replaces all NaN values (from None, '', etc.) with 0.
        df[f'num_c_{block_num}'] = pd.to_numeric(num_c_series, errors='coerce').fillna(0)
        df[f'num_db_{block_num}'] = pd.to_numeric(num_db_series, errors='coerce').fillna(0)
        
        # 'extra' column remains as is (can be string or None)
        df[f'extra_{block_num}'] = blocks_series.apply(get_block_val, key='extra')

    # 6. Calculate Total num_c and num_db across ALL blocks
    # We define a helper to sum values from the list of dictionaries safely
    def calculate_total(blocks, key):
        total = 0
        for block in blocks:
            val = block.get(key)
            try:
                # Convert to float to handle numeric strings, then add.
                # If it's None or non-numeric, it goes to except block.
                total += float(val)
            except (ValueError, TypeError):
                continue
        return total

    df['num_c'] = blocks_series.apply(lambda blocks: calculate_total(blocks, 'num_c'))
    df['num_db'] = blocks_series.apply(lambda blocks: calculate_total(blocks, 'num_db'))

    return df


def generate_chain_info_from_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where 'chain_info' is empty, generates the chain_info list
    from the 'name' column using the chain parsing logic.
    
    The chain_info is a list of 8 values:
    [num_c_1, num_db_1, num_c_2, num_db_2, num_c_3, num_db_3, num_c_4, num_db_4]
    
    Example:
        - "Cer 19:2;2O_2:0" → [19, 2, 2, 0, 0, 0, 0, 0]
        - "TG 9:0_9:0_9:0" → [9, 0, 9, 0, 9, 0, 0, 0]
    
    Args:
        df: DataFrame with 'name' and 'chain_info' columns
        
    Returns:
        Modified DataFrame with chain_info filled in where it was empty
    """
    if 'name' not in df.columns or 'chain_info' not in df.columns:
        print("Error: DataFrame must have 'name' and 'chain_info' columns.")
        return df
    
    # Identify rows where chain_info is empty
    def is_empty_chain_info(val):
        """Check if chain_info value is empty (None, NaN, empty list, etc.)"""
        if val is None:
            return True
        if isinstance(val, float) and pd.isna(val):
            return True
        if isinstance(val, (list, tuple)) and len(val) == 0:
            return True
        if isinstance(val, str) and val.strip() == '':
            return True
        return False
    
    empty_mask = df['chain_info'].apply(is_empty_chain_info)
    empty_count = empty_mask.sum()
    
    print(f"Detected {empty_count} features with empty chain_info")
    
    if empty_count == 0:
        # No empty chain_info rows, return as is
        print("Finished - no empty chain_info to process")
        return df
    
    # Process the rows with empty chain_info using the existing parsing function
    df_to_process = df[empty_mask].copy()
    df_processed = process_chain_dataframe(df_to_process)
    
    # Generate chain_info list from the extracted columns
    def create_chain_info_list(row):
        """Extract chain info values into a list of 8 elements"""
        return [
            int(row.get('num_c_1', 0)),
            int(row.get('num_db_1', 0)),
            int(row.get('num_c_2', 0)),
            int(row.get('num_db_2', 0)),
            int(row.get('num_c_3', 0)),
            int(row.get('num_db_3', 0)),
            int(row.get('num_c_4', 0)),
            int(row.get('num_db_4', 0))
        ]
    
    # Apply the function to create chain_info lists
    chain_info_lists = df_processed.apply(create_chain_info_list, axis=1)
    
    # Assign the generated chain_info back to the original dataframe
    df.loc[empty_mask, 'chain_info'] = chain_info_lists.values
    
    print(f"Successfully generated chain_info for {empty_count} features")
    print("Finished")
    
    return df


if __name__ == "__main__":
    print("Chain parser module loaded successfully.")
    print("Use generate_chain_info_from_name(df) to fill empty chain_info from name column.")
