# import os
# import ast
# import pandas as pd
# import lp_data_anaylsis as lpda
# import sys
# import argparse

# def generate_report(
#     input_path,
#     groups,
#     group_1,
#     group_2,
#     output_path='results',
#     int_threshold=3000,
#     keep_cols=['index', 'precursor_mz', 'adduct', 'MS2_norm'],
#     p_value_threshold=0.5,
#     fc_threshold=0
# ):
#     """
#     Generates a lipidomics analysis report.

#     Parameters:
#     - input_path: Path to the CSV input file.
#     - groups: List of group prefixes (e.g., ['0h', '2h']).
#     - group_1: First group for differential analysis.
#     - group_2: Second group for differential analysis.
#     - output_path: Directory to save results (default: 'results').
#     - int_threshold: Intensity threshold (default: 3000).
#     - keep_cols: List of columns to keep for MS table (default: ['index', 'precursor_mz', 'adduct']).
#     - p_value_threshold: P-value threshold for volcano plot (default: 0.5).
#     - fc_threshold: Fold change threshold (default: 0).
#     """

#     # --- 1. Validation ---
#     print(f"Validating input: {input_path}")
    
#     # Check input file existence
#     if not os.path.exists(input_path):
#         raise FileNotFoundError(f"Input file not found: {input_path}")

#     # Read CSV
#     try:
#         df = pd.read_csv(input_path)
#     except Exception as e:
#         raise ValueError(f"Failed to read CSV file: {e}")

#     # Check required columns for data integrity
#     required_check_cols = ['chain_info', 'class', 'category']
#     missing_cols = [c for c in required_check_cols if c not in df.columns]
#     if missing_cols:
#         raise ValueError(f"Input CSV missing required columns: {missing_cols}")

#     # Check for at least 20 rows with non-empty values in required columns
#     valid_rows = df.dropna(subset=required_check_cols)
#     if len(valid_rows) < 20:
#         raise ValueError(f"Input file must have at least 20 rows with non-empty values in {required_check_cols}. Found {len(valid_rows)}.")

#     # Check groups column existence
#     # (Check if there is at least one column that starts with each group name)
#     df_columns = df.columns.tolist()
#     for g in groups:
#         if not any(col.startswith(g) for col in df_columns):
#             raise ValueError(f"No columns found starting with group prefix: '{g}'")

#     # Check group_1 and group_2 existence in groups list
#     if group_1 not in groups:
#         raise ValueError(f"group_1 '{group_1}' is not in the provided groups list: {groups}")
#     if group_2 not in groups:
#         raise ValueError(f"group_2 '{group_2}' is not in the provided groups list: {groups}")

#     print("Validation successful. Processing data...")

#     # --- 2. Setup Directories ---
#     # Define sub-folder for materials
#     materials_dir_name = 'report_materials'
#     materials_path = os.path.join(output_path, materials_dir_name)
    
#     os.makedirs(materials_path, exist_ok=True)

#     # --- 3. Data Preparation ---
    
#     # 1. Drop NA chain_info first
#     df = df.dropna(subset=['chain_info'])
    
#     # Put name in the first column if it exists
#     if 'name' in df.columns:
#         cols = ['name'] + [c for c in df.columns if c != 'name']
#         df = df[cols]

#     # Handle keep_cols for MS Table
#     if keep_cols is None:
#         ms_df = df.copy()
#     else:
#         existing_cols = [c for c in keep_cols if c in df.columns]
#         ms_df = df[existing_cols]

#     # Parse chain_info and Calculate metrics
#     print("Processing chain_info column...")
    
#     # 2. Convert chain_info value from string to list, drop rows that fail
#     def safe_convert_chain_info(x):
#         """Convert chain_info from string to list, return None if conversion fails"""
#         if isinstance(x, list):
#             return x
#         if isinstance(x, str):
#             try:
#                 result = ast.literal_eval(x)
#                 if isinstance(result, list):
#                     return result
#                 else:
#                     return None
#             except (ValueError, SyntaxError):
#                 return None
#         return None
    
#     # Apply conversion
#     df['chain_info_converted'] = df['chain_info'].apply(safe_convert_chain_info)
    
#     # Drop rows where conversion failed
#     rows_before = len(df)
#     df = df[df['chain_info_converted'].notna()]
#     rows_after = len(df)
#     if rows_before > rows_after:
#         print(f"Dropped {rows_before - rows_after} rows with invalid chain_info")
    
#     # Replace original chain_info with converted version
#     df['chain_info'] = df['chain_info_converted']
#     df = df.drop(columns=['chain_info_converted'])
    
#     # 3. Calculate 'length': biggest integer in chain_info list
#     def get_length(x):
#         """Extract the biggest integer from chain_info list"""
#         if isinstance(x, list) and len(x) > 0:
#             try:
#                 # Filter for numeric values and get max
#                 numeric_values = [val for val in x if isinstance(val, (int, float))]
#                 if numeric_values:
#                     return max(numeric_values)
#             except Exception:
#                 pass
#         return 0
    
#     df['length'] = df['chain_info'].apply(get_length)
    
#     # 4. Calculate 'unsaturation': sum of 2nd, 4th, 6th, and 8th values (indices 1, 3, 5, 7)
#     def get_unsaturation(x):
#         """Extract unsaturation as sum of 2nd, 4th, 6th, and 8th values"""
#         if isinstance(x, list) and len(x) >= 8:
#             try:
#                 indices = [1, 3, 5, 7]
#                 values = [x[i] for i in indices if i < len(x)]
#                 # Filter for numeric values
#                 numeric_values = [val for val in values if isinstance(val, (int, float))]
#                 return sum(numeric_values)
#             except Exception:
#                 pass
#         return 0
    
#     df['unsaturation'] = df['chain_info'].apply(get_unsaturation)
    
#     print(f"Successfully processed chain_info. Length and unsaturation columns created.")
#     print(f"Final dataset has {len(df)} rows.")

#     # --- 4. Analysis & Figure Generation ---
    
#     print("Running differential expression analysis...")
#     met_sig, met_data = lpda.analyze_differential_expression(
#         df=df,
#         group1_id=group_1,
#         group2_id=group_2,
#         fc_threshold=fc_threshold,
#         p_value_threshold=p_value_threshold
#     )

#     print("Calculating averages...")
#     met_avg = lpda.merge_and_average_columns(df, groups)

#     # --- FIX: Robustly Restore metadata columns to met_avg ---
#     # The averaging function drops non-numeric columns. We need to merge them back.
#     # We assume the index of met_avg corresponds to the identifier (name).
    
#     # 1. Ensure met_avg has a 'name' column for merging
#     if 'name' not in met_avg.columns:
#         met_avg = met_avg.reset_index()
#         # If the index didn't have a name 'name', the new column might be 'index' or 'level_0'
#         # We assume the first column is now the identifier
#         if 'name' not in met_avg.columns:
#             met_avg.rename(columns={met_avg.columns[0]: 'name'}, inplace=True)

#     # 2. Merge metadata from original df
#     # Columns we definitely need for plots
#     needed_cols = ['unsaturation', 'length', 'class', 'category', 'chain_info']
#     # Filter to what is actually available in the source
#     available_cols = [c for c in needed_cols if c in df.columns]
    
#     # Check which ones are still missing in met_avg
#     missing_in_avg = [c for c in available_cols if c not in met_avg.columns]
    
#     if missing_in_avg:
#         # Create a clean metadata lookup table
#         # We assume 'name' is the key. Drop duplicates to avoid 1:many merge issues.
#         meta_source = df[['name'] + missing_in_avg].drop_duplicates(subset=['name'])
#         met_avg = pd.merge(met_avg, meta_source, on='name', how='left')

#     print(f"met_avg columns available for plotting: {met_avg.columns.tolist()}")

#     print("Generating figures...")
    
#     # Helper to join path
#     def mat_path(filename):
#         return os.path.join(materials_path, filename)

#     # 1. Mass Spec Table
#     lpda.create_interactive_spectrogram_table(ms_df, file_path=mat_path('MS.html'))

#     # 2. Overall - Unsaturation & Length
#     lpda.create_violin(met_avg, groups, int_threshold, value_column='unsaturation', output_filename=mat_path('unsaturation.html'))
#     lpda.create_violin(met_avg, groups, int_threshold, value_column='length', output_filename=mat_path('length.html'))

#     # 3. Overall - Class (Sunburst)
#     try:
#         lpda.create_sunburst(df=met_avg, filename=mat_path('class.html'))
#     except TypeError:
#         print("Warning: Could not save generic class sunburst (check function signature).")

#     # 4. ThemeRiver
#     lpda.create_themeriver(
#         df=met_avg,
#         group=groups,
#         threshold=int_threshold,
#         filename=mat_path("class_themeriver.html")
#     )

#     # 5. PCA
#     lpda.create_pca(df=met_data, groups=groups, output_filename=mat_path("pca.html"))

#     # 6. Differential - Volcano
#     lpda.create_volcano(
#         df=met_data,
#         p_value_threshold=p_value_threshold,
#         fc_threshold=fc_threshold,
#         result_path=mat_path("volcano_plot.html")
#     )

#     # 7. Differential - Class, Unsaturation, Length
#     lpda.create_sunburst(met_sig, filename=mat_path('class_sig.html'))
#     lpda.create_violin(met_sig, column_list=None, threshold=0, value_column='unsaturation', output_filename=mat_path('unsaturation_sig.html'))
#     lpda.create_violin(met_sig, column_list=None, threshold=0, value_column='length', output_filename=mat_path('length_sig.html'))

#     # --- 5. Report Merging ---
#     print("Merging report...")
    
#     def rel_src(filename):
#         return f"{materials_dir_name}/{filename}"

#     # Generate HTML content
#     id_content = f'''<div class="iframe-container"><iframe src="{rel_src('MS.html')}" title="Spectrum" style="height: 800px;"></iframe></div>'''

#     trend_content_parts = []
#     trend_content_parts.append(create_plot_embed_html(rel_src('class_themeriver.html'), height='80vh', title='Class Themeriver'))
    
#     class_plot = create_plot_embed_html(rel_src('class.html'), title='Class')
#     pca_plot = create_plot_embed_html(rel_src('pca.html'), title='PCA')
    
#     trend_content_parts.append(f'''
#     <div class="horizontal-container">
#         {class_plot}
#         {pca_plot}
#     </div>''')
    
#     length_plot = create_plot_embed_html(rel_src('length.html'), title='Length')
#     unsaturation_plot = create_plot_embed_html(rel_src('unsaturation.html'), title='Unsaturation')
    
#     trend_content_parts.append(f'''
#     <div class="horizontal-container">
#         {length_plot}
#         {unsaturation_plot}
#     </div>''')
    
#     trend_content = "\n".join(trend_content_parts)

#     comparison_top = create_plot_embed_html(rel_src('volcano_plot.html'), height='600px', title='Volcano Plot')
    
#     comparison_bottom_parts = []
#     comparison_bottom_parts.append(f'''
#     <div class="plot-wrapper" style="padding-bottom: 10px;">
#         <h3>Class (Significant Lipids)</h3>
#         <iframe src="{rel_src('class_sig.html')}" title="Class (Significant Lipids)"></iframe>
#     </div>''')
    
#     comparison_bottom_parts.append(create_plot_embed_html(rel_src('length_sig.html'), title='Length (Significant Lipids)'))
#     comparison_bottom_parts.append(create_plot_embed_html(rel_src('unsaturation_sig.html'), title='Unsaturation (Significant Lipids)'))

#     comparison_bottom = "\n".join(comparison_bottom_parts)

#     # HTML Template
#     final_html = HTML_TEMPLATE.format(
#         identification_content=id_content,
#         trend_content=trend_content,
#         comparison_top_content=comparison_top,
#         comparison_bottom_content=comparison_bottom
#     )

#     final_report_path = os.path.join(output_path, 'report.html')
#     try:
#         with open(final_report_path, 'w', encoding='utf-8') as f:
#             f.write(final_html)
#         print(f"Success! Report generated at: {os.path.abspath(final_report_path)}")
#     except IOError as e:
#         print(f"Error writing report file: {e}")

# def create_plot_embed_html(src_path, height='450px', title="Plot"):
#     return f'''
# <div class="plot-wrapper">
#     <h3>{title}</h3>
#     <iframe src="{src_path}" title="{title}" style="height: {height};"></iframe>
# </div>'''

# # --- HTML TEMPLATE (Fixed: Double curly braces for CSS/JS) ---
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Lipidomics Report</title>
#     <style>
#         body {{
#             font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
#             margin: 0;
#             background-color: #f8f9fa;
#             color: #212529;
#             transition: background-color 0.3s, color 0.3s;
#         }}
#         .nav {{
#             background-color: #ffffff;
#             overflow: hidden;
#             border-bottom: 1px solid #dee2e6;
#             box-shadow: 0 2px 4px rgba(0,0,0,0.05);
#             position: sticky;
#             top: 0;
#             z-index: 1000;
#             transition: background-color 0.3s, border-bottom-color 0.3s;
#         }}
#         .nav button.tablinks {{
#             background-color: inherit;
#             float: left;
#             border: none;
#             outline: none;
#             cursor: pointer;
#             padding: 14px 20px;
#             transition: all 0.3s ease-in-out;
#             font-size: 16px;
#             font-weight: 500;
#             color: #495057;
#             border-bottom: 3px solid transparent;
#         }}
#         .nav button.tablinks:hover {{
#             background-color: #f1f3f5;
#             color: #0d6efd;
#         }}
#         .nav button.tablinks.active {{
#             color: #0d6efd;
#             border-bottom-color: #0d6efd;
#         }}
#         .tabcontent {{
#             display: none;
#             padding: 12px 24px;
#             animation: fadeIn 0.5s;
#         }}
#         @keyframes fadeIn {{
#             from {{ opacity: 0; transform: translateY(10px); }}
#             to {{ opacity: 1; transform: translateY(0); }}
#         }}
#         .content-wrapper {{
#             max-width: 1200px;
#             margin: 0 auto;
#         }}
#         .plot-wrapper {{
#             border: 1px solid #dee2e6;
#             border-radius: 8px;
#             overflow: hidden;
#             background-color: #ffffff;
#             box-shadow: 0 4px 6px rgba(0,0,0,0.05);
#             transition: background-color 0.3s, border-color 0.3s;
#         }}
#         .plot-wrapper h3 {{
#             margin: 0;
#             padding: 12px 16px;
#             background-color: #f8f9fa;
#             border-bottom: 1px solid #dee2e6;
#             font-size: 14px;
#             font-weight: 600;
#             transition: background-color 0.3s, border-bottom-color 0.3s;
#         }}
#         .iframe-container {{
#             border: 1px solid #dee2e6;
#             border-radius: 8px;
#             overflow: hidden;
#             transition: border-color 0.3s;
#         }}
#         iframe {{
#             width: 100%;
#             height: 450px;
#             border: none;
#             display: block;
#         }}
#         .vertical-container {{ display: flex; flex-direction: column; gap: 24px; }}
#         .horizontal-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; }}
#         .comparison-grid {{ gap: 40px; }}

#         /* Theme Toggle Button */
#         .theme-toggle-container {{
#             float: right;
#             display: flex;
#             align-items: center;
#             height: 57px;
#             padding-right: 15px;
#         }}
#         #theme-toggle-btn {{
#             cursor: pointer;
#             background-color: #e9ecef;
#             border: 1px solid #dee2e6;
#             color: #495057;
#             border-radius: 20px;
#             padding: 8px 16px;
#             font-size: 14px;
#             font-weight: 500;
#             transition: all 0.3s ease;
#         }}
#         #theme-toggle-btn:hover {{
#             background-color: #ced4da;
#         }}

#         /* Night Mode Theme */
#         body.dark-mode {{
#             background-color: #121212;
#             color: #e0e0e0;
#         }}
#         body.dark-mode .nav {{
#             background-color: #1e1e1e;
#             border-bottom: 1px solid #3a3a3a;
#         }}
#         body.dark-mode .nav button.tablinks {{
#             color: #bbbbbb;
#         }}
#         body.dark-mode .nav button.tablinks:hover {{
#             background-color: #333;
#             color: #ffffff;
#         }}
#         body.dark-mode .nav button.tablinks.active {{
#             color: #4dabf7;
#             border-bottom-color: #4dabf7;
#         }}
#         body.dark-mode .plot-wrapper {{
#             background-color: #1e1e1e;
#             border: 1px solid #3a3a3a;
#             box-shadow: 0 4px 6px rgba(0,0,0,0.2);
#         }}
#         body.dark-mode .plot-wrapper h3 {{
#             background-color: #252525;
#             border-bottom: 1px solid #3a3a3a;
#         }}
#         body.dark-mode .iframe-container {{
#             border-color: #3a3a3a;
#         }}
#         body.dark-mode #theme-toggle-btn {{
#             background-color: #333;
#             color: #e0e0e0;
#             border-color: #555;
#         }}
#         body.dark-mode #theme-toggle-btn:hover {{
#             background-color: #444;
#         }}

#         #Identification {{ display: block; }}
#     </style>
# </head>
# <body>

# <div class="nav">
#     <div class="content-wrapper">
#         <div class="theme-toggle-container">
#             <button id="theme-toggle-btn" onclick="toggleTheme()">Night Mode</button>
#         </div>
#         <button class="tablinks active" onclick="openTab(event, 'Identification')">Identification</button>
#         <button class="tablinks" onclick="openTab(event, 'Trend')">Trend</button>
#         <button class="tablinks" onclick="openTab(event, 'Comparison')">Comparison</button>
#     </div>
# </div>

# <div class="content-wrapper">
#     <div id="Identification" class="tabcontent">
#         {identification_content}
#     </div>

#     <div id="Trend" class="tabcontent">
#         <div class="vertical-container">
#             {trend_content}
#         </div>
#     </div>

#     <div id="Comparison" class="tabcontent">
#         <div class="vertical-container">
#             {comparison_top_content}
#             <div class="horizontal-container comparison-grid">
#                 {comparison_bottom_content}
#             </div>
#         </div>
#     </div>
# </div>

# <script>
# function openTab(evt, tabName) {{
#     let i, tabcontent, tablinks;
#     tabcontent = document.getElementsByClassName("tabcontent");
#     for (i = 0; i < tabcontent.length; i++) {{
#         tabcontent[i].style.display = "none";
#     }}
#     tablinks = document.getElementsByClassName("tablinks");
#     for (i = 0; i < tablinks.length; i++) {{
#         tablinks[i].classList.remove("active");
#     }}
#     document.getElementById(tabName).style.display = "block";
#     evt.currentTarget.classList.add("active");
# }}

# function toggleTheme() {{
#     document.body.classList.toggle('dark-mode');
#     const isDarkMode = document.body.classList.contains('dark-mode');
#     localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
#     document.getElementById('theme-toggle-btn').textContent = isDarkMode ? 'Light Mode' : 'Night Mode';
# }}

# document.addEventListener('DOMContentLoaded', () => {{
#     if (localStorage.getItem('theme') === 'dark') {{
#         document.body.classList.add('dark-mode');
#         document.getElementById('theme-toggle-btn').textContent = 'Light Mode';
#     }}
# }});
# </script>

# </body>
# </html>
# """

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate Lipidomics Analysis Report")
    
#     # Required args
#     parser.add_argument("--input_path", required=True, help="Path to the CSV input file")
#     parser.add_argument("--groups", required=True, nargs='+', help="List of group prefixes (e.g. 0h 2h 4h)")
#     parser.add_argument("--group_1", required=True, help="First group for differential analysis")
#     parser.add_argument("--group_2", required=True, help="Second group for differential analysis")
    
#     # Optional args
#     parser.add_argument("--output_path", default="results", help="Directory to save results")
#     parser.add_argument("--int_threshold", type=int, default=3000, help="Intensity threshold")
#     parser.add_argument("--keep_cols", nargs='+', default=['index', 'name', 'precursor_mz', 'adduct', 'MS2_norm'], help="List of columns to keep for MS table")
#     parser.add_argument("--p_value_threshold", type=float, default=0.05, help="P-value threshold")
#     parser.add_argument("--fc_threshold", type=float, default=1.2, help="Fold change threshold")

#     args = parser.parse_args()

#     try:
#         generate_report(
#             input_path=args.input_path,
#             groups=args.groups,
#             group_1=args.group_1,
#             group_2=args.group_2,
#             output_path=args.output_path,
#             int_threshold=args.int_threshold,
#             keep_cols=args.keep_cols,
#             p_value_threshold=args.p_value_threshold,
#             fc_threshold=args.fc_threshold
#         )
#     except Exception as e:
#         print(f"Execution Error: {e}")























import os
import ast
import pandas as pd
import lp_data_anaylsis as lpda
import sys
import argparse

def generate_report(
    input_path,
    groups,
    group_1,
    group_2,
    output_path='results',
    int_threshold=3000,
    keep_cols=['index', 'precursor_mz', 'adduct', 'MS2_norm'],
    p_value_threshold=0.5,
    fc_threshold=0
):
    """
    Generates a lipidomics analysis report.

    Parameters:
    - input_path: Path to the CSV input file.
    - groups: List of group prefixes (e.g., ['0h', '2h']).
    - group_1: First group for differential analysis.
    - group_2: Second group for differential analysis.
    - output_path: Directory to save results (default: 'results').
    - int_threshold: Intensity threshold (default: 3000).
    - keep_cols: List of columns to keep for MS table (default: ['index', 'precursor_mz', 'adduct']).
    - p_value_threshold: P-value threshold for volcano plot (default: 0.5).
    - fc_threshold: Fold change threshold (default: 0).
    """

    # --- 1. Validation ---
    print(f"Validating input: {input_path}")
    
    # Check input file existence
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read CSV
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # --- MODIFICATION START: Handle missing chain_info ---
    if 'chain_info' not in df.columns:
        if 'plsf_rank1' in df.columns:
            print("Notice: 'chain_info' column missing. Using 'plsf_rank1' as substitute.")
            df['chain_info'] = df['plsf_rank1']
        else:
            # We won't raise here, we let the standard check below catch it
            # so the error lists all missing columns consistently.
            pass
    # --- MODIFICATION END ---

    # Check required columns for data integrity
    required_check_cols = ['chain_info', 'class', 'category']
    missing_cols = [c for c in required_check_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV missing required columns: {missing_cols}. (Note: If 'chain_info' is missing, ensure 'plsf_rank1' is present).")

    # Check for at least 20 rows with non-empty values in required columns
    valid_rows = df.dropna(subset=required_check_cols)
    if len(valid_rows) < 20:
        raise ValueError(f"Input file must have at least 20 rows with non-empty values in {required_check_cols}. Found {len(valid_rows)}.")

    # Check groups column existence
    # (Check if there is at least one column that starts with each group name)
    df_columns = df.columns.tolist()
    for g in groups:
        if not any(col.startswith(g) for col in df_columns):
            raise ValueError(f"No columns found starting with group prefix: '{g}'")

    # Check group_1 and group_2 existence in groups list
    if group_1 not in groups:
        raise ValueError(f"group_1 '{group_1}' is not in the provided groups list: {groups}")
    if group_2 not in groups:
        raise ValueError(f"group_2 '{group_2}' is not in the provided groups list: {groups}")

    print("Validation successful. Processing data...")

    # --- 2. Setup Directories ---
    # Define sub-folder for materials
    materials_dir_name = 'report_materials'
    materials_path = os.path.join(output_path, materials_dir_name)
    
    os.makedirs(materials_path, exist_ok=True)

    # --- 3. Data Preparation ---
    
    # 1. Drop NA chain_info first
    df = df.dropna(subset=['chain_info'])
    
    # Put name in the first column if it exists
    if 'name' in df.columns:
        cols = ['name'] + [c for c in df.columns if c != 'name']
        df = df[cols]

    # Handle keep_cols for MS Table
    if keep_cols is None:
        ms_df = df.copy()
    else:
        existing_cols = [c for c in keep_cols if c in df.columns]
        ms_df = df[existing_cols]

    # Parse chain_info and Calculate metrics
    print("Processing chain_info column...")
    
    # 2. Convert chain_info value from string to list, drop rows that fail
    def safe_convert_chain_info(x):
        """Convert chain_info from string to list, return None if conversion fails"""
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                result = ast.literal_eval(x)
                if isinstance(result, list):
                    return result
                else:
                    return None
            except (ValueError, SyntaxError):
                return None
        return None
    
    # Apply conversion
    df['chain_info_converted'] = df['chain_info'].apply(safe_convert_chain_info)
    
    # Drop rows where conversion failed
    rows_before = len(df)
    df = df[df['chain_info_converted'].notna()]
    rows_after = len(df)
    if rows_before > rows_after:
        print(f"Dropped {rows_before - rows_after} rows with invalid chain_info")
    
    # Replace original chain_info with converted version
    df['chain_info'] = df['chain_info_converted']
    df = df.drop(columns=['chain_info_converted'])
    
    # 3. Calculate 'length': biggest integer in chain_info list
    def get_length(x):
        """Extract the biggest integer from chain_info list"""
        if isinstance(x, list) and len(x) > 0:
            try:
                # Filter for numeric values and get max
                numeric_values = [val for val in x if isinstance(val, (int, float))]
                if numeric_values:
                    return max(numeric_values)
            except Exception:
                pass
        return 0
    
    df['length'] = df['chain_info'].apply(get_length)
    
    # 4. Calculate 'unsaturation': sum of 2nd, 4th, 6th, and 8th values (indices 1, 3, 5, 7)
    def get_unsaturation(x):
        """Extract unsaturation as sum of 2nd, 4th, 6th, and 8th values"""
        if isinstance(x, list) and len(x) >= 8:
            try:
                indices = [1, 3, 5, 7]
                values = [x[i] for i in indices if i < len(x)]
                # Filter for numeric values
                numeric_values = [val for val in values if isinstance(val, (int, float))]
                return sum(numeric_values)
            except Exception:
                pass
        return 0
    
    df['unsaturation'] = df['chain_info'].apply(get_unsaturation)
    
    print(f"Successfully processed chain_info. Length and unsaturation columns created.")
    print(f"Final dataset has {len(df)} rows.")

    # --- 4. Analysis & Figure Generation ---
    
    print("Running differential expression analysis...")
    met_sig, met_data = lpda.analyze_differential_expression(
        df=df,
        group1_id=group_1,
        group2_id=group_2,
        fc_threshold=fc_threshold,
        p_value_threshold=p_value_threshold
    )

    print("Calculating averages...")
    met_avg = lpda.merge_and_average_columns(df, groups)

    # --- FIX: Robustly Restore metadata columns to met_avg ---
    # The averaging function drops non-numeric columns. We need to merge them back.
    # We assume the index of met_avg corresponds to the identifier (name).
    
    # 1. Ensure met_avg has a 'name' column for merging
    if 'name' not in met_avg.columns:
        met_avg = met_avg.reset_index()
        # If the index didn't have a name 'name', the new column might be 'index' or 'level_0'
        # We assume the first column is now the identifier
        if 'name' not in met_avg.columns:
            met_avg.rename(columns={met_avg.columns[0]: 'name'}, inplace=True)

    # 2. Merge metadata from original df
    # Columns we definitely need for plots
    needed_cols = ['unsaturation', 'length', 'class', 'category', 'chain_info']
    # Filter to what is actually available in the source
    available_cols = [c for c in needed_cols if c in df.columns]
    
    # Check which ones are still missing in met_avg
    missing_in_avg = [c for c in available_cols if c not in met_avg.columns]
    
    if missing_in_avg:
        # Create a clean metadata lookup table
        # We assume 'name' is the key. Drop duplicates to avoid 1:many merge issues.
        meta_source = df[['name'] + missing_in_avg].drop_duplicates(subset=['name'])
        met_avg = pd.merge(met_avg, meta_source, on='name', how='left')

    print(f"met_avg columns available for plotting: {met_avg.columns.tolist()}")

    print("Generating figures...")
    
    # Helper to join path
    def mat_path(filename):
        return os.path.join(materials_path, filename)

    # 1. Mass Spec Table
    lpda.create_interactive_spectrogram_table(ms_df, file_path=mat_path('MS.html'))

    # 2. Overall - Unsaturation & Length
    lpda.create_violin(met_avg, groups, int_threshold, value_column='unsaturation', output_filename=mat_path('unsaturation.html'))
    lpda.create_violin(met_avg, groups, int_threshold, value_column='length', output_filename=mat_path('length.html'))

    # 3. Overall - Class (Sunburst)
    try:
        lpda.create_sunburst(df=met_avg, filename=mat_path('class.html'))
    except TypeError:
        print("Warning: Could not save generic class sunburst (check function signature).")

    # 4. ThemeRiver
    lpda.create_themeriver(
        df=met_avg,
        group=groups,
        threshold=int_threshold,
        filename=mat_path("class_themeriver.html")
    )

    # 5. PCA
    lpda.create_pca(df=met_data, groups=groups, output_filename=mat_path("pca.html"))

    # 6. Differential - Volcano
    lpda.create_volcano(
        df=met_data,
        p_value_threshold=p_value_threshold,
        fc_threshold=fc_threshold,
        result_path=mat_path("volcano_plot.html")
    )

    # 7. Differential - Class, Unsaturation, Length
    lpda.create_sunburst(met_sig, filename=mat_path('class_sig.html'))
    lpda.create_violin(met_sig, column_list=None, threshold=0, value_column='unsaturation', output_filename=mat_path('unsaturation_sig.html'))
    lpda.create_violin(met_sig, column_list=None, threshold=0, value_column='length', output_filename=mat_path('length_sig.html'))

    # --- 5. Report Merging ---
    print("Merging report...")
    
    def rel_src(filename):
        return f"{materials_dir_name}/{filename}"

    # Generate HTML content
    id_content = f'''<div class="iframe-container"><iframe src="{rel_src('MS.html')}" title="Spectrum" style="height: 800px;"></iframe></div>'''

    trend_content_parts = []
    trend_content_parts.append(create_plot_embed_html(rel_src('class_themeriver.html'), height='80vh', title='Class Themeriver'))
    
    class_plot = create_plot_embed_html(rel_src('class.html'), title='Class')
    pca_plot = create_plot_embed_html(rel_src('pca.html'), title='PCA')
    
    trend_content_parts.append(f'''
    <div class="horizontal-container">
        {class_plot}
        {pca_plot}
    </div>''')
    
    length_plot = create_plot_embed_html(rel_src('length.html'), title='Length')
    unsaturation_plot = create_plot_embed_html(rel_src('unsaturation.html'), title='Unsaturation')
    
    trend_content_parts.append(f'''
    <div class="horizontal-container">
        {length_plot}
        {unsaturation_plot}
    </div>''')
    
    trend_content = "\n".join(trend_content_parts)

    comparison_top = create_plot_embed_html(rel_src('volcano_plot.html'), height='600px', title='Volcano Plot')
    
    comparison_bottom_parts = []
    comparison_bottom_parts.append(f'''
    <div class="plot-wrapper" style="padding-bottom: 10px;">
        <h3>Class (Significant Lipids)</h3>
        <iframe src="{rel_src('class_sig.html')}" title="Class (Significant Lipids)"></iframe>
    </div>''')
    
    comparison_bottom_parts.append(create_plot_embed_html(rel_src('length_sig.html'), title='Length (Significant Lipids)'))
    comparison_bottom_parts.append(create_plot_embed_html(rel_src('unsaturation_sig.html'), title='Unsaturation (Significant Lipids)'))

    comparison_bottom = "\n".join(comparison_bottom_parts)

    # HTML Template
    final_html = HTML_TEMPLATE.format(
        identification_content=id_content,
        trend_content=trend_content,
        comparison_top_content=comparison_top,
        comparison_bottom_content=comparison_bottom
    )

    final_report_path = os.path.join(output_path, 'report.html')
    try:
        with open(final_report_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        print(f"Success! Report generated at: {os.path.abspath(final_report_path)}")
    except IOError as e:
        print(f"Error writing report file: {e}")

def create_plot_embed_html(src_path, height='450px', title="Plot"):
    return f'''
<div class="plot-wrapper">
    <h3>{title}</h3>
    <iframe src="{src_path}" title="{title}" style="height: {height};"></iframe>
</div>'''

# --- HTML TEMPLATE (Fixed: Double curly braces for CSS/JS) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lipidomics Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            background-color: #f8f9fa;
            color: #212529;
            transition: background-color 0.3s, color 0.3s;
        }}
        .nav {{
            background-color: #ffffff;
            overflow: hidden;
            border-bottom: 1px solid #dee2e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            position: sticky;
            top: 0;
            z-index: 1000;
            transition: background-color 0.3s, border-bottom-color 0.3s;
        }}
        .nav button.tablinks {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 20px;
            transition: all 0.3s ease-in-out;
            font-size: 16px;
            font-weight: 500;
            color: #495057;
            border-bottom: 3px solid transparent;
        }}
        .nav button.tablinks:hover {{
            background-color: #f1f3f5;
            color: #0d6efd;
        }}
        .nav button.tablinks.active {{
            color: #0d6efd;
            border-bottom-color: #0d6efd;
        }}
        .tabcontent {{
            display: none;
            padding: 12px 24px;
            animation: fadeIn 0.5s;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .content-wrapper {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .plot-wrapper {{
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
            background-color: #ffffff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: background-color 0.3s, border-color 0.3s;
        }}
        .plot-wrapper h3 {{
            margin: 0;
            padding: 12px 16px;
            background-color: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            font-size: 14px;
            font-weight: 600;
            transition: background-color 0.3s, border-bottom-color 0.3s;
        }}
        .iframe-container {{
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
            transition: border-color 0.3s;
        }}
        iframe {{
            width: 100%;
            height: 450px;
            border: none;
            display: block;
        }}
        .vertical-container {{ display: flex; flex-direction: column; gap: 24px; }}
        .horizontal-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; }}
        .comparison-grid {{ gap: 40px; }}

        /* Theme Toggle Button */
        .theme-toggle-container {{
            float: right;
            display: flex;
            align-items: center;
            height: 57px;
            padding-right: 15px;
        }}
        #theme-toggle-btn {{
            cursor: pointer;
            background-color: #e9ecef;
            border: 1px solid #dee2e6;
            color: #495057;
            border-radius: 20px;
            padding: 8px 16px;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        #theme-toggle-btn:hover {{
            background-color: #ced4da;
        }}

        /* Night Mode Theme */
        body.dark-mode {{
            background-color: #121212;
            color: #e0e0e0;
        }}
        body.dark-mode .nav {{
            background-color: #1e1e1e;
            border-bottom: 1px solid #3a3a3a;
        }}
        body.dark-mode .nav button.tablinks {{
            color: #bbbbbb;
        }}
        body.dark-mode .nav button.tablinks:hover {{
            background-color: #333;
            color: #ffffff;
        }}
        body.dark-mode .nav button.tablinks.active {{
            color: #4dabf7;
            border-bottom-color: #4dabf7;
        }}
        body.dark-mode .plot-wrapper {{
            background-color: #1e1e1e;
            border: 1px solid #3a3a3a;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }}
        body.dark-mode .plot-wrapper h3 {{
            background-color: #252525;
            border-bottom: 1px solid #3a3a3a;
        }}
        body.dark-mode .iframe-container {{
            border-color: #3a3a3a;
        }}
        body.dark-mode #theme-toggle-btn {{
            background-color: #333;
            color: #e0e0e0;
            border-color: #555;
        }}
        body.dark-mode #theme-toggle-btn:hover {{
            background-color: #444;
        }}

        #Identification {{ display: block; }}
    </style>
</head>
<body>

<div class="nav">
    <div class="content-wrapper">
        <div class="theme-toggle-container">
            <button id="theme-toggle-btn" onclick="toggleTheme()">Night Mode</button>
        </div>
        <button class="tablinks active" onclick="openTab(event, 'Identification')">Identification</button>
        <button class="tablinks" onclick="openTab(event, 'Trend')">Trend</button>
        <button class="tablinks" onclick="openTab(event, 'Comparison')">Comparison</button>
    </div>
</div>

<div class="content-wrapper">
    <div id="Identification" class="tabcontent">
        {identification_content}
    </div>

    <div id="Trend" class="tabcontent">
        <div class="vertical-container">
            {trend_content}
        </div>
    </div>

    <div id="Comparison" class="tabcontent">
        <div class="vertical-container">
            {comparison_top_content}
            <div class="horizontal-container comparison-grid">
                {comparison_bottom_content}
            </div>
        </div>
    </div>
</div>

<script>
function openTab(evt, tabName) {{
    let i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {{
        tabcontent[i].style.display = "none";
    }}
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {{
        tablinks[i].classList.remove("active");
    }}
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.classList.add("active");
}}

function toggleTheme() {{
    document.body.classList.toggle('dark-mode');
    const isDarkMode = document.body.classList.contains('dark-mode');
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
    document.getElementById('theme-toggle-btn').textContent = isDarkMode ? 'Light Mode' : 'Night Mode';
}}

document.addEventListener('DOMContentLoaded', () => {{
    if (localStorage.getItem('theme') === 'dark') {{
        document.body.classList.add('dark-mode');
        document.getElementById('theme-toggle-btn').textContent = 'Light Mode';
    }}
}});
</script>

</body>
</html>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Lipidomics Analysis Report")
    
    # Required args
    parser.add_argument("--input_path", required=True, help="Path to the CSV input file")
    parser.add_argument("--groups", required=True, nargs='+', help="List of group prefixes (e.g. 0h 2h 4h)")
    parser.add_argument("--group_1", required=True, help="First group for differential analysis")
    parser.add_argument("--group_2", required=True, help="Second group for differential analysis")
    
    # Optional args
    parser.add_argument("--output_path", default="results", help="Directory to save results")
    parser.add_argument("--int_threshold", type=int, default=3000, help="Intensity threshold")
    parser.add_argument("--keep_cols", nargs='+', default=['index', 'name', 'precursor_mz', 'adduct', 'MS2_norm'], help="List of columns to keep for MS table")
    parser.add_argument("--p_value_threshold", type=float, default=0.05, help="P-value threshold")
    parser.add_argument("--fc_threshold", type=float, default=1.2, help="Fold change threshold")

    args = parser.parse_args()

    try:
        generate_report(
            input_path=args.input_path,
            groups=args.groups,
            group_1=args.group_1,
            group_2=args.group_2,
            output_path=args.output_path,
            int_threshold=args.int_threshold,
            keep_cols=args.keep_cols,
            p_value_threshold=args.p_value_threshold,
            fc_threshold=args.fc_threshold
        )
    except Exception as e:
        print(f"Execution Error: {e}")